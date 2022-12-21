import cv2
import glob
import time
import tqdm
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import settings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_preparation(data_folder):
    """Read data from folder and create Pandas DataFrames for train/valid"""
    # Create DataFrame
    images_list = sorted(glob.glob(f'{data_folder}/train/*'))
    masks_list = sorted(glob.glob(f'{data_folder}/masks/*'))
    assert len(images_list) == len(masks_list)
    df = pd.DataFrame()
    df['images'] = images_list
    df['masks'] = masks_list
    df['is_smt'] = df['masks'].apply(lambda x: 1 if np.any(cv2.imread(x)) else 0)
    df = df.loc[df['is_smt'] == 1]
    return df


class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.images_files = data['images'].tolist()
        self.masks_files = data['masks'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        # Select on image-mask couple
        image_path = self.images_files[index]
        mask_path = self.masks_files[index]
        # Image processing
        image_full = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_full = image_full.astype(np.uint8)
        # Maks processing
        mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_full = np.expand_dims(mask_full, axis=-1)
        mask_full = mask_full.astype(np.uint8)

        image = np.zeros((settings.IMAGE_SIZE, settings.IMAGE_SIZE, image_full.shape[-1]))
        mask = np.zeros((settings.IMAGE_SIZE, settings.IMAGE_SIZE, mask_full.shape[-1]))
        while not np.any(mask):
            start_x = random.randint(0, max(image_full.shape) - settings.IMAGE_SIZE)
            start_y = random.randint(0, max(image_full.shape) - settings.IMAGE_SIZE)
            image = image_full[start_x:start_x+settings.IMAGE_SIZE, start_y:start_y+settings.IMAGE_SIZE, :]
            mask = mask_full[start_x:start_x+settings.IMAGE_SIZE, start_y:start_y+settings.IMAGE_SIZE, :]

        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        return image, mask


def batch_image_mask_show(dataloader, number_of_images=5):
    """Plot samples after augmentation"""
    images, masks = next(iter(dataloader))
    images = images.numpy().transpose(0, 2, 3, 1)
    masks = masks.numpy()

    fig = plt.figure(figsize=(20, 5))
    for i in range(number_of_images):
        image = settings.STD * images[i] + settings.MEAN
        image = image * 255
        image = image.astype(np.uint8)
        mask = masks[i][0]
        mask = mask.astype(np.uint8)

        fig.add_subplot(1, number_of_images + 1, i + 1)
        plt.imshow(image)
        plt.imshow(mask, alpha=.3, cmap='gnuplot')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_one_image(image, mask, epoch, index=0, path_to_save='./results'):
    # Image preprocessing
    image = image[index]
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = settings.STD * image + settings.MEAN
    image = image * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Mask preprocessing
    mask = mask[index][0]
    mask = mask.cpu().numpy()
    mask = mask * 255
    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    output = cv2.hconcat([mask, image])
    cv2.imwrite(f'{path_to_save}/{epoch}.png', output)


def linear_beta_schedule(time_steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, time_steps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cum_prod_t = get_index_from_list(sqrt_alphas_cum_prod, t, x_0.shape)
    sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(
        sqrt_one_minus_alphas_cum_prod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cum_prod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cum_prod_t.to(device) * noise.to(device), noise.to(device)


betas = linear_beta_schedule(time_steps=settings.TIME_STEPS)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cum_prod = torch.cumprod(alphas, axis=0)
alphas_cum_prod_prev = torch.nn.functional.pad(alphas_cum_prod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cum_prod = torch.sqrt(alphas_cum_prod)
sqrt_one_minus_alphas_cum_prod = torch.sqrt(1. - alphas_cum_prod)
posterior_variance = betas * (1. - alphas_cum_prod_prev) / (1. - alphas_cum_prod)


@torch.no_grad()
def sample_timestep(model, x, mask_local, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(
        sqrt_one_minus_alphas_cum_prod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, mask_local) / sqrt_one_minus_alphas_cum_prod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, mask_local, epoch):
    # Sample noise
    img = torch.randn((1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE), device=device)
    mask_local = torch.unsqueeze(mask_local, dim=0)
    num_images = 1
    step_size = int(settings.TIME_STEPS / num_images)

    for i in (range(0, settings.TIME_STEPS)[::-1]):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model=model, x=img, mask_local=mask_local, t=t)
        if i % step_size == 0:
            plot_one_image(image=img, mask=mask_local, epoch=f'{epoch}_{i}')


def train_model(model, dataloader, optimizer, num_epochs):
    model = model.to(device)
    min_loss = 1e3
    for epoch in range(num_epochs):
        train_loss_epoch_history = []
        train_l1_epoch_history = []
        train_l2_epoch_history = []
        with tqdm.tqdm(dataloader, unit='batch') as tqdm_loader:
            for image, mask in tqdm_loader:
                tqdm_loader.set_description(f'Epoch {epoch}')
                time.sleep(.1)
                image = image.to(device)
                mask = mask.float()
                mask = mask.to(device)

                optimizer.zero_grad()

                t = torch.randint(0, settings.TIME_STEPS, (settings.BATCH_SIZE,), device=device).long()
                x_noisy, noise = forward_diffusion_sample(image, t)
                noise_predict = model(x_noisy, t, mask)
                # Loss
                loss = torch.nn.functional.smooth_l1_loss(noise, noise_predict)
                train_loss_epoch_history.append(loss.item())
                # Metrics
                l1 = torch.nn.functional.l1_loss(noise, noise_predict)
                l2 = torch.nn.functional.mse_loss(noise, noise_predict)
                train_l1_epoch_history.append(l1.item())
                train_l2_epoch_history.append(l2.item())

                loss.backward()
                optimizer.step()

                tqdm_loader.set_postfix(Loss=loss.item(), L1=l1.item(), L2=l2.item())
                time.sleep(.1)
        # Statistic during epoch
        epoch_loss = np.mean(train_loss_epoch_history)
        epoch_l1 = np.mean(train_l1_epoch_history)
        epoch_l2 = np.mean(train_l2_epoch_history)
        # Save best model
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            model = model.cpu()
            torch.save(model, 'checkpoint/best.pth')
            model = model.to(device)
        if epoch % 5 == 0:
            sample_plot_image(model=model, mask_local=mask[0], epoch=epoch)
        print(f'Epoch {epoch:03d} Loss: {epoch_loss:.4f} L1: {epoch_l1:.4f} L2: {epoch_l2:.4f}')
        time.sleep(.1)
    torch.save(model, 'checkpoint/last.pth')
