import torch

import utils
import settings
import transforms


# Get train/valid DataFrames
df = utils.data_preparation(settings.DATA_FOLDER)
# Datasets
train_dataset = utils.LocalDataset(
    df,
    transform=transforms.train_aug
)
# Datasets
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=settings.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)
# Plot Images with masks
if __name__ == '__main__':
    # Check data shape for image-mask
    print(f'Image shape:\n{list(train_dataset[0][0].shape)}')
    print(f'Mask shape:\n{list(train_dataset[0][1].shape)}\n')
    # Check train-valid size
    print(f'Train dataset length: {train_dataset.__len__()}')
    utils.batch_image_mask_show(train_dataloader)
