import torch
import random
import numpy as np

import utils
import model
import dataset
import settings


def main():
    # Fix random
    random.seed(settings.RANDOM_STATE)
    np.random.seed(settings.RANDOM_STATE)
    torch.manual_seed(settings.RANDOM_STATE)
    torch.cuda.manual_seed(settings.RANDOM_STATE)

    generator = model.Unet(
        dim=settings.IMAGE_SIZE,
        channels=settings.CHANNELS,
        dim_mults=(1, 2, 4)
    )
    optimizer = torch.optim.AdamW(generator.parameters(), lr=settings.LEARNING_RATE)
    utils.train_model(
        model=generator,
        dataloader=dataset.train_dataloader,
        optimizer=optimizer,
        num_epochs=settings.EPOCHS
    )


if __name__ == '__main__':
    main()
