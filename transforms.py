import albumentations as A
from albumentations.pytorch import ToTensorV2

import settings

train_aug = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1.),
        A.VerticalFlip(p=1.),
        A.RandomRotate90(p=1.),
        A.Transpose(p=1)
    ], p=.5),
    A.OneOf([
        A.ElasticTransform(p=1., alpha=120, sigma=120 * .05, alpha_affine=120 * .03),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.),
        A.GridDistortion(p=1.)
    ], p=.5),
    A.Normalize(mean=settings.MEAN, std=settings.STD, p=1.),
    ToTensorV2(transpose_mask=True)
])
