from .dataset import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def generate_loader(phase, opt):
    dataset = Dataset
    img_size = opt.input_size
    mean=[0, 0, 0]
    std=[1, 1, 1]
    scale = 1.1

    if phase == 'train':
        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=int(img_size * scale)),
                A.PadIfNeeded(
                    min_height=int(img_size * scale),
                    min_width=int(img_size * scale),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomCrop(width=img_size, height=img_size),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
                A.OneOf(
                    [A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                     A.IAAAffine(shear=15, p=0.5, mode="constant")],
                     p=1.0
                     ),
                A.HorizontalFlip(p=0.5),
                A.Blur(p=0.1),
                A.CLAHE(p=0.1),
                A.Posterize(p=0.1),
                A.ToGray(p=0.1),
                A.ChannelShuffle(p=0.05),
                A.Normalize(mean, std, max_pixel_value=255),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
        )

    else:
        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean, std, max_pixel_value=255),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
        )

    dataset = dataset(opt, phase, transform=transform)

    kwargs = {
        "batch_size": opt.batch_size if phase == 'train' else opt.eval_batch_size
    }

    return torch.utils.data.DataLoader(dataset, **kwargs)