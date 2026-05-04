from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS


def build_transforms(train: bool = False) -> transforms.Compose:
	if train:
		return transforms.Compose(
			[
				transforms.RandomResizedCrop(IMAGE_SIZE),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
			]
		)

	return transforms.Compose(
		[
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.ToTensor(),
		]
	)


def build_imagefolder_dataset(root_dir: Path, train: bool = False) -> datasets.ImageFolder:
	return datasets.ImageFolder(root_dir, transform=build_transforms(train=train))


def build_dataloader(root_dir: Path, train: bool = False, batch_size: int = BATCH_SIZE) -> DataLoader:
	dataset = build_imagefolder_dataset(root_dir, train=train)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=train,
		num_workers=NUM_WORKERS,
		pin_memory=True,
	)
