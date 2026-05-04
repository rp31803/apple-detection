from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn

from config import CHECKPOINT_DIR, DEFAULT_CHECKPOINT, NUM_EPOCHS, PROJECT_ROOT, SEED, TRAIN_DIR, VAL_DIR, LEARNING_RATE
from src.dataset import build_dataloader, build_imagefolder_dataset
from src.model import build_model


def train_one_epoch(model: nn.Module, dataloader, criterion, optimizer, device: torch.device) -> float:
	model.train()
	running_loss = 0.0

	for images, labels in dataloader:
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)

	return running_loss / max(len(dataloader.dataset), 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader, criterion, device: torch.device) -> tuple[float, float]:
	model.eval()
	running_loss = 0.0
	correct = 0

	for images, labels in dataloader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)

		running_loss += loss.item() * images.size(0)
		predictions = outputs.argmax(dim=1)
		correct += (predictions == labels).sum().item()

	total = max(len(dataloader.dataset), 1)
	return running_loss / total, correct / total


def parse_args() -> ArgumentParser:
	parser = ArgumentParser(description="Train an apple classifier.")
	parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
	parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
	parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
	parser.add_argument("--lr", type=float, default=LEARNING_RATE)
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	return parser


def main() -> None:
	args = parse_args().parse_args()
	torch.manual_seed(SEED)

	train_dataset = build_imagefolder_dataset(args.train_dir, train=True)
	val_dataset = build_imagefolder_dataset(args.val_dir, train=False)

	if not train_dataset.classes:
		raise SystemExit(f"No classes found in {args.train_dir}. Use ImageFolder structure with class subfolders.")

	train_loader = build_dataloader(args.train_dir, train=True)
	val_loader = build_dataloader(args.val_dir, train=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = build_model(num_classes=len(train_dataset.classes)).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	best_val_accuracy = 0.0
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

	for epoch in range(1, args.epochs + 1):
		train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

		if val_accuracy >= best_val_accuracy:
			best_val_accuracy = val_accuracy
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"classes": train_dataset.classes,
				},
				args.checkpoint,
			)

		print(
			f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
		)

	print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
	main()
