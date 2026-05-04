from argparse import ArgumentParser
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from config import DEFAULT_CHECKPOINT, IMAGE_SIZE
from src.model import build_model


def parse_args() -> ArgumentParser:
	parser = ArgumentParser(description="Run inference with a trained apple classifier.")
	parser.add_argument("image", type=Path)
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	return parser


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str]]:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	classes = checkpoint["classes"]
	model = build_model(num_classes=len(classes))
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	return model, classes


def load_model_from_bytes(checkpoint_bytes: bytes) -> tuple[torch.nn.Module, list[str]]:
	checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
	classes = checkpoint.get("classes")
	state = checkpoint.get("model_state_dict")

	if not isinstance(classes, list) or not classes or not all(isinstance(item, str) for item in classes):
		raise ValueError("Checkpoint must contain 'classes' as list[str].")
	if state is None:
		raise ValueError("Checkpoint must contain 'model_state_dict'.")

	model = build_model(num_classes=len(classes))
	model.load_state_dict(state)
	model.eval()
	return model, classes


def is_apple_label(label: str) -> bool:
	normalized = label.strip().lower().replace("\t", " ")

	if normalized in {"apple", "yes", "true", "positive"}:
		return True

	if normalized in {
		"not apple",
		"not_apple",
		"not-apple",
		"non apple",
		"non_apple",
		"non-apple",
		"no apple",
		"no_apple",
		"no-apple",
		"negative",
		"false",
		"0",
	}:
		return False

	if "apple" in normalized and ("not" in normalized or normalized.startswith(("non", "no"))):
		return False

	return "apple" in normalized


def build_inference_transform() -> transforms.Compose:
	return transforms.Compose(
		[
			transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
			transforms.ToTensor(),
		]
	)


def predict_image(image: Image.Image, checkpoint_path: Path) -> tuple[str, float]:
	model, classes = load_model(checkpoint_path)
	transform = build_inference_transform()
	input_tensor = transform(image.convert("RGB")).unsqueeze(0)

	with torch.no_grad():
		logits = model(input_tensor)
		probabilities = torch.softmax(logits, dim=1)[0]
		confidence, class_index = torch.max(probabilities, dim=0)

	return classes[class_index.item()], confidence.item()


def predict_is_apple(image: Image.Image, checkpoint_path: Path) -> tuple[bool, float]:
	"""Binary apple decision with confidence.

	Confidence is the probability of the returned boolean decision.
	"""

	model, classes = load_model(checkpoint_path)
	transform = build_inference_transform()
	input_tensor = transform(image.convert("RGB")).unsqueeze(0)

	with torch.no_grad():
		logits = model(input_tensor)
		probabilities = torch.softmax(logits, dim=1)[0]

	apple_indices = [idx for idx, name in enumerate(classes) if is_apple_label(name)]
	if apple_indices:
		apple_prob = float(probabilities[apple_indices].sum().item())
		verdict = apple_prob >= 0.5
		confidence = apple_prob if verdict else (1.0 - apple_prob)
		return bool(verdict), float(confidence)

	# Fallback: treat top-1 as the decision.
	conf, class_index = torch.max(probabilities, dim=0)
	verdict = is_apple_label(classes[int(class_index.item())])
	return bool(verdict), float(conf.item())


def predict_is_apple_from_bytes(image: Image.Image, checkpoint_bytes: bytes) -> tuple[bool, float]:
	model, classes = load_model_from_bytes(checkpoint_bytes)
	transform = build_inference_transform()
	input_tensor = transform(image.convert("RGB")).unsqueeze(0)

	with torch.no_grad():
		logits = model(input_tensor)
		probabilities = torch.softmax(logits, dim=1)[0]

	apple_indices = [idx for idx, name in enumerate(classes) if is_apple_label(name)]
	if apple_indices:
		apple_prob = float(probabilities[apple_indices].sum().item())
		verdict = apple_prob >= 0.5
		confidence = apple_prob if verdict else (1.0 - apple_prob)
		return bool(verdict), float(confidence)

	conf, class_index = torch.max(probabilities, dim=0)
	verdict = is_apple_label(classes[int(class_index.item())])
	return bool(verdict), float(conf.item())


def predict(image_path: Path, checkpoint_path: Path) -> tuple[str, float]:
	image = Image.open(image_path)
	return predict_image(image, checkpoint_path)


def main() -> None:
	args = parse_args().parse_args()
	label, confidence = predict(args.image, args.checkpoint)
	print(f"prediction={label} confidence={confidence:.4f}")


if __name__ == "__main__":
	main()
