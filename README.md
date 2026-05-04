# apple-detection
# Apple Detection

A simple deep learning project for detecting apples in images using PyTorch and Streamlit.

## Features
- Image classification to detect apples
- Streamlit web app for easy demo
- Custom dataset support
- Model training and inference scripts

## Project Structure
```
apple-detection/
├── app.py               # Streamlit web app for apple detection demo
├── config.py            # Configuration for paths and hyperparameters
├── requirements.txt     # Python dependencies
├── data/                # Dataset directory (train/val/test)
├── src/
│   ├── dataset.py       # Dataset and dataloader utilities
│   ├── inference.py     # Inference utilities
│   ├── model.py         # Model definition (ResNet18)
│   └── train.py         # Training and evaluation scripts
```

## Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/rp31803/apple-detection.git
   cd apple-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Prepare your dataset:
   - Place your images in `data/train`, `data/val`, and `data/test` folders, organized by class subfolders (e.g., `apple/`, `not_apple/`).

## Training
To train the model:
```sh
python src/train.py
```

## Running the Demo App
To launch the Streamlit demo:
```sh
streamlit run app.py
```

## Inference
You can use the provided inference utilities in `src/inference.py` for batch or custom predictions.

## License
MIT License
