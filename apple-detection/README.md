# Apple Detection Web App

A web application for detecting apples in images using deep learning with PyTorch and Flask.

## Features
- Image classification to detect apples
- Modern web interface with drag-and-drop upload
- Real-time prediction results with confidence scores
- Fallback to heuristic-based detection when no trained model is available
- Responsive design that works on desktop and mobile

## Project Structure
```
apple-detection/
├── app.py               # Flask web application
├── app_streamlit.py     # Original Streamlit app (backup)
├── config.py            # Configuration for paths and hyperparameters
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Main web page template
├── static/              # Static assets (CSS, JS, images)
├── data/                # Dataset directory (train/val/test)
├── checkpoints/         # Trained model checkpoints
└── src/
    ├── dataset.py       # Dataset and dataloader utilities
    ├── inference.py     # Inference utilities
    ├── model.py         # Model definition (ResNet18)
    └── train.py         # Training and evaluation scripts
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

3. (Optional) Prepare your dataset and train a model:
   - Place your images in `data/train`, `data/val`, and `data/test` folders, organized by class subfolders (e.g., `apple/`, `not_apple/`).
   - Train the model: `python src/train.py`

## Running the Web App
To launch the Flask web application:
```sh
python app.py
# Or use the launcher script:
python run.py
```

The app will be available at `http://localhost:5000`

## Docker Deployment
For production deployment using Docker:

1. Build and run with Docker Compose:
   ```sh
   docker-compose up --build
   ```

2. Or build and run manually:
   ```sh
   docker build -t apple-detection .
   docker run -p 5000:5000 apple-detection
   ```

## How It Works
- **Trained Model**: If a checkpoint exists at `checkpoints/apple_classifier.pt`, the app uses the trained PyTorch model for predictions
- **Demo Mode**: If no checkpoint is found, the app falls back to a heuristic-based detection using color analysis
- **Web Interface**: Modern, responsive interface with drag-and-drop file upload and real-time results

## Training
To train your own model:
```sh
python src/train.py
```

## Inference
You can also use the inference utilities programmatically:
```python
from src.inference import predict_is_apple
from PIL import Image

image = Image.open('path/to/image.jpg')
is_apple, confidence = predict_is_apple(image, 'checkpoints/apple_classifier.pt')
```

## License
MIT License
