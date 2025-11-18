<center>

# Digit-DL – Handwritten Digit Recognition

</center>

## Overview
- Digit-DL delivers an end-to-end MNIST digit recognition experience with a Flask backend and a static HTML/CSS/JS frontend.
- The service accepts user drawings or uploaded images, runs an enhanced preprocessing pipeline, and returns the top predictions from a CNN model.
- Everything needed to retrain the model from the original MNIST dataset is included for reproducibility.

## Project Structure
- `backend/` – Flask API, pretrained models, dependency manifests.
- `frontend/` – Static app for drawing/uploading digits and visualizing predictions.
- `notebooks/traine.py` – Script/notebook for training and exporting the CNN.
- `data/` – Placeholder for any custom datasets or inference assets.
- `reports/` – Reserved for evaluation outputs and experiment notes.
- `venv/` – (Optional) Python virtual environment; not required if you manage envs differently.

## Tech Stack
- **Modeling** – TensorFlow/Keras CNN trained on MNIST with augmentation.
- **Serving** – Flask + Flask-CORS with OpenCV/Pillow preprocessing.
- **Frontend** – Responsive vanilla HTML/CSS/JS canvas app.
- **Runtime** – Python 3.10 (see `backend/runtime.txt` for deployment compatibility).

## Getting Started
- **Prerequisites**
  - Python 3.10+
  - Node is *not* required (frontend is static).
  - (Optional) Virtual environment tool (`venv`, `conda`, etc.).
- **Clone & Setup**
  - `python -m venv venv` (or activate existing one under `venv/`).
  - `pip install -r backend/requirements.txt`.
  - Ensure `backend/model/mnist_model_best.h5` is present (fallback `mnist_model.h5` is also available).

## Running the Backend
- Activate your environment and `cd backend`.
- Launch with `python app.py` for local development (defaults to `http://127.0.0.1:5000`).
- Health check: `GET /health` → confirms model status.
- Environment variables:
  - `FLASK_ENV=development` for verbose logging.
  - `PORT` if deploying behind different port (update frontend accordingly).

## Using the Frontend
- Open `frontend/index.html` directly in a browser or serve via any static file server.
- Update the `API_URL` constant near the top of the script to match your backend host.
- Draw digits on the canvas or upload an image to trigger predictions.

## API Reference
- `POST /predict`
  - Body: multipart form with `file` containing the digit image.
  - Response: JSON `{ prediction, confidence, top_3, all_predictions }`.
- `GET /health`
  - Response: `{ status, model_loaded }`.
- `GET /`
  - Returns usage instructions for quick verification.

## Model Training
- Run `python notebooks/traine.py` (requires TensorFlow GPU or CPU).
- Outputs stored in `backend/model/`:
  - `mnist_model_best.h5` – best checkpoint via early stopping.
  - `mnist_model.h5` – final model after full training.
  - `training_history.pkl` – serialized training metrics.
- Tune augmentation and architecture directly inside the script for experiments.

## Deployment Notes
- `backend/requirements.txt` and `runtime.txt` support Heroku-like deployments; consider `gunicorn` for production.
- Add HTTPS termination and authentication if exposing publicly.
- For static hosting, deploy `frontend/` to any CDN/static host (Netlify, GitHub Pages, etc.) and point it to the live API.

## Troubleshooting
- **Model not loaded** – confirm the `.h5` files exist under `backend/model/` and match the expected filename.
- **CORS errors** – ensure `flask-cors` is installed and that the backend host matches the `API_URL`.
- **Bad predictions** – retrain using `notebooks/traine.py` and consider supplying domain-specific data in `data/`.

## License
- No explicit license is provided. Add one if you intend to share or open-source the project.

