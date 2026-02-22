# Cats vs Dogs Classification - MLOps Pipeline

A complete end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) with automated CI/CD, containerization, Kubernetes deployment, and a Streamlit web interface.

## Project Overview

This project implements a production-ready machine learning system covering the full ML lifecycle:

- **Data Versioning**: DVC for tracking datasets and model artifacts
- **Model Training**: Baseline CNN with MLflow experiment tracking
- **API Service**: FastAPI REST API for inference
- **Web Interface**: Streamlit UI for interactive image classification
- **Containerization**: Docker for reproducible deployments
- **CI/CD**: GitHub Actions for automated testing, building, and deployment
- **Orchestration**: Kubernetes for scalable deployment
- **Monitoring**: Request logging and performance tracking

## Project Structure

```
Assignment2/
├── .github/workflows/           # CI/CD pipelines
│   ├── ci.yml                   # Continuous Integration
│   └── cd.yml                   # Continuous Deployment
├── data/                        # Dataset (tracked by DVC)
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── deployment/                  # Deployment configurations
│   ├── kubernetes/              # Kubernetes manifests
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── kustomization.yaml
│   └── docker-compose.yml       # Docker Compose config (API + UI)
├── models/                      # Model artifacts (tracked by DVC)
├── notebooks/                   # Jupyter notebooks for exploration
├── scripts/                     # Utility scripts
│   ├── smoke_test.py            # Post-deployment tests
│   └── performance_tracking.py  # Model evaluation
├── src/                         # Source code
│   ├── api/                     # FastAPI application
│   │   ├── main.py
│   │   └── schemas.py
│   ├── data/                    # Data processing
│   ├── models/                  # Model definitions and training
│   ├── ui/                      # Streamlit web interface
│   │   └── streamlit_app.py
│   └── utils/                   # Shared utilities
│       └── inference.py
├── tests/                       # Unit tests (24 tests)
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── dvc.yaml                     # DVC pipeline
└── params.yaml                  # Hyperparameters
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker
- Git
- DVC
- kubectl (for Kubernetes deployment)
- minikube (for local Kubernetes cluster)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/manishbitswilp/MLOPS_Assignment2.git
cd Assignment2
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download and organize dataset**

```bash
# Download from: https://www.kaggle.com/datasets/salader/dogs-vs-cats
# Extract to data/raw/, then run:
python src/data/organize_dataset.py
```

5. **Initialize DVC**

```bash
dvc init
dvc add data/processed
git add data/processed.dvc .dvc/config
git commit -m "Initialize DVC tracking"
```

## Model Training

### Train the baseline model

```bash
python src/models/train.py
```

This will:
- Load preprocessed data with augmentation
- Train a baseline CNN model
- Log experiments to MLflow
- Save the model to `models/cats_dogs_classifier.h5`
- Generate training plots and metrics

### View experiments in MLflow

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiment runs.

### Hyperparameters

Edit `params.yaml` to modify training parameters:

```yaml
train:
  learning_rate: 0.001
  batch_size: 32
  epochs: 20
  img_size: [224, 224]

model:
  architecture: baseline_cnn
  dropout_rate: 0.5
```

Then reproduce the pipeline:

```bash
dvc repro
```

## API Usage

### Run locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

**Health check:**

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Make prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "class": "dog",
  "confidence": 0.9523,
  "dog_probability": 0.9523,
  "cat_probability": 0.0477
}
```

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Streamlit Web Interface

The Streamlit UI provides a browser-based interface for uploading images and viewing classification results.

### Run locally

```bash
streamlit run src/ui/streamlit_app.py --server.port 8501
```

Navigate to `http://localhost:8501`.

### Features

- Sidebar showing live API health status and model availability
- Image upload supporting JPG, JPEG, and PNG formats
- Displays the uploaded image preview
- Shows prediction result (Cat or Dog) with confidence score
- Side-by-side probability breakdown for both classes

## Docker Deployment

### Docker Hub image

```
manishbitswilp/cats-dogs-classifier:latest
```

### Build image

```bash
docker build -t manishbitswilp/cats-dogs-classifier:latest .
```

### Run API only

```bash
docker run -p 8000:8000 manishbitswilp/cats-dogs-classifier:latest
```

### Run with Docker Compose (API + UI)

Docker Compose starts both the API service and the Streamlit UI:

```bash
cd deployment
DOCKER_USERNAME=manishbitswilp docker compose up -d
```

Services:
| Service | Container | Port | Description |
|---|---|---|---|
| classifier | cats-dogs-classifier | 8000 | FastAPI inference API |
| ui | cats-dogs-ui | 8501 | Streamlit web interface |

The UI communicates with the API via the Docker internal network using the service name `classifier`.

**Stop services:**

```bash
DOCKER_USERNAME=manishbitswilp docker compose down
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (minikube for local)
- kubectl configured

### Start minikube

```bash
minikube start
```

### Deploy

1. **Substitute your Docker Hub username:**

```bash
export DOCKER_USERNAME=manishbitswilp
sed -i "s/\${DOCKER_USERNAME}/$DOCKER_USERNAME/g" deployment/kubernetes/deployment.yaml
```

2. **Apply manifests:**

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

3. **Check status:**

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

4. **Access the service:**

```bash
# For minikube
minikube service cats-dogs-classifier --url

# For cloud providers
kubectl get service cats-dogs-classifier
```

5. **Run smoke tests:**

```bash
python scripts/smoke_test.py --url <service-url>
```

### Scale deployment

```bash
kubectl scale deployment cats-dogs-classifier --replicas=3
```

### View logs

```bash
kubectl logs -l app=cats-dogs-classifier --tail=100 -f
```

## CI/CD Pipeline

### CI Pipeline (`.github/workflows/ci.yml`)

Triggered on push to `main` or `develop`, and on pull requests to `main`:

1. **Test**: Run 24 unit tests with coverage reporting (Codecov)
2. **Lint**: Code quality checks with flake8, black, and isort
3. **Build and Push**: Build Docker image and push to Docker Hub (on push to `main` only)

### CD Pipeline (`.github/workflows/cd.yml`)

Triggered automatically after a successful CI run on `main`:

1. **Deploy**: Apply Kubernetes manifests with the new image tag
2. **Smoke Tests**: Verify the deployed service is healthy
3. **Rollback**: Automatically rolls back on failure

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub password or access token |
| `KUBECONFIG` | Base64-encoded kubeconfig for the target cluster |

```bash
# Encode kubeconfig
cat ~/.kube/config | base64
```

## Testing

### Run unit tests

```bash
pytest tests/ -v
```

### Run with coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

View the coverage report at `htmlcov/index.html`.

### Smoke tests

```bash
python scripts/smoke_test.py --url http://localhost:8000
```

## Architecture

### Model Architecture

**Baseline CNN:**
- Conv2D(32) + MaxPooling2D
- Conv2D(64) + MaxPooling2D
- Conv2D(128) + MaxPooling2D
- Flatten + Dense(128) + Dropout(0.5)
- Dense(1, sigmoid) — binary output (dog probability)

Input: 224x224x3 (RGB, normalized to [0, 1])
Output: probability in [0, 1]; values above 0.5 are classified as dog

### Image Preprocessing

Images are decoded using PIL (supports JPEG, PNG, WEBP, BMP, and more), resized to 224x224, and normalized to [0, 1] before inference.

### Data Pipeline

1. Download raw dataset from Kaggle
2. Organize into train/val/test splits (80/10/10)
3. Apply preprocessing and augmentation (rotation, flip, zoom, brightness)

### Deployment Architecture

```
User
 |
 +-- Browser (port 8501) --> Streamlit UI
                                  |
                                  v
                         FastAPI API (port 8000)
                                  |
                                  v
                         TF/Keras CNN Model
                         (cats_dogs_classifier.h5)
```

Kubernetes:
```
Internet --> LoadBalancer --> Kubernetes Service
                                     |
                              Deployment (2 replicas)
                                     |
                              Pod (FastAPI + Model)
```

## Monitoring

### API logs

Logs are written to stdout and include:
- Request timestamps
- Uploaded file name and size
- Predicted class and confidence score
- Inference latency per request

```bash
# Docker
docker logs cats-dogs-classifier -f

# Kubernetes
kubectl logs -l app=cats-dogs-classifier -f
```

### Performance tracking

Evaluate the deployed model on the test set:

```bash
python scripts/performance_tracking.py \
  --api-url http://localhost:8000 \
  --test-dir data/processed/test \
  --max-samples 100
```

Outputs accuracy, precision, recall, F1, confusion matrix, and latency statistics to `models/performance/`.

## Troubleshooting

### Model file not found

```bash
python src/models/train.py
ls models/cats_dogs_classifier.h5
```

### Docker build fails

```bash
ls requirements.txt src/ models/
docker build --no-cache -t cats-dogs-classifier:latest .
```

### API returns 503

Model not loaded. Check that `MODEL_PATH` is correct and the model file exists in the container:

```bash
docker exec cats-dogs-classifier ls /app/models/
```

### Kubernetes pod not starting

```bash
kubectl logs <pod-name>
kubectl describe pod <pod-name>
```

Common causes: image pull errors (check Docker Hub credentials), resource limits (adjust in `deployment.yaml`), health check timeout (increase `initialDelaySeconds`).

### Streamlit UI shows API Unreachable

When running via Docker Compose, the UI connects to the API using the internal service name. Ensure the `API_URL` environment variable is set to `http://classifier:8000` in `docker-compose.yml` and that both containers are on the same Docker network (`mlops-network`).

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## License

This project is for educational purposes as part of an MLOps assignment.

## Author

Group 128

## Acknowledgments

- Dataset: [Dogs vs. Cats on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- Assignment designed for MLOps course
