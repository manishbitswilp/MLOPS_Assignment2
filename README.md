# Cats vs Dogs Classification - MLOps Pipeline

A complete end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) with automated CI/CD, containerization, Kubernetes deployment, and monitoring.

## 🎯 Project Overview

This project implements a production-ready machine learning system that demonstrates the full ML lifecycle:

- **Data Versioning**: DVC for tracking datasets and model artifacts
- **Model Training**: Baseline CNN with MLflow experiment tracking
- **API Service**: FastAPI REST API for inference
- **Containerization**: Docker for reproducible deployments
- **CI/CD**: GitHub Actions for automated testing, building, and deployment
- **Orchestration**: Kubernetes for scalable deployment
- **Monitoring**: Request logging and performance tracking

## 📁 Project Structure

```
Assignment2/
├── .github/workflows/       # CI/CD pipelines
│   ├── ci.yml              # Continuous Integration
│   └── cd.yml              # Continuous Deployment
├── data/                   # Dataset (tracked by DVC)
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── deployment/             # Deployment configurations
│   ├── kubernetes/         # Kubernetes manifests
│   └── docker-compose.yml  # Docker Compose config
├── models/                 # Model artifacts (tracked by DVC)
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Utility scripts
│   ├── setup_dvc.sh       # DVC initialization
│   ├── smoke_test.py      # Post-deployment tests
│   └── performance_tracking.py  # Model evaluation
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── data/              # Data processing
│   ├── models/            # Model definitions
│   └── utils/             # Utilities
├── tests/                  # Unit tests
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── dvc.yaml              # DVC pipeline
└── params.yaml           # Hyperparameters
```

## 🚀 Quick Start

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
git clone <repository-url>
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
# Option 1: Using Kaggle API (requires kaggle credentials)
python src/data/download.py

# Option 2: Manual download from Kaggle
# Download from: https://www.kaggle.com/datasets/salader/dogs-vs-cats
# Extract to data/raw/
# Then run:
python src/data/download.py  # This will organize the dataset
```

5. **Initialize DVC**

```bash
dvc init
dvc add data/processed
git add data/processed.dvc .dvc/config
git commit -m "Initialize DVC tracking"
```

## 📊 Model Training

### Train the baseline model

```bash
python src/models/train.py
```

This will:
- Load preprocessed data with augmentation
- Train a baseline CNN model
- Log experiments to MLflow
- Save model to `models/cats_dogs_classifier.h5`
- Generate training plots and metrics

### View experiments in MLflow

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiments.

### Customize training

Edit `params.yaml` to modify hyperparameters:

```yaml
train:
  learning_rate: 0.001
  batch_size: 32
  epochs: 20
```

Then run:

```bash
dvc repro  # Re-run DVC pipeline with new parameters
```

## 🔧 API Usage

### Run locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Test endpoints

**Health check:**

```bash
curl http://localhost:8000/health
```

**Make prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/image.jpg"
```

**Response:**

```json
{
  "class": "dog",
  "confidence": 0.9523,
  "dog_probability": 0.9523,
  "cat_probability": 0.0477
}
```

### API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🐳 Docker Deployment

### Build Docker image

```bash
docker build -t cats-dogs-classifier:latest .
```

### Run container

```bash
docker run -p 8000:8000 cats-dogs-classifier:latest
```

### Using Docker Compose

```bash
# Edit docker-compose.yml to set DOCKER_USERNAME
docker-compose -f deployment/docker-compose.yml up -d
```

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (minikube for local)
- kubectl configured

### Start minikube

```bash
minikube start
```

### Deploy to Kubernetes

1. **Update deployment with your Docker Hub username:**

```bash
export DOCKER_USERNAME=your-dockerhub-username
sed -i "s/\${DOCKER_USERNAME}/$DOCKER_USERNAME/g" deployment/kubernetes/deployment.yaml
```

2. **Apply manifests:**

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

3. **Check deployment status:**

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

4. **Access the service:**

```bash
# For minikube
minikube service cats-dogs-classifier --url

# For cloud providers, get LoadBalancer IP
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

## 🔄 CI/CD Pipeline

### CI Pipeline (`.github/workflows/ci.yml`)

Triggered on push to `main` or pull requests:

1. **Test Job**: Run unit tests with coverage
2. **Lint Job**: Code quality checks (black, isort, flake8)
3. **Build Job**: Build and push Docker image to Docker Hub

### CD Pipeline (`.github/workflows/cd.yml`)

Triggered after successful CI:

1. **Deploy Job**: Deploy to Kubernetes
2. **Smoke Tests**: Verify deployment health
3. **Rollback**: Automatic rollback on failure

### Setup GitHub Secrets

Configure these secrets in your GitHub repository:

- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password/token
- `KUBECONFIG`: Base64-encoded kubeconfig file

```bash
# Encode kubeconfig
cat ~/.kube/config | base64
```

## 🧪 Testing

### Run unit tests

```bash
pytest tests/ -v
```

### Run with coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report at `htmlcov/index.html`.

### Smoke tests (deployment verification)

```bash
python scripts/smoke_test.py --url http://localhost:8000
```

## 📈 Monitoring & Performance Tracking

### View API logs

Logs include:
- Request timestamps
- Image file information
- Prediction results and confidence
- Latency per request

```bash
# Local
tail -f api.log

# Kubernetes
kubectl logs -l app=cats-dogs-classifier -f
```

### Track model performance

Evaluate deployed model on test set:

```bash
python scripts/performance_tracking.py \
  --api-url http://localhost:8000 \
  --test-dir data/processed/test \
  --max-samples 100
```

This generates:
- Accuracy, Precision, Recall, F1 metrics
- Confusion matrix visualization
- Latency statistics
- Metrics saved to `models/performance/`

## 🏗️ Architecture

### Model Architecture

**Baseline CNN:**
- Conv2D(32) → MaxPooling2D
- Conv2D(64) → MaxPooling2D
- Conv2D(128) → MaxPooling2D
- Flatten → Dense(128) → Dropout(0.5)
- Dense(1, sigmoid) for binary classification

### Data Pipeline

1. Download raw dataset
2. Organize into train/val/test splits (80/10/10)
3. Apply preprocessing:
   - Resize to 224×224
   - Normalize to [0, 1]
   - Data augmentation (rotation, flip, zoom, brightness)

### Deployment Architecture

```
User Request → LoadBalancer → Kubernetes Service
                                     ↓
                              Deployment (2 replicas)
                                     ↓
                              Pod (FastAPI + Model)
```

## 🐛 Troubleshooting

### Model file not found

Ensure model is trained and saved:

```bash
python src/models/train.py
ls models/cats_dogs_classifier.h5
```

### Docker build fails

Check that all required files are present:

```bash
ls requirements.txt src/ models/
```

### Kubernetes pod not starting

Check pod logs:

```bash
kubectl logs <pod-name>
kubectl describe pod <pod-name>
```

Common issues:
- Image pull errors: Verify Docker Hub credentials
- Resource limits: Adjust in `deployment.yaml`
- Health check failures: Increase `initialDelaySeconds`

### API returns 503

Model not loaded. Check:
- Model file exists in container
- `MODEL_PATH` environment variable is correct
- Container logs for loading errors

## 📝 Assignment Deliverables

### Milestone Checklist

- [x] M1: Model Development & Experiment Tracking
  - Data pipeline with DVC
  - Baseline CNN model
  - MLflow experiment tracking

- [x] M2: Model Packaging & Containerization
  - FastAPI inference service
  - Dockerfile and containerization

- [x] M3: CI Pipeline
  - Unit tests
  - GitHub Actions CI
  - Docker image publishing

- [x] M4: CD Pipeline & Deployment
  - Kubernetes manifests
  - Automated deployment
  - Smoke tests

- [x] M5: Monitoring & Documentation
  - Request logging
  - Performance tracking
  - Comprehensive documentation

### Demo Video Instructions

Record a 5-minute demo showing:

1. **Code Change**: Make a small modification (e.g., update hyperparameter)
2. **Git Push**: Commit and push to GitHub
3. **CI Pipeline**: Show GitHub Actions running (tests, Docker build)
4. **CD Pipeline**: Show automatic deployment
5. **Verification**:
   ```bash
   kubectl get pods
   curl <service-url>/health
   curl -X POST -F "file=@test_image.jpg" <service-url>/predict
   ```
6. **Monitoring**: Show logs with prediction captured

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## 📄 License

This project is for educational purposes as part of an MLOps assignment.

## 👥 Author

[Your Name]

## 🙏 Acknowledgments

- Dataset: [Dogs vs. Cats on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- Assignment designed for MLOps course

---

**Last Updated**: 2024

For questions or issues, please open an issue in the repository.
