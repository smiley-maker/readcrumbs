# ReadCrumbs

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Required-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)

An end-to-end Machine Learning Operations (MLOps) project designed to deliver personalized book recommendations in a production-ready environment. The system includes experiment tracking (Weights & Biases), a model registry, a FastAPI serving backend, persistent logging, and separate user and monitoring interfaces, all containerized and ready for deployment on AWS EC2.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Local Development Setup](#local-development-setup)
- [Running the Project Locally](#running-the-project-locally)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [API Documentation](#api-documentation)
- [Frontend Usage](#frontend-usage)
- [Monitoring Dashboard](#monitoring-dashboard)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

- **Personalized Book Recommendations**: Matrix Factorization model provides tailored book suggestions based on user preferences
- **Model Registry**: Integration with Weights & Biases for model versioning and tracking
- **Real-time Predictions**: FastAPI backend serves recommendations with low latency
- **Persistent Logging**: All predictions logged to DynamoDB for analysis and monitoring
- **Monitoring Dashboard**: Streamlit dashboard for tracking model performance, latency, and data drift
- **User Feedback Collection**: Built-in feedback mechanism to improve model accuracy
- **Containerized Deployment**: Fully containerized with Docker for easy deployment
- **Production Ready**: Designed for AWS EC2 deployment with proper error handling and logging

## Architecture

The system is split into three main containerized services:

1. **ML Model Backend**: A Python FastAPI application that:
   - Loads the Matrix Factorization model from S3
   - Serves predictions via `/predict` endpoint
   - Logs all requests to DynamoDB for monitoring
   - Handles model inference with supporting lookup tables

2. **Frontend Interface**: A Streamlit application that:
   - Provides a user-friendly interface for inputting favorite books
   - Displays real-time recommendations from the FastAPI backend
   - Handles user interactions and API communication

3. **Model Monitoring Dashboard**: A Streamlit dashboard that:
   - Connects directly to DynamoDB to visualize prediction logs
   - Tracks prediction latency over time
   - Monitors data drift and prediction distribution
   - Collects and displays user feedback for model accuracy

## Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: Streamlit
- **ML Framework**: scikit-learn, joblib
- **Cloud Services**: AWS (S3, DynamoDB, EC2)
- **Experiment Tracking**: Weights & Biases (W&B)
- **Containerization**: Docker, Docker Compose
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## Prerequisites

Before you begin, ensure you have the following installed and configured:

### System Requirements

- **Python**: 3.11 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Git**: For cloning the repository

### AWS Account Setup

1. **AWS Account**: Create an AWS account if you don't have one
2. **AWS CLI**: Install and configure AWS CLI (optional, for local development)
3. **Required AWS Services**:
   - **S3 Bucket**: Named `readcrumbs` (or update code to use your bucket name)
     - Store model files: `models/als_model-small-v1.pkl`
     - Store lookup tables: `data/v1/index_to_title.json`, `data/v1/title_to_index.json`
   - **DynamoDB Table**: Named `readcrumbs-logs` (or set via `DDB_TABLE` env var)
     - Used for storing prediction logs
   - **EC2 Instance**: For production deployment
     - Recommended: t3.medium or larger
     - Ubuntu 22.04 LTS or Amazon Linux 2

### Weights & Biases Setup

1. Create a W&B account at [wandb.ai](https://wandb.ai)
2. Install W&B: `pip install wandb`
3. Login: `wandb login`
4. Create a project named `readcrumbs` (or update in code)

### IAM Permissions

Your AWS credentials need the following permissions:

- **S3**: `s3:GetObject` on the `readcrumbs` bucket
- **DynamoDB**: 
  - `dynamodb:PutItem`
  - `dynamodb:Scan`
  - `dynamodb:GetItem`

## Environment Variables

Create a `.env` file in the project root (never commit this file to git). Here's the structure:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# DynamoDB Configuration
DDB_TABLE=readcrumbs-logs

# Optional: AWS Session Token (for temporary credentials)
# AWS_SESSION_TOKEN=your_session_token_here

# W&B Configuration (if using experiment tracking)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=readcrumbs
```

### Security Best Practices

- **Never commit `.env` files** to version control
- Use IAM roles on EC2 instances instead of access keys when possible
- Rotate access keys regularly
- Use AWS Secrets Manager or Parameter Store for production deployments
- Set appropriate file permissions: `chmod 600 .env`

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/smiley-maker/readcrumbs
cd readcrumbs
```

### 2. Set Up Python Environment

Choose one of the following options:

**Option A: Using Conda**

```bash
conda create -n readcrumbs python=3.11 -y
conda activate readcrumbs
pip install -r requirements.txt
```

**Option B: Using Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Create .env file from template (if .env.example exists)
cp .env.example .env

# Or create manually
touch .env
```

Edit `.env` and add your AWS credentials and configuration (see [Environment Variables](#environment-variables) section).

### 4. Verify AWS Configuration

Test your AWS setup:

```bash
# Test S3 access
aws s3 ls s3://readcrumbs/

# Test DynamoDB access
aws dynamodb describe-table --table-name readcrumbs-logs
```

## Running the Project Locally

### Using Docker Compose (Recommended)

The entire system is containerized and managed via Docker Compose.

#### 1. Build Containers

Build the Docker images for all services:

```bash
docker compose build
```

#### 2. Run All Services

Start the entire MLOps system in detached mode:

```bash
docker compose up -d
```

To see logs:

```bash
docker compose up
```

#### 3. Verify Services are Running

Check that all containers are up:

```bash
docker compose ps
```

You should see three services running:
- `backend` (FastAPI)
- `frontend` (Streamlit)
- `monitoring` (Streamlit Dashboard)

#### 4. Access the Services

Once running, access the services:

- **FastAPI Backend API**: http://localhost:8000
  - Health Check: http://localhost:8000/health
  - API Docs: http://localhost:8000/docs
- **Frontend Interface**: http://localhost:8080/
- **Monitoring Dashboard**: http://localhost:8081/

#### 5. Stop Services

To stop and remove containers:

```bash
docker compose down
```

To stop and remove containers with volumes:

```bash
docker compose down -v
```

### Running Individual Services

If you prefer to run services individually:

#### Backend Only

```bash
cd backend
docker build -t readcrumbs-backend .
docker run -p 8000:8000 --env-file ../.env readcrumbs-backend
```

#### Frontend Only

```bash
cd frontend
docker build -t readcrumbs-frontend .
docker run -p 8080:8501 readcrumbs-frontend
```

## AWS EC2 Deployment

### Prerequisites

1. **EC2 Instance**:
   - Launch an EC2 instance (Ubuntu 22.04 LTS recommended)
   - Minimum: t3.medium (2 vCPU, 4 GB RAM)
   - Recommended: t3.large or larger for production
   - Storage: 20 GB minimum

2. **Security Group Configuration**:
   - Open the following ports:
     - **Port 8000**: FastAPI backend (HTTP)
     - **Port 8080**: Frontend interface (HTTP)
     - **Port 8081**: Monitoring dashboard (HTTP)
     - **Port 22**: SSH (for initial setup)

3. **IAM Role** (Recommended):
   - Attach an IAM role to your EC2 instance with S3 and DynamoDB permissions
   - This eliminates the need for access keys

### Step-by-Step Deployment

#### 1. Connect to EC2 Instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip-address
```

#### 2. Install Docker and Docker Compose

```bash
# Update system packages
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io

# Install Docker Compose
sudo apt-get install -y docker-compose

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

#### 3. Clone Repository

```bash
git clone https://github.com/smiley-maker/readcrumbs
cd readcrumbs
```

#### 4. Set Up Environment Variables

```bash
# Create .env file
nano .env
```

Add your environment variables (see [Environment Variables](#environment-variables) section).

**Note**: If using IAM roles, you may only need `AWS_REGION` and `DDB_TABLE`.

#### 5. Build and Run Containers

```bash
# Build containers
docker compose build

# Run in detached mode
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

#### 6. Verify Deployment

Test each service:

```bash
# Backend health check
curl http://localhost:8000/health

# Or from your local machine
curl http://your-ec2-ip:8000/health
```

#### 7. Set Up as Systemd Service (Optional)

For automatic startup on reboot:

```bash
# Create systemd service file
sudo nano /etc/systemd/system/readcrumbs.service
```

Add the following:

```ini
[Unit]
Description=ReadCrumbs MLOps Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/readcrumbs
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=ubuntu
Group=docker

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable readcrumbs
sudo systemctl start readcrumbs
sudo systemctl status readcrumbs
```

### Troubleshooting Deployment

**Issue: Containers won't start**

```bash
# Check logs
docker compose logs

# Check if ports are already in use
sudo netstat -tulpn | grep -E '8000|8080|8081'

# Restart Docker daemon
sudo systemctl restart docker
```

**Issue: AWS credentials not working**

```bash
# Verify IAM role is attached (if using roles)
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://readcrumbs/

# Test DynamoDB access
aws dynamodb list-tables
```

**Issue: Services not accessible from outside**

- Check security group rules
- Verify EC2 instance has a public IP
- Check if firewall is blocking ports

## API Documentation

The FastAPI backend provides a RESTful API for book recommendations.

### Base URL

- **Local**: `http://localhost:8000`
- **Production**: `http://your-ec2-ip:8000`

### Endpoints

#### 1. Health Check

Check if the API is running.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok"
}
```

**cURL Example**:
```bash
curl http://localhost:8000/health
```

**Python Example**:
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
# Output: {"status": "ok"}
```

#### 2. Get Random Item

Retrieve a random prediction log from DynamoDB.

**Endpoint**: `GET /random`

**Response**:
```json
{
  "items": ["The Great Gatsby", "1984"],
  "user_id": "1234567890",
  "timestamp": "2024-01-15T10:30:00Z",
  "prediction": ["To Kill a Mockingbird", "Pride and Prejudice", ...]
}
```

**cURL Example**:
```bash
curl http://localhost:8000/random
```

**Python Example**:
```python
import requests

response = requests.get("http://localhost:8000/random")
data = response.json()
print(f"User ID: {data['user_id']}")
print(f"Items: {data['items']}")
```

#### 3. Get Recommendations

Get personalized book recommendations based on favorite books.

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "items": ["The Great Gatsby", "1984", "To Kill a Mockingbird"],
  "userid": "1234567890"
}
```

**Response**:
```json
{
  "recs": [
    "Pride and Prejudice",
    "The Catcher in the Rye",
    "Lord of the Flies",
    "Brave New World",
    "Animal Farm",
    "Fahrenheit 451",
    "The Picture of Dorian Gray",
    "Jane Eyre",
    "Wuthering Heights",
    "Moby Dick"
  ]
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "items": ["The Great Gatsby", "1984", "To Kill a Mockingbird"],
    "userid": "1234567890"
  }'
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "items": ["The Great Gatsby", "1984", "To Kill a Mockingbird"],
    "userid": "1234567890"
}

response = requests.post(url, json=data)
recommendations = response.json()

print("Recommended Books:")
for i, book in enumerate(recommendations["recs"], 1):
    print(f"{i}. {book}")
```

**Error Responses**:

- **400 Bad Request**: Invalid input (e.g., empty items list)
```json
{
  "detail": "Must enter at least one favorite book."
}
```

- **500 Internal Server Error**: Server error
```json
{
  "detail": "Error message here"
}
```

**Error Handling Example**:
```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json={"items": [], "userid": "123"}  # Invalid: empty items
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"Error: {e}")
    print(f"Details: {response.json()}")
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Frontend Usage

The frontend is a Streamlit application that provides a user-friendly interface.

### Accessing the Frontend

- **Local**: http://localhost:8080/
- **Production**: http://your-ec2-ip:8080/

### How to Use

1. **Enter Favorite Books**:
   - In the text area, enter your favorite book titles
   - Separate multiple books with commas
   - Example: `The Great Gatsby, 1984, To Kill a Mockingbird`

2. **Get Recommendations**:
   - Click the "Analyze Sentiment" button (button text may vary)
   - Wait for the API to process your request
   - View your personalized book recommendations

3. **View Results**:
   - Recommendations appear as a numbered list
   - Each recommendation is a book title

### Input Format

- Books should be entered as plain text titles
- Separate multiple books with commas
- Case-insensitive
- The system will match titles from the model's vocabulary

### Expected Output

The frontend displays:
- A list of 10 recommended book titles
- Based on your input favorites
- Ranked by relevance

## Monitoring Dashboard

The monitoring dashboard provides real-time insights into model performance and system health.

### Accessing the Dashboard

- **Local**: http://localhost:8081/
- **Production**: http://your-ec2-ip:8081/

### Features

#### 1. Prediction Latency Over Time

- Visualizes the time taken to process predictions
- Helps identify performance degradation
- Shows trends over time

#### 2. Prediction Distribution (Target Drift)

- Displays the distribution of predicted book titles
- Helps detect data drift
- Shows which books are being recommended most frequently

#### 3. User Feedback Collection

- Allows users to provide feedback on predictions
- Tracks model accuracy based on user feedback
- Calculates live accuracy metrics

#### 4. Live Model Accuracy

- Displays accuracy percentage based on user feedback
- Updates in real-time as feedback is collected
- Helps monitor model performance

### How to Use

1. **View Metrics**: The dashboard automatically loads and displays metrics from DynamoDB

2. **Provide Feedback**:
   - Enter a User ID in the text input
   - View the most recent prediction for that user
   - Select whether the prediction was correct
   - Click "Submit Feedback"

3. **Monitor Performance**:
   - Check the "Live Model Accuracy" metric
   - Review latency trends
   - Monitor prediction distributions

### Interpreting Metrics

- **High Latency**: May indicate model or infrastructure issues
- **Skewed Distribution**: Could indicate data drift or model bias
- **Low Accuracy**: May require model retraining or data quality improvements

## Project Structure

```
readcrumbs/
├── backend/
│   ├── api.py                 # FastAPI application and endpoints
│   ├── Dockerfile             # Backend container configuration
│   ├── requirements.txt       # Python dependencies
│   └── tests/
│       └── test_api.py        # API tests
├── frontend/
│   ├── readcrumbs_app.py      # Streamlit frontend application
│   ├── Dockerfile             # Frontend container configuration
│   └── requirements.txt       # Python dependencies
├── monitoring/
│   ├── app.py                 # Streamlit monitoring dashboard
│   ├── Dockerfile             # Monitoring container configuration
│   └── requirements.txt       # Python dependencies
├── experiment-tracking/
│   └── wandb.py               # W&B experiment tracking and model registry
├── experiments/
│   ├── training/
│   │   ├── preprocess.py      # Data preprocessing utilities
│   │   ├── train_model.py     # Model training script
│   │   └── utils.py           # Training utilities
│   └── notebooks/
│       └── eda.ipynb          # Exploratory data analysis
├── tests/
│   └── test_preprocess.py     # Preprocessing tests
├── data/
│   └── README.md              # Data documentation
├── docker-compose.yml          # Multi-container orchestration
├── requirements.txt           # Root-level dependencies
└── README.md                  # This file
```

### Key Files

- **`backend/api.py`**: Main FastAPI application with prediction endpoints
- **`frontend/readcrumbs_app.py`**: User-facing Streamlit interface
- **`monitoring/app.py`**: Monitoring and analytics dashboard
- **`experiment-tracking/wandb.py`**: W&B integration for model tracking
- **`docker-compose.yml`**: Container orchestration configuration

## Testing

### Running Tests

#### Backend API Tests

```bash
cd backend
pytest tests/test_api.py -v
```

#### Preprocessing Tests

```bash
pytest tests/test_preprocess.py -v
```

#### Run All Tests

```bash
pytest tests/ -v
```

### Test Coverage

The test suite includes:
- API endpoint functionality
- Health check verification
- Prediction endpoint validation
- Error handling tests
- Data preprocessing tests

### Example Test Output

```bash
$ pytest backend/tests/test_api.py -v

======================== test session starts ========================
backend/tests/test_api.py::test_health_check PASSED
backend/tests/test_api.py::test_get_random PASSED
backend/tests/test_api.py::test_predict PASSED
======================== 3 passed in 2.34s ========================
```

## Troubleshooting

### Common Issues

#### 1. Docker Containers Won't Start

**Symptoms**: `docker compose up` fails or containers exit immediately

**Solutions**:
```bash
# Check logs
docker compose logs

# Rebuild containers
docker compose build --no-cache

# Check if ports are in use
sudo lsof -i :8000
sudo lsof -i :8080
sudo lsof -i :8081
```

#### 2. AWS Credentials Not Working

**Symptoms**: S3 or DynamoDB access errors

**Solutions**:
```bash
# Verify credentials in .env file
cat .env | grep AWS

# Test AWS CLI access
aws s3 ls s3://readcrumbs/
aws dynamodb list-tables

# Check IAM permissions
aws iam get-user
```

#### 3. Model Not Loading

**Symptoms**: Backend starts but predictions fail

**Solutions**:
- Verify model file exists in S3: `aws s3 ls s3://readcrumbs/models/`
- Check model file path in `backend/api.py`
- Verify S3 bucket permissions

#### 4. DynamoDB Connection Issues

**Symptoms**: Logs not saving or monitoring dashboard empty

**Solutions**:
```bash
# Verify table exists
aws dynamodb describe-table --table-name readcrumbs-logs

# Check table permissions
aws iam get-role-policy --role-name YourRoleName --policy-name YourPolicyName

# Verify table name in environment variables
echo $DDB_TABLE
```

#### 5. Frontend Not Connecting to Backend

**Symptoms**: Frontend shows errors when requesting predictions

**Solutions**:
- Check API URL in `frontend/readcrumbs_app.py`
- Verify backend is running: `curl http://localhost:8000/health`
- Check CORS settings if needed
- Verify network connectivity between containers

#### 6. Monitoring Dashboard Shows No Data

**Symptoms**: Dashboard loads but shows empty charts

**Solutions**:
- Verify DynamoDB table has data: `aws dynamodb scan --table-name readcrumbs-logs --limit 5`
- Check table name matches in `monitoring/app.py`
- Verify AWS credentials for monitoring container
- Make some predictions first to generate data

### Debugging Tips

1. **View Container Logs**:
   ```bash
   docker compose logs backend
   docker compose logs frontend
   docker compose logs monitoring
   ```

2. **Access Container Shell**:
   ```bash
   docker compose exec backend bash
   docker compose exec frontend bash
   ```

3. **Check Environment Variables**:
   ```bash
   docker compose exec backend env | grep AWS
   ```

4. **Test API Manually**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"items": ["test"], "userid": "123"}'
   ```

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to the branch: `git push origin feature/your-feature-name`
8. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

### Reporting Issues

If you encounter a bug or have a feature request, please open an issue on GitHub with:
- Description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, Docker version)

## License

This project is open source and available for educational and research purposes.

## Authors

Developed by **Jordan Sinclair** and **Jordan Larson**

- GitHub: [smiley-maker/readcrumbs](https://github.com/smiley-maker/readcrumbs)

## Acknowledgments

- Built with FastAPI, Streamlit, and Docker
- Model tracking powered by Weights & Biases
- Deployed on AWS infrastructure

---

For questions or support, please open an issue on the GitHub repository.
