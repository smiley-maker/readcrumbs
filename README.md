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
   - Serves a `/feedback` endpoint for saving user feedback to DynamoDB.
   - Logs all requests to DynamoDB for monitoring
   - Handles model inference with supporting lookup tables

2. **Frontend Interface**: A Streamlit application that:
   - Provides a user-friendly interface for inputting favorite books
   - Displays real-time recommendations from the FastAPI backend
   - Handles user interactions and API communication
   - Provides an option for feedback on recommendations. 

3. **Model Monitoring Dashboard**: A Streamlit dashboard that:
   - Connects directly to DynamoDB to visualize prediction logs
   - Tracks prediction latency over time
   - Collects and displays user feedback for model accuracy
   - Calculates Recall at K based on new feedback.

## Technology Stack

- **Backend**: FastAPI, Python 3.12+
- **Frontend**: Streamlit
- **ML Framework**: scikit-learn, joblib, implicit for ALS model
- **Cloud Services**: AWS (S3, DynamoDB, EC2)
- **Experiment Tracking**: Weights & Biases (W&B)
- **Containerization**: Docker
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

## Prerequisites

Before running the code locally, ensure you have the following installed and configured:

### System Requirements

- **Python**: 3.12 or higher
- **Docker**: 20.10 or higher (if running locally)
- **Git**: For cloning the repository
- **AWS**: If planning to run using EC2 and AWS.  

### AWS Account Setup

1. **AWS Account**: Create an AWS account if you don't have one
2. **AWS CLI**: Install and configure AWS CLI (optional for local development)
3. **Required AWS Services**:
   - **S3 Bucket**: Named `readcrumbs` (or update code to use your bucket name)
     - Store model files: `models/als_model-small-v1.pkl`
     - Store lookup tables: `data/v1/index_to_title.json`, `data/v1/title_to_index.json`
   - **DynamoDB Table**: Named `readcrumbs-logs`
     - Used for storing prediction logs
   - **DynamoDB Table**: Named `readcrumbs-feedback`
     - Used to store user feedback on provided predictions.
   - **EC2 Instance**: For production deployment
     - Recommended: t3.medium or larger
     - Ubuntu 22.04 LTS or Amazon Linux 2

### Weights & Biases Setup

1. Create a W&B account at [wandb.ai](https://wandb.ai)
2. Install W&B: `pip install wandb` or run in Docker container
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

If not running inside Docker container on EC2 with IAM role, you should create a `.env` file in each separate part of the project (never commit this file to git). Here's the structure:

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

## Running the Project With EC2

- Create four EC2 containers in AWS for the frontend, monitoring dashboard, backend server, and training the model, respectively. 
- SSH into each container, add the files using the public IP, and create a Docker container. 
- Run the Docker container, which should start each service. 

#### 4. Access the Services

Once running, access the services:

- **FastAPI Backend API**: http://localhost:8000 (or http://ec2-ip-address:8000)
  - Health Check: http://localhost:8000/health
  - API Docs: http://localhost:8000/docs
  - Prediction endpoint: http://localhost:8000/predict
  - Feedback endpoint: http://localhost:8000/feedback
- **Frontend Interface**: http://localhost:8501/ (or http://ec2-ip-address:8501)
- **Monitoring Dashboard**: http://localhost:8501/ (or http://ec2-ip-address:8501)

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
docker run -p 8501:8501 readcrumbs-frontend
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
     - **Port 22**: SSH (for initial setup)
  - Allow access via HTTP and SSH. 

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

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
```

#### 3. Clone Repository

```bash
git clone https://github.com/smiley-maker/readcrumbs
cd readcrumbs
```

Or you can copy the files from your local computer to the EC2 container using:
```bash
scp -r -i path/to/your/key.pem folder/to/pass ubuntu@ipv4-address:~/       
```

#### 4. Set Up Environment Variables

```bash
# Create .env file
nano .env
```

Add your environment variables (see [Environment Variables](#environment-variables) section).

**Note**: If using IAM roles, you may only need `AWS_REGION` and `DDB_TABLE`.


### Troubleshooting Deployment

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

**CURL Example**:
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
│   ├── dashboard_app.py       # Streamlit monitoring dashboard
│   ├── Dockerfile             # Monitoring container configuration
│   └── requirements.txt       # Python dependencies
├── experiments/
│   ├── training/
│   │   ├── preprocess.py      # Data preprocessing utilities
│   │   ├── train_model.py     # Model training script
│   └── notebooks/
│       └── eda.ipynb          # Exploratory data analysis
|   |-- tracking/
│       └── wandb_tracking.py  # W&B experiment tracking and model registry
├── data/
│   └── README.md              # Data documentation
└── README.md                  # This file
```

### Key Files

- **`backend/api.py`**: Main FastAPI application with prediction endpoints
- **`frontend/readcrumbs_app.py`**: User-facing Streamlit interface
- **`monitoring/dashboard_app.py`**: Monitoring and analytics dashboard
- **`experiments/tracking/wandb_tracking.py`**: W&B integration for model tracking

## Testing

### Running Tests

#### Backend API Tests

```bash
cd backend
pytest tests/test_api.py -v
```

#### Preprocessing Tests

```bash
cd experiments/training
pytest tests/test_preprocess.py -v
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

## License

This project is open source and available for educational and research purposes.

## Authors

Developed by **Jordan Sinclair** and **Jordan Larson**

- GitHub: [smiley-maker/readcrumbs](https://github.com/smiley-maker/readcrumbs)