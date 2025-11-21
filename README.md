# ReadCrumbs

This is an end-to-end Machine Learning Operations (MLOps) project designed to deliver personalized book recommendations in a production-ready environment. The system includes experiment tracking (W&B), a model registry, a FastAPI serving backend, persistent logging, and separate user and monitoring interfaces, all containerized and ready for deployment on AWS EC2.

## Core System Components

The architecture is split into three main containerized services:

1. ML Model Backend: A Python FastAPI application that loads the "Production" Matrix Factorization model from the Model Registry (W&B), serves predictions via a /predict endpoint, and logs all requests to the persistent DynamoDB/RDS store.
2. Frontend Interface: A React application allowing users to input books and view real-time recommendations from the FastAPI backend.
3. Model Monitoring Dashboard: A dedicated Streamlit/Python dashboard that connects directly to the database to visualize live prediction latency, data drift, and model performance metrics.


## Local Setup and Installation

Follow these steps to get the environment ready for development:

1. Prerequisites
You should have Docker downloaded on your system, and follow the steps below to set up an environment. 

```bash
# Create virtual environment or conda environment.
# Conda:
conda create -n readcrumbs -y 
conda activate readcrumbs
# -- or create a virtual environment --
python -m venv venv
source venv/bin/activate

# Install dependencies for whole project
pip install -r requirements.txt
```

2. Clone the Repository

```bash
git clone https://github.com/smiley-maker/readcrumbs
cd readcrumbs
```

3. Environment Variables

DO NOT COMMIT YOUR SECRETS TO GIT.

Copy the structure from the example file to create your local secrets file:

```bash
cp .env.example .env
```

Fill in the actual, sensitive values (API keys, passwords, etc.) into the new .env file.

## Running the Project Locally

The entire system is containerized and managed via docker-compose. This allows us to run the three main services (Backend API, Frontend, Monitoring) simultaneously.

1. Build Containers

Build the Docker images for all services defined in the docker-compose.yml file:

```bash
docker compose build
```

2. Run All Services

Start the entire MLOps system in detached mode:

```bash
docker compose up -d
```

3. Accessing the Services

Once running, you can access the three key components in your browser:

- FastAPI Backend API (Health Check): http://localhost:8000/health
- Frontend Interface: http://localhost:8080/
- Monitoring Dashboard: http://localhost:8081/

4. Shut Down

To stop and remove the containers:

```bash
docker compose down
```