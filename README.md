# ReadCrumbs

This is an end-to-end Machine Learning Operations (MLOps) project designed to deliver personalized book recommendations in a production-ready environment. The system includes experiment tracking (W&B), a model registry, a FastAPI serving backend, persistent logging, and separate user and monitoring interfaces, all containerized and ready for deployment on AWS EC2.

## Core System Components

The architecture is split into three main containerized services:

1. ML Model Backend: A Python FastAPI application that loads the "Production" Matrix Factorization model from the Model Registry (W&B), serves predictions via a /predict endpoint, and logs all requests to the persistent DynamoDB database.
2. Frontend Interface: A Streamlit application allowing users to input favorite books and get recommendations from the FastAPI backend.
3. Model Monitoring Dashboard: A dedicated Streamlit/Python dashboard that connects directly to the database to visualize live prediction latency, data drift, and model performance metrics.


2. Clone the Repository

```bash
git clone https://github.com/smiley-maker/readcrumbs
cd readcrumbs
```