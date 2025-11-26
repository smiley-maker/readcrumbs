FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application code into the container at /app
COPY api.py /app/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the API server when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]