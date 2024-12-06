<!-- write Readme for this file -->

docker build -t my-fastapi-app .
docker run -d -p 8000:8000 my-fastapi-app /app/runs/run.ini
## MLOps Course Track

This project is part of the MLOps course track for Fall 2024. It demonstrates how to build and run a FastAPI application using Docker.

### Prerequisites

- Docker installed on your machine
- Basic knowledge of FastAPI and Docker

### Building the Docker Image

To build the Docker image, run the following command:

```sh
docker build -t my-fastapi-app .
```

### Running the Docker Container

To run the Docker container, use the command below:

```sh
docker run -d -p 8000:8000 my-fastapi-app /app/runs/run.ini
```

This will start the FastAPI application and make it accessible at `http://localhost:8000`.

### Project Structure

- `/app`: Contains the FastAPI application code
- `/app/runs`: Directory for configuration files and runtime settings

### License

This project is licensed under the MIT License.