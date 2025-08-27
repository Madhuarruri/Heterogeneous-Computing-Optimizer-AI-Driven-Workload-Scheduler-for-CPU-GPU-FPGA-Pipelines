
# SmartCompute Optimizer - Development Makefile

.PHONY: dev test build clean docker-build docker-run bench demo-data

# Development setup
dev:
	@echo "Starting development environment..."
	docker-compose up --build

# Run tests
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v
	python test_kernels.py

# Build production images
build:
	@echo "Building production containers..."
	docker-compose -f docker-compose.prod.yml build

# Clean up containers and images
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Build Docker images
docker-build:
	@echo "Building Docker images..."
	docker build -t smartcompute-backend ./backend
	docker build -t smartcompute-frontend ./frontend

# Run with Docker
docker-run:
	@echo "Running application with Docker..."
	docker-compose up -d

# Run benchmarks
bench:
	@echo "Running performance benchmarks..."
	python benchmarks.py

# Populate demo data
demo-data:
	@echo "Populating demo data..."
	python scripts/populate_demo_data.py

# Code quality checks
lint:
	@echo "Running code quality checks..."
	ruff check .
	mypy backend/

# Database migrations
migrate:
	@echo "Running database migrations..."
	alembic upgrade head

# Setup development environment
setup:
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Run smoke tests
smoke-test:
	@echo "Running smoke tests..."
	python -c "import requests; print('Backend:', requests.get('http://localhost:8000/health').status_code)"
	curl -f http://localhost:3000 || echo "Frontend health check"

# Generate API documentation
docs:
	@echo "Generating API documentation..."
	python -c "import uvicorn; uvicorn.run('main:app', host='localhost', port=8000)" &
	sleep 5
	curl -o api-docs.json http://localhost:8000/openapi.json
	pkill -f uvicorn

help:
	@echo "Available commands:"
	@echo "  dev         - Start development environment"
	@echo "  test        - Run test suite"
	@echo "  build       - Build production containers"
	@echo "  clean       - Clean up Docker resources"
	@echo "  bench       - Run performance benchmarks"
	@echo "  demo-data   - Populate demo data"
	@echo "  lint        - Run code quality checks"
	@echo "  migrate     - Run database migrations"
	@echo "  setup       - Setup development environment"
	@echo "  smoke-test  - Run basic health checks"
	@echo "  docs        - Generate API documentation"
