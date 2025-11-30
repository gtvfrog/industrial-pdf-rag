install:
	pip install -r requirements.txt

run:
	cd backend && uvicorn app.main:app --reload

test:
	cd backend && pytest -q

format:
	black backend/

lint:
	flake8 backend/

clean:
	rm -rf models_cache/* metrics_history/*

all: install run

docker-build-backend:
	docker build -t rag-backend -f backend/Dockerfile .

docker-build-frontend:
	docker build -t rag-frontend -f frontend/Dockerfile .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down
