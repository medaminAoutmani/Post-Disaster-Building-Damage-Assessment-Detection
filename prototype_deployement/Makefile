.PHONY: build up down test logs shell-be shell-fe seed

build:
	podman-compose build

up:
	podman-compose up -d

down:
	podman-compose down -v

test:
	podman-compose exec backend pytest tests/ -v --tb=short

logs:
	podman-compose logs -f backend

shell-be:
	podman-compose exec backend bash

shell-fe:
	podman-compose exec frontend sh

seed:
	podman-compose exec backend python /app/scripts/seed_demo_data.py