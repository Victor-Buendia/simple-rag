all:
	docker build -t local-rag:latest .
	docker run --name local-rag -p 8501:8501 --restart unless-stopped local-rag:latest
lint:
	autoflake -r --quiet --remove-all-unused-imports --exclude venv -i .
	black .