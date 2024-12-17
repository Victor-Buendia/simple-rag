FROM python:3.10-slim

RUN pip install poetry poetry-plugin-export

COPY pyproject.toml ./
RUN poetry export -f requirements.txt --output requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY application application
COPY interfaces interfaces
COPY main.py config.py config.yaml ./

CMD [ "streamlit", "run", "main.py" ]