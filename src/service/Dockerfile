FROM python:3.13-slim

ENV POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app
RUN curl -sSL https://install.python-poetry.org | python3 -
# RUN apt-get update && apt-get install python3-dev
RUN pip install poetry
COPY ./src/ ./src/
COPY ./data ./data
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock


RUN poetry install
RUN mkdir ./cache
RUN poetry run python ./src/model.py
ENTRYPOINT ["poetry"]
CMD ["run", "python", "-m", "flask", "--app", "./src/main.py", "run", "--host=0.0.0.0"]
