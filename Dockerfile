# Use Python 3.12 bullseye-slim image
FROM python:3.12-slim-bullseye

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Copy project files
COPY . .

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Expose port
EXPOSE 8501

ENV PORT=8501

# Run streamlit
CMD poetry run streamlit run main.py --server.port 8501 --server.address 0.0.0.0