FROM python:3.11-slim

WORKDIR /app

# Create non-root user (uid 1000) for security
RUN useradd -u 1000 -m appuser

# Install production dependencies only — no dev deps in final image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Grant ownership to non-root user
RUN chown -R appuser /app

USER appuser

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
