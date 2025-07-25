FROM python:3.12-slim

WORKDIR /app

# Copy code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Unbuffered output (for logs)
ENV PYTHONUNBUFFERED=1

# Expose port (FastAPI default is 8000)
EXPOSE 8000

# Start the app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]