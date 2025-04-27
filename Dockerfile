# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Chainlit runs on (default 8501 or 7860, check your app)
EXPOSE 8501

# Set environment variables (optional, or use docker-compose)
# ENV OPENAI_API_KEY=your-key

# Start the app (adjust if you use a different entrypoint)
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8501"]