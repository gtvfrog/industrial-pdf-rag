FROM python:3.10-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY frontend/ frontend/
COPY config/ config/

# Expose port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "frontend/Chat.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
