# Use lightweight Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create Streamlit config folder inside app
RUN mkdir -p /app/.streamlit

# Environment variable to point Streamlit to writable config
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit in headless mode
CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", "--server.address", "0.0.0.0"]
