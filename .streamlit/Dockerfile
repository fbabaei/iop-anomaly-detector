# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /home/user/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set Streamlit config dir so it doesn’t try writing to "/.streamlit"
ENV STREAMLIT_CONFIG_DIR=/home/user/app/.streamlit

# Expose port for Streamlit
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
