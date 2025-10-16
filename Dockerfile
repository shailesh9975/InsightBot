# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by pypdf and potentially other libraries
# For pypdf, we sometimes need libicu-dev for unicode support, though often not strictly necessary for basic use.
# For other libs (like certain vector DBs or image processing), you might need more.
# For now, let's keep it lean. If issues arise, consider adding:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libicu-dev \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit application
# We use environment variables for API keys, which will be passed during `docker run` or by orchestration.
# The `CMD` instruction runs our app.py when the container starts.
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
