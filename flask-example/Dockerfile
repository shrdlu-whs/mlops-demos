# Lightweight Debian Docker image with Python
FROM python:3.9.12-slim

RUN pip install --upgrade pip

# Install packages
RUN pip install numpy
# Install more compact Pytorch for CPU only 
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install tokenizers
RUN pip install transformers
RUN pip install flask
RUN pip install flask-restful
RUN pip install gunicorn

COPY . /app
WORKDIR /app

# Use gunicorn WSGI server instead of built-in Flask server
ENTRYPOINT ["./gunicorn.sh"]