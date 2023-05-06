# Lightweight Debian Docker image with Python
FROM python:3.9.12-slim
# Install packages
COPY requirements.txt ./
RUN pip install --upgrade pip \
&& pip install -r requirements.txt \
&& pip cache purge

COPY . /app
WORKDIR /app

# Use gunicorn WSGI server instead of built-in Flask server
ENTRYPOINT ["./gunicorn.sh"]