# Lightweight Debian Docker image with Python preinstalled
FROM python:3.9.12-slim as builder
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc
# Create virtualenv:
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Install packages
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip \
&& pip install -r requirements.txt \
&& pip cache purge
COPY . /src

FROM python:3.9.12-slim as worker
COPY --from=builder /opt/venv /opt/venv

# Use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
COPY --from=builder /src .
EXPOSE 5000
# Use gunicorn WSGI server instead of built-in Flask server
ENTRYPOINT ["./gunicorn.sh"]