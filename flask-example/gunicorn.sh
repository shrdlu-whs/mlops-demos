#!/bin/sh
echo "$PATH"
gunicorn  app:app --timeout 1000 -w 2 -b 0.0.0.0:5000