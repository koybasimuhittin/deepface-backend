# syntax=docker/dockerfile:1
FROM python:3.8.12

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .


WORKDIR /app/src
RUN mkdir /database

EXPOSE 5000
# Use Waitress instead of Gunicorn
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "--call", "app:create_app"]

