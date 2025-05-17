FROM python:3.12-slim

WORKDIR /app

RUN apt update && apk add git ffmpeg -y

RUN git clone https://github.com/Ioann44/webrtc-cv-public .

RUN pip install -r resources/requirements.txt