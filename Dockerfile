FROM debian:latest
RUN apt-get update && apt-get install python3-pip -y && pip3 install fastapi uvicorn
COPY ./app /app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]




