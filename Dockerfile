# Dockerfile
FROM python:3.9-slim
WORKDIR /root
COPY requirements.txt /root/
RUN apt-get -y update
RUN apt-get -y install git
RUN pip install -r requirements.txt
COPY ml_models /root/ml_models
COPY app.py /root/
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["app.py"]
