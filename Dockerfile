# Dockerfile
FROM python:3.9-slim
WORKDIR /root
COPY requirements.txt /root/
RUN pip install -r requirements.txt
COPY app.py /root/
COPY model /root/model
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["app.py"]
