FROM python:3.9-slim
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY iris-model.py ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
# Run model when the container launches
CMD ["python", "./iris-model.py"]