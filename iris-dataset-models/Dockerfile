FROM python:3.9-slim
WORKDIR /usr/src/app
COPY requirements.txt ./
COPY iris_model.py ./
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt
# entrypoint
ENTRYPOINT ["python", "iris_model.py"]