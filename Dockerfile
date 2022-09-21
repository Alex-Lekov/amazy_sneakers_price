#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
FROM python:3.10-buster

RUN pip install --upgrade pip
#RUN pip install --upgrade setuptools

COPY requirements.txt ./

RUN pip install -r requirements.txt
#RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN mkdir app
#COPY ./app /app

#WORKDIR /app

#CMD ["uvicorn", "--host","0.0.0.0", "--port", "8008","main:app"]
#CMD ["python", "main.py"]