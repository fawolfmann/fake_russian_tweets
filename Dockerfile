FROM python:3.8

WORKDIR /code

COPY . /code
RUN pip install . -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "api:app", "--port", "8000", "--host",  "0.0.0.0"]
