FROM python:3.6

ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ADD . .

EXPOSE 8000

# CMD ["gunicorn", "--bind", ":8000", "--workers", "3", "PortfolioApi.wsgi:application"]
CMD gunicorn PortfolioApi.wsgi:application --bind 0.0.0.0:$PORT