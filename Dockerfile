FROM python:3.7

# Set up environment
WORKDIR /app

COPY Pipfile* /app/

RUN set -ex && pip install pipenv && \
    pipenv install --deploy --system

COPY . /app

# Run Flask app
ENTRYPOINT ["python"]
CMD ["service.py"]