FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install torch \
    pip install flask \
    pip install werkzeug \
    pip install pymongo \
    pip install torchvision \
    pip install bs4 \
    pip install selenium

COPY . /app


ENTRYPOINT python app.py
EXPOSE 5000

CMD ["python", "app.py"]
