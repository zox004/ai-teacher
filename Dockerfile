FROM python:3.8

COPY . .

WORKDIR /home/vagrant/flaskImage

RUN git clone https://github.com/zox004/ai-teacher.git
RUN pip install flask

WORKDIR ai-teacher/

RUN pip install -r requirements.txt

ENTRYPOINT python app.py
EXPOSE 5000

CMD ["python", "app.py"]
