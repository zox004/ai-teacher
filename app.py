from flask import Flask, request, render_template, redirect, send_file, jsonify
from werkzeug.utils import secure_filename
import model as md
from pymongo import MongoClient
import pymongo
import os
from gridfs import GridFS

client = MongoClient("mongodb+srv://aiteacher:1234@aiteacher.2urehvj.mongodb.net/?retryWrites=true&w=majority")
db = client.aiteacher
collection = db.data
gfs = GridFS(db)

app = Flask(__name__)

@app.route('/')
def hellohtml():
    return render_template("hello.html")


@app.route('/upload_class1', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' :
        files = request.files.getlist("file1[]")

        for f in files:
            file_name = f.filename
            data = f.read()
            content_type = f.content_type
            insertImg = gfs.put(data, content_type=content_type, file_name=file_name)
        
        # return str(insertImg)
        # return redirect("/")
            # img = {'img' : f}
            # db.data.insert_one(img)
        return redirect("/")
        # return jsonify({'msg' : '저장되었습니다'})


@app.route('/upload_class2', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST' :
        files = request.files.getlist("file2[]")

        for f in files:
            f.save('./data/train/class2/' + secure_filename(f.filename))
        return redirect("/")

@app.route('/train', methods=['GET', 'POST'])
def train() :
    if request.method == 'POST' :

        md.train()

        return redirect("/")

@app.route('/predict', methods=['GET', 'POST'])
def img_prediction() :
    if request.method == 'POST' :
        os.makedirs('./data/test',exist_ok=True)
        f = request.files['prediction_file']
        f.save('./data/test/' + secure_filename(f.filename))
        md.prediction()

        return redirect("/")
        # return render_template("/prediction")


@app.route('/download', methods = ['GET', 'POST'])
def download_file():
    PATH = 'weight/model_best_epoch.pt'
    return send_file(PATH, as_attachment=True)
    # files_list = os.listdir("./weight")
    # if request.method == 'POST':
    #     sw = 0
    #     for x in files_list:
    #         if(x==request.form['downloadfile']):
    #             sw=1
    #     try:
    #         path = "./weight/"
    #         return send_file(path + request.form['downloadfile'],
    #                 download_name = request.form['downloadfile'],
    #                 as_attachment=True)
    #     except:
    #         print("download error")
    # return render_template('hello.html', files=files_list)


if __name__ == '__main__' :
    app.run(debug=True,host='0.0.0.0')
    if app.config['DEBUG']:
        app.config['SEND_FILE_MAX_AGE_DEFAULT']