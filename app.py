from flask import Flask, request, render_template, redirect, send_file, make_response
from werkzeug.utils import secure_filename
import model as md
from model import models
from pymongo import MongoClient
import os, uuid
from gridfs import GridFS
from s3 import s3_connection, s3_put_object
from s3_config import AWS_S3_BUCKET_NAME

client = MongoClient("mongodb+srv://aiteacher:1234@aiteacher.2urehvj.mongodb.net/?retryWrites=true&w=majority")
db = client.aiteacher
collection = db.data
gfs = GridFS(db)
s3 = s3_connection()
app = Flask(__name__)


@app.route('/')
def hellohtml():
    
    return render_template("hello.html")

@app.route('/start')
def startAIteacher():
    global uid
    uid = str(uuid.uuid4())
    
    return render_template("hello.html")

@app.route('/upload_class1', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' :
        files = request.files.getlist("file1[]")

        for f in files:
            filename = secure_filename(f.filename)
            acl = "public-read"
            ret = s3.upload_fileobj(f, AWS_S3_BUCKET_NAME, "class1/"+secure_filename(f.filename), ExtraArgs={"ACL": acl, "ContentType" : f.content_type})
            if ret == True:
                print("파일 저장 성공")
            else :
                print(uid)
            # saved = f.save('./data/train/class1/' + secure_filename(f.filename))
            # ret = s3_put_object(s3, AWS_S3_BUCKET_NAME, './data/train/class1/', saved)

        # for f in files:
        #     file_name = f.filename
        #     data = f.read()
        #     content_type = f.content_type
        #     insertImg = gfs.put(data, content_type=content_type, file_name=file_name)

        return redirect("/")
        # return jsonify({'msg' : '저장되었습니다'})


@app.route('/upload_class2', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST' :
        files = request.files.getlist("file2[]")

        for f in files:
            filename = secure_filename(f.filename)
            acl = "public-read"
            ret = s3.upload_fileobj(f, AWS_S3_BUCKET_NAME, "class2/"+secure_filename(f.filename),
            ExtraArgs={"ACL": acl, "ContentType" : f.content_type})
            if ret == True:
                print("파일 저장 성공")
            else :
                print("파일 저장 실패")

        
            file_name = f.filename
            data = f.read()
            content_type = f.content_type
            insertImg = gfs.put(data, content_type=content_type, file_name=file_name)
        # for f in files:
        #     f.save('./data/train/class2/' + secure_filename(f.filename))
        return redirect("/")

@app.route('/train', methods=['GET', 'POST'])
def train() :
    if request.method == 'POST' :
        
        md.train()
        # model_doc = {"model_name": "my_model", "model_state_dict": model.state_dict()}
        # collection.insert_one(model_doc)
        # gfs.put(model_doc)
        # collection.insert_one(model_doc)
        # model = torch.load(archtecture, weights)
        # architecture_file = io.StringIO(archtecture)
        # model = torch.load(architecture_file, weights)

        # acl = "public-read"
        # ret = s3.upload_fileobj(model, AWS_S3_BUCKET_NAME, "class2/model.pt",
        # ExtraArgs={"ACL": acl})



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
    PATH = 'weight/model.pt'

    return send_file(PATH, as_attachment=True)

if __name__ == '__main__' :
    app.run(debug=True,host='0.0.0.0')
    if app.config['DEBUG']:
        app.config['SEND_FILE_MAX_AGE_DEFAULT']

