from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import model as md

app = Flask(__name__)

@app.route('/')
def hellohtml():
    return render_template("hello.html")


@app.route('/upload_class1', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' :
        files = request.files.getlist("file1[]")

        for f in files:
            f.save('./data/train/cat/' + secure_filename(f.filename))

        return redirect("/")


@app.route('/upload_class2', methods=['GET', 'POST'])
def upload_file2():
    if request.method == 'POST' :
        files = request.files.getlist("file2[]")

        for f in files:
            f.save('./data/train/dog/' + secure_filename(f.filename))
        return redirect("/")


@app.route('/train', methods=['GET', 'POST'])
def train_model() :
    if request.method == 'POST' :
        md.train()
        
        return redirect("/")
    
@app.route('/predict', methods=['GET', 'POST'])
def img_prediction() :
    if request.method == 'POST' :
        f = request.files['prediction_file']
        f.save('./data/test/' + secure_filename(f.filename))
        md.prediction()
        
        return redirect("/")
        
        

if __name__ == '__main__' :
    app.run(debug=True)
    if app.config['DEBUG']:
	    app.config['SEND_FILE_MAX_AGE_DEFAULT']
