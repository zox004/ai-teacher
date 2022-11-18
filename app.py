from flask import Flask, request, render_template
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
@app.route('/hello')
def hellohtml():
    return render_template("hello.html")


@app.route('/predict', methods=['GET', 'POST'])
def method():
    if request.method == 'POST' :
        file = request.file['upload1']
        if not file : return "No Files"
        image_bytes = file.read()

        upload_image = Image.open(io.BytesIO(image_bytes))
        upload_image.save("./static/img/", "png")

if __name__ == '__main__':
    app.run(debug=True)
    if app.config['DEBUG']:
	    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1