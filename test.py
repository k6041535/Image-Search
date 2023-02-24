from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os

app = Flask(__name__)

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'gif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp、gif"})
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/imgupload', secure_filename(f.filename))
        f.save(upload_path)
        #return render_template('upload.html')
        return render_template('show.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()