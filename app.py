from flask import Flask
from flask import render_template, jsonify
from flask import request
from datetime import timedelta
from werkzeug.utils import secure_filename
from PIL import Image
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np
app = Flask(__name__)

def gifSplit(src_path, dest_path, suffix = "png"):
    img = Image.open(src_path)
    for i in range(3):
        img.seek(i)
        new = Image.new("RGBA", img.size)
        new.paste(img)
        new.save(os.path.join(dest_path, "%d.%s" %(i, suffix)))
        
def resize(img):
    image = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
    return image

def addimg(lena_binary, new_image):
    res = cv2.add(lena_binary, new_image)
    return res

def p_hash(img):

    img = cv2.resize(img, (32, 32))   
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    dct = cv2.dct(np.float32(gray))
    
    dct_roi = dct[0:8, 0:8]
 
    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def hamming(str1, str2):
    if len(str1) != len(str2):
        return
    count = 1
    for i in range(0, len(str1)):
        if str1[i] != str2[i]:
            count += 1
    return count

def cmpHash(hash1, hash2):
    n = 1
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

def classify_hist_with_split(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

def return_img_stream(img_local_path):
    
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream
 
"""
@app.route('/')
def hello_world():
        img_path = ('C:/Users/kaiyuchou/flask/static/images/1.gif')
        img_stream = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/2.gif')
        img_stream2 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/3.gif')
        img_stream3 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/4.gif')
        img_stream4 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/5.gif')
        img_stream5 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/6.gif')
        img_stream6 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/7.gif')
        img_stream7 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/8.gif')
        img_stream8 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/9.gif')
        img_stream9 = return_img_stream(img_path)
        img_path = ('C:/Users/kaiyuchou/flask/static/images/10.gif')
        img_stream10 = return_img_stream(img_path)
        return render_template('index.html'
                            , img_stream=img_stream
                            , img_stream2=img_stream2
                            , img_stream3=img_stream3
                            , img_stream4=img_stream4
                            , img_stream5=img_stream5
                            , img_stream6=img_stream6
                            , img_stream7=img_stream7
                            , img_stream8=img_stream8
                            , img_stream9=img_stream9
                            , img_stream10=img_stream10)
"""
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
        return render_template('show.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
    gifSplit('C:/Users/kaiyuchou/flask/static/imgupload/29.gif', 'C:/Users/kaiyuchou/Multimedia/result')
    lena = cv2.imread('C:/Users/kaiyuchou/Multimedia/result/0.png')
    lena2 = cv2.imread("C:/Users/kaiyuchou/Multimedia/result/1.png")
    
    lena = resize(lena)
    lena2 = resize(lena2)
    
    x =  lena.shape[0] #寬尺寸
    y =  lena.shape[1] #長尺寸
    
    lena_binary = np.zeros((64, 64), int)
    
    lena_binary = cv2.add(lena, lena2)
    
    for i in range(2, 3):
        image = cv2.imread("C:/Users/kaiyuchou/Multimedia/result/%d.png"%(i))
        new_image = resize(image)
        for i in range(x):
            for j in range(y):
                lena_binary[i][j] = lena_binary[i][j] + image[i][j]
    for i in range(x):
        for j in range(y):
            lena_binary[i][j] = lena_binary[i][j] / 4
    
    
    ans = np.array(range(135)) 
    for i in range(135):
        ans[i] = 0
    sorce = np.zeros((135, 100), int)
    
    for k in range(1, 133):
        gifSplit('C:/Users/kaiyuchou/Multimedia/gifs/%d.gif'%(k), r'C:/Users/kaiyuchou/Multimedia/data')       
    
        com = cv2.imread('C:/Users/kaiyuchou/Multimedia/data/0.png')
        com2 = cv2.imread("C:/Users/kaiyuchou/Multimedia/data/1.png")
        
        com = resize(com)
        com2 = resize(com2)
        
        x2 =  com.shape[0] #寬尺寸
        y2 =  com.shape[1] #長尺寸
        #print(x2)
    
        com_binary = np.zeros((64, 64), int)
    
        com_binary = cv2.add(com, com2)
        
        
        
        for i in range(2, 3):
            image2 = cv2.imread("C:/Users/kaiyuchou/Multimedia/data/%d.png"%(i))
            new_image2 = resize(image2)
            for q in range(x):
                for w in range(y):
                    #print(w)
                    com_binary[q][w] = com_binary[q][w] + image2[q][w]
                
        for i in range(x2):
            for j in range(y2):
                com_binary[i][j] = com_binary[i][j] / 4
        
        h1 = p_hash(lena_binary)
        h2 = p_hash(com_binary)
        
        result = cmpHash(h1, h2)
        rgb = np.round(classify_hist_with_split(lena_binary, com_binary))
        result = rgb + result
        result = result.astype('int64')
        print(result)
        sorce[k][result] = 1
        #print(sorce)
        
    b = 0
    for h in range(99):
        for y in range(135):
            if sorce[y][h] == 1:
                ans[b] = y
                b += 1
    print(sorce)
    print(ans)
    
    for i in range(10):
        a = ans[i]
        if a > 0:
            gif = Image.open('C:/Users/kaiyuchou/Multimedia/gifs/%d.gif'%(a))
            gif.thumbnail( (400,100) ) 
            print(gif.size)
            gif.save(os.path.join(r'C:/Users/kaiyuchou/flask/static/images', "%d.gif" %(i+1)))
