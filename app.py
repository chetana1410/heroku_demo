from flask import Flask, render_template, Response, flash, request, redirect, url_for, send_from_directory
import cv2
#import datetime, time
import os, sys
import numpy as np
#from threading import Thread
#import tensorflow as tf
import shutil
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
# from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.metrics import Recall, Precision
# from tensorflow.keras import backend as K 
from processing import generate_outputs


#input types
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#required directories
#removing
try:
    shutil.rmtree('./shots')
except :
    pass

try:
    shutil.rmtree('./static/images')
except :
    pass

try:
    shutil.rmtree('./static/results')
except :
    pass

#creating
try:
    os.mkdir('./shots')
except OSError as error:
    pass

try:
    os.mkdir('./static/images')
except :
    pass

try:
    os.mkdir('./static/results')
except :
    pass


global capture, grey, switch, neg
capture=0
grey=0
neg=0
switch=1


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')





def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(1)
    global out, capture
    while True:
        success, frame = camera.read() 
        if success:
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)

                                

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('home_page.html')
    
@app.route('/upload_imgs')   
def upload_imgs():
    return render_template('upload (1).html')


@app.route("/upload", methods=["POST"])
def upload():

    if request.form.get('sub') == 'Submit':
        for upload in request.files.getlist("file"):
            print(upload)
            print("{} is the file name".format(upload.filename))
            filename = upload.filename
            destination = "/".join(['shots', filename])
            print ("Accept incoming file:", filename)
            print ("Save it to:", destination)
            upload.save(destination)
        return render_template("complete.html", image_name=filename)

    elif  request.form.get('out') == 'Outputs':
        props = generate_outputs()
        images=os.listdir('./static/images')
        results=os.listdir('./static/results') 
        return render_template("out (1).html", images=images, results=results, props=props, len=len(images))


@app.route('/capture')   
def capture_img():
    return render_template('capture1.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg

        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                #cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        
        elif  request.form.get('out') == 'Outputs':
                props = generate_outputs()
                images=os.listdir('./static/images')
                results=os.listdir('./static/results') 
                return render_template("out (1).html", images=images, results=results, props=props, len=len(images))
        
           
    elif request.method=='GET':
        return render_template('capture1.html')
    
    return render_template('capture1.html')



if __name__ == '__main__':
    app.run(debug=True)
    
camera.release()
#cv2.destroyAllWindows()     
