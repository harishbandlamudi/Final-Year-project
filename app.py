
# Importing essential python libraries

from __future__ import division, print_function
import os
import numpy as np
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# Flask utils
from flask import Flask,  url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array



os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Define a flask app
app = Flask(__name__, static_url_path='')


app.config['UPLOAD_FOLDER'] = 'uploads'
MODEL_PATH = 'outmodel'
model1 = load_model('unet_brain_mri_seg.hdf5', compile=False)

#Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')
med=["*Medication:* Chemotherapy drugs like temozolomide, targeted therapy drugs like bevacizumab, and tumor-treating fields (TTFields) may be prescribed.",
    "- *Medication:* Manage symptoms with medications and consider chemotherapy or targeted therapy if needed.",
    "No tumor Detected ",
    "*Medication:* Regulate hormone levels with medications like bromocriptine, cabergoline, or somatostatin analogs."
    ]

prec=["*Precautions:* Minimize radiation exposure, maintain a healthy lifestyle, and undergo regular check-ups.",
       "*Precautions:* Regular check-ups, avoid unnecessary radiation exposure, manage hormonal factors.",
       "",
       "*Precautions:* Monitor hormone levels, maintain a healthy weight, manage stress."]

cs=["*Causes:* Genetic mutations, radiation exposure, and certain genetic disorders.",
      "*Causes:* Spontaneous development, older age, radiation exposure, hormonal factors.",
      "",
      "*Causes:* Genetic factors, genetic syndromes, and rarely, radiation exposure."
      ]  

classes = ['glioma','meningioma','notumor','pituitary']


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def model_predict(img_path, model):
    print(img_path)
    
    imgp = load_img(img_path, target_size = (180,180,3))
    imgp = img_to_array(imgp)
    imgp = imgp/255
    imgp = np.expand_dims(imgp,axis=0)
    h=model.predict(imgp,verbose=0)
    p=np.argmax(h)
    mn=np.max(h)*100
    
    
    img = cv2.imread(img_path)
    img = cv2.resize(img ,(128,128))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model1.predict(img)
    img1=np.squeeze(pred) > .5
    mask_image = (img1 * 180).astype(np.uint8)  # Convert to uint8 format
    mask_image=cv2.resize(mask_image,(512,512))
    cv2.imwrite("./uploads/img.jpg", mask_image)
        
    
    
    return classes[p],med[p],mn,prec[p],cs[p]


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
       
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        file_name=os.path.basename(file_path)
        pred,medi,acu,pr,cau = model_predict(file_path, model)
        print(file_name)
        
        fname="img.jpg"
            
        
    return render_template('predict.html',file_name=file_name, fname=fname,result=pred,sugg=medi,ac=acu,pr=pr,cau=cau)




if __name__ == '__main__':
        app.run()

