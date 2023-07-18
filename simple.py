import numpy as np
import os 
from flask import Flask, request, render_template 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

model=load_model(r"model.h5", compile=False)
app=Flask(__name__, static_folder='templates/photos')

#Render html page
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/index.html')
def home():
    return render_template('index.html')
@app.route('/about.html')
def about():
    return render_template('about.html')
@app.route('/images.html') 
def sub():
    return render_template('images.html')
@app.route('/predict.html')
def predict():
    return render_template("predict.html")

@app.route('/predict', methods=["GET", "POST"]) 
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath, 'uploads',f.filename) 
        f.save(filepath)
        img=image.load_img(filepath,target_size=(180,180,3))
        x=image.img_to_array(img) 
        x=np.expand_dims(x,axis=0)
        img_data=preprocess_input(x)
        prediction=np.argmax(model.predict(img_data), axis=1)
        
        index=['cloudy','foggy', 'rainy','shine', 'sunrise']

        result=str(index [prediction[0]]) 
        print(result)
        return render_template('predict.html', result=result)
    
    
if __name__ == "__main__" :
    app.run(debug=True)


    