import numpy as np
import pandas as pd
import os
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import preprocess_input

import tensorflow as tf
app=Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG']= True
app.config['TESTING'] = True
@app.route('/')
def hello_world():
    return 'hello world'
if __name__=='__main__':
    app.run()
@app.route('/index')
def home():
    return render_template('index.html')

    