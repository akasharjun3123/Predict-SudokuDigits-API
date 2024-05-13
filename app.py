from flask import Flask, request, app, jsonify, url_for, render_template, redirect
import numpy as np
import pandas as pd
import pickle
from mainProcess import *
from flask_cors import CORS 




app = Flask(__name__)


@app.route('/')
def home():
    return render_template('entry.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    data = main(img)
    data = [int(x) for x in data]
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
    
    
