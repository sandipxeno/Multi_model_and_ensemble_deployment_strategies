from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np
import joblib
from utils import load_models, preprocess_image, bagging_predict, boosting_predict, stacking_predict, ab_testing, low_latency_ensemble
from PIL import Image

app = Flask(__name__)

# Load models
bagging_models, boosting_model, stacking_cnn, meta_model, class_names = load_models()

# Define allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Helper function to check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join('uploads', filename)
        file.save(image_path)
        
        image_tensor = preprocess_image(image_path)
        
        method = request.form.get('method')
        result = None
        if method == 'bagging':
            result = bagging_predict(bagging_models, image_tensor, class_names)
        elif method == 'boosting':
            result = boosting_predict(boosting_model, image_tensor, class_names)
        elif method == 'stacking':
            result = stacking_predict(bagging_models, stacking_cnn, boosting_model, meta_model, image_tensor, class_names)
        elif method == 'ab_testing':
            result = ab_testing(bagging_models, boosting_model, stacking_cnn, meta_model, image_tensor, class_names)
        elif method == 'low_latency':
            result = low_latency_ensemble(bagging_models, boosting_model, stacking_cnn, meta_model, image_tensor, class_names)
        
        return render_template('result.html', prediction=result[0], confidence=result[1] if len(result) > 1 else None)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
