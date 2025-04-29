# app.py
import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.preprocessing import image # type: ignore
import joblib
 
# Load models
bagging_model = joblib.load('models/bagging_model.pkl')
boosting_model = joblib.load('models/boosting_model.pkl')
stacking_model = joblib.load('models/stacking_model.pkl')

app = Flask(__name__)

# HTML template for uploading image
UPLOAD_FORM = '''
    <!doctype html>
    <title>Upload Image for Prediction</title>
    <h1>Upload an Image (Lion or Cheetah)</h1>
    <form method=post enctype=multipart/form-data action="/predict">
      <input type=file name=file>
      <input type=submit value=Predict>
    </form>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(UPLOAD_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    img_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(img_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Flatten image for stacking model
    img_flat = img_array.reshape((img_array.shape[0], -1))

    # Predict using each model
    bagging_pred = bagging_model.predict(img_flat)[0]
    boosting_pred = boosting_model.predict(img_flat)[0]
    stacking_pred = stacking_model.predict(img_flat)[0]

    # Map class index to label
    labels = {0: 'Cheetah', 1: 'Lion'}
    
    return f'''
    <h1>Prediction Results</h1>
    <p><b>Bagging:</b> {labels[bagging_pred]}</p>
    <p><b>Boosting:</b> {labels[boosting_pred]}</p>
    <p><b>Stacking:</b> {labels[stacking_pred]}</p>
    <a href="/">Upload Another Image</a>
    '''

if __name__ == '__main__':
    app.run(debug=True)
