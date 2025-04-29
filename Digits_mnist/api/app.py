from flask import Flask, request, jsonify, render_template
from ensemble import EnsembleAggregator
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
aggregator = EnsembleAggregator()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Open the image using PIL
        image = Image.open(file.stream)
        
        # Convert to grayscale and resize (28x28 for MNIST-like models)
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize image to 28x28
        
        # Convert to numpy array and normalize it (between 0 and 1)
        image_array = np.array(image) / 255.0  # Normalize image
        
        # Add the batch dimension (required by models)
        image_array = np.expand_dims(image_array, axis=0)  # Shape (1, 28, 28)

        # Get the selected method from the form
        method = request.form.get('method', 'bagging')

        if method == 'ab':
            preds = aggregator.a_b_test(image_array)
        else:
            preds = aggregator.predict(image_array, method)
        
        # Return prediction result
        return jsonify({'prediction': int(preds[0])})
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
