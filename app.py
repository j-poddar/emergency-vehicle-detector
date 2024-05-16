from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Define the upload folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load the trained VGG16 model
model = load_model('model_vgg16.h5')

# Define the prediction function
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return 'Emergency Vehicle'
    else:
        return 'Not an Emergency Vehicle'

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')

        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')

        # If the file exists and is allowed
        if file:
            # Create the uploads folder if it doesn't exist
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
           
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template('result.html', prediction=prediction, uploaded_image=file_path)

    return render_template('index.html')

# Define the route for the 'Predict another Vehicle' button
@app.route('/predict_another', methods=['GET', 'POST'])
def predict_another():
    return redirect(url_for('upload_image'))

if __name__ == '__main__':
    app.run(debug=True)