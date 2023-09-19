from flask import Flask, request, send_file, render_template
import os
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        uploaded_image = request.files['image']
        processing_option = request.form['processing_option']

        print("Received request with processing_option:", processing_option)

        # create the 'uploads' directory
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # save the uploaded image 
        uploaded_image_path = 'uploads/temp_image.png'
        uploaded_image.save(uploaded_image_path)

        print("Saved uploaded image to:", uploaded_image_path)

        # TODO - apply our model logic here and replace the grayscale method
        if processing_option == 'sketches':
            # process the image to grayscale 
            processed_image = process_to_grayscale(uploaded_image_path)
        else:
            # return the original image
            with open(uploaded_image_path, 'rb') as image_file:
              processed_image = image_file.read()

        print("Processing complete")

         # create a file-like object
        processed_image_io = BytesIO(processed_image)

        return send_file(processed_image_io, mimetype='image/png')

    except Exception as e:
        print("Error:", str(e))
        return str(e)

def process_to_grayscale(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # convert to grayscale
    grayscale_data = np.array(img)
    grayscale_image = Image.fromarray(grayscale_data)

    with BytesIO() as buffer:
        grayscale_image.save(buffer, format="PNG")
        return BytesIO.getvalue(buffer)

if __name__ == '__main__':
    app.run(debug=True)
