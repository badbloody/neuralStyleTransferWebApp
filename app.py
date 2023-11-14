from flask import Flask, request, send_file, render_template
import os
from PIL import Image
import numpy as np
from io import BytesIO
import random
import torch
from torchvision.transforms import transforms

from models import *

app = Flask(__name__, static_url_path='', static_folder='static')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = 32
transform_net = TransformNet(base).to(device)
metanet = MetaNet(transform_net.get_param_dict()).to(device)
metanet.load_state_dict(torch.load('style_transfer_models/metanet_base32_style20_tv1e-06.pth', map_location=torch.device('cpu')))
transform_net.load_state_dict(torch.load('style_transfer_models/metanet_base32_style20_tv1e-06_transform_net.pth', map_location=torch.device('cpu')))
print("Loaded models successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        uploaded_image = request.files['image']
        processing_option = request.form['processing_option']
        print("Received request with processing_option:", processing_option)

        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # save the uploaded image 
        uploaded_image_path = 'uploads/temp_image.png'
        uploaded_image.save(uploaded_image_path)
        print("Saved uploaded image to:", uploaded_image_path)

        processed_image = perform_style_transfer(uploaded_image_path, processing_option)
        print("Processing complete")

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

def perform_style_transfer(content_image_path, processing_option):
    style_folder = os.path.join('static', 'style_images', processing_option)
    style_images = os.listdir(style_folder)
    random_style_image = random.choice(style_images)
    style_image_path = os.path.join(style_folder, random_style_image)

    content_image = Image.open(content_image_path)
    style_image = Image.open(style_image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    content_tensor = preprocess(content_image).unsqueeze(0)
    style_tensor = preprocess(style_image).unsqueeze(0)

     # Get style features using VGG
    style_features = vgg16(style_tensor)
    style_mean_std = mean_std(style_features)

    # Style transfer parameters
    style_weight_mine = 20
    content_weight = 1
    tv_weight = 1e-6

    # Define optimizer for the models
    optimizer = optim.Adam([
        {'params': metanet.parameters()},
        {'params': transform_net.parameters()}
    ], lr=1e-3)

    # Perform style transfer
    n_batch = 13  # Number of batches for training
    for batch in range(n_batch):
        optimizer.zero_grad()

        # Calculate weights from metanet
        weights = metanet.forward(mean_std(style_features))
        transform_net.set_weights(weights, 0)

        # Get transformed images
        transformed_images = transform_net(content_tensor)

        # Calculate content features
        content_features = vgg16(content_tensor)
        transformed_features = vgg16(transformed_images)
        transformed_mean_std = mean_std(transformed_features)

        # Calculate losses
        content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])
        style_loss = style_weight_mine * F.mse_loss(transformed_mean_std,
                                                    style_mean_std.expand_as(transformed_mean_std))

        y = transformed_images
        tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                               torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        loss = content_loss + style_loss + tv_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Get the stylized image from the transform_net
    stylized_image = transform_net(content_tensor)

    output_image = transforms.ToPILImage()(stylized_image.squeeze(0).cpu())
    output_image_path = 'transferred_files/output_image.png' 
    output_image.save(output_image_path)

    return output_image_path

    

if __name__ == '__main__':
    app.run(debug=True)
