from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

device = "cpu"

app = Flask(__name__)

UPLOAD_FOLDER = 'C:\\SIH_Tapash\\SIH_TAPASH\\Penul\\static\\uploads'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

disease_names = {
    0: "melanoma",
    1: "benign keratosis-like lesions",
    2: "basal cell carcinoma",
    3: "actinic keratoses",
    4: "squamous cell carcinoma",
    5: "dermatofibroma",
    6: "vascular lesions",
    7: "pigmented benign keratosis"
}

def find_max(values):
    max_value = values[0]
    for value in values:
        if value.item() > max_value.item():
            max_value = value
    return max_value

def find_second_highest(values):
    highest_value = max(values, key=lambda x: x.item())
    second_highest_value = None
    for value in values:
        if value != highest_value and (second_highest_value is None or value.item() > second_highest_value.item()):
            second_highest_value = value
    return second_highest_value

class ModelFunction(nn.Module):
    def __init__(self):
        super(ModelFunction, self).__init__()
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input_shape=(3, 28, 28) because PyTorch expects (channels, height, width)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256 * 1 * 1, 128),  # Adjust the input size here based on the final output size of the conv layers
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 7),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

model = ModelFunction()
model.load_state_dict(torch.load('C:\\SIH_Tapash\\SIH_TAPASH\\Penul\\model\\skind.pth', map_location=torch.device('cpu')))

def preprocess(image_path):
    img = Image.open(image_path)
    data_transform = transforms.Compose([
        transforms.Resize(size=(28, 28)),
        transforms.ToTensor()
    ])
    transformed_image = data_transform(img)
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
    with torch.inference_mode():
        model.eval()
        outputs = model(transformed_image)
        _, predicted = torch.max(outputs, 1)
        ind = predicted[0].item()
        maxi = find_max(outputs[0])
        secondmaxi = find_second_highest(outputs[0])
        if maxi.item() - secondmaxi.item() > 0.5:
            return disease_names[ind]
        return "Model is unsure, please see a doctor."

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Image successfully uploaded and displayed below')

        # Call the preprocess function with the uploaded image path
        prediction = preprocess(file_path)

        return render_template('index.html', filename=filename, prediction=prediction)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
