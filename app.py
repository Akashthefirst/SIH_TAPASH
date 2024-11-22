from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_manager, login_required, logout_user, current_user, LoginManager
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import urllib.request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import pipeline

#-------------------------------------------------------------------------------MODEL-------------------------------------------------------------------------
#------------------------------------TRANSFORMERS PIPELINES----------------------------------------------
pipe_76 = pipeline("image-classification", model="shreyasguha/22class_skindiseases_76acc_possibleoverfit")
pipe_57 = pipeline("image-classification", model="shreyasguha/22class_skindiseases_57acc")
pipe_80 = pipeline("image-classification", model="shreyasguha/22class_skindiseases_80acc")

categories = [
    "Seborrheic Keratoses and other Benign Tumors",
    "Vascular Tumors",
    "Light Diseases and Disorders of Pigmentation",
    "Vasculitis Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Nail Fungus and other Nail Disease",
    "Exanthems and Drug Eruptions",
    "Systemic Disease",
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Lupus and other Connective Tissue diseases",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Eczema Photos",
    "Warts Molluscum and other Viral Infections",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Bullous Disease Photos",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Atopic Dermatitis Photos",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Urticaria Hives",
    "Herpes HPV and other STDs Photos"
]

def preprocess(image_path, mix=False):
    image = Image.open(image_path)
    ans1 = pipe_57(image)
    ans2 = pipe_76(image)
    ans3 = pipe_80(image)

    op = [ans1, ans2, ans3]
    s = [ans[0]['label'] for ans in op]
    ansf = [int(element.replace("LABEL_", "")) for element in s]
    conditions = [categories[ans] for ans in ansf]

    if(conditions[0] == conditions[1] and conditions[1] == conditions[2]):
        return conditions[0]
    if((conditions[0] != conditions[1] and conditions[1] != conditions[2]) or mix==False):
        return conditions[1]
    else:
        if(conditions[0] == conditions[1]):
            return conditions[0]
        elif(conditions[0] == conditions[2]):
            return conditions[0]
        else:
            return conditions[1]

device = "cpu"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['UPLOAD_FOLDER'] = 'D:\\SIH_TAPASH\\SIH_TAPASH\\static\\uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

# Create the database tables within an application context
with app.app_context():
    db.create_all()

class Registerform(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username is already taken')

class Loginform(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('first_page.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = Loginform()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = Registerform()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/tnc')
def tnc():
    return render_template('tnc.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = preprocess(file_path)
        return jsonify({'prediction': prediction})
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)