from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
import gdown

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- PROSES DOWNLOAD & LOAD MODEL DARI GOOGLE DRIVE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_1_PATH = os.path.join(BASE_DIR, 'best_plant_model.keras')
MODEL_2_PATH = os.path.join(BASE_DIR, 'model_cnn_plantvillage.keras')

# Pastikan folder upload ada sebelum aplikasi jalan
os.makedirs(os.path.join(BASE_DIR, UPLOAD_FOLDER), exist_ok=True)

# Otomatis download Model 1 jika belum ada di server Railway
if not os.path.exists(MODEL_1_PATH):
    print("Mendownload best_plant_model.keras dari Google Drive...")
    id_model1 = '191qxbAlp6NSyRTANEwneFeVYe5303uJC'
    url_model1 = f'https://drive.google.com/uc?id={id_model1}'
    gdown.download(url_model1, MODEL_1_PATH, quiet=False, fuzzy=True)

# Load model utama yang dipakai untuk aplikasi Flask
model = tf.keras.models.load_model(MODEL_1_PATH)

# Jika suatu saat kamu mau pakai model cnn satunya, tinggal hilangkan pagar di bawah:
# model_cnn = tf.keras.models.load_model(MODEL_2_PATH)


# --- LOAD CLASS NAMES ---
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_idx]) * 100
    
    # Top 3 prediksi
    top3_idx = np.argsort(predictions[0])[::-1][:3]
    top3 = [
        {
            'label': format_label(class_names[i]),
            'confidence': round(float(predictions[0][i]) * 100, 2)
        }
        for i in top3_idx
    ]
    
    return format_label(class_names[predicted_idx]), round(confidence, 2), top3


def format_label(label):
    # Ubah "Apple___Apple_scab" jadi "Apple - Apple Scab"
    label = label.replace('___', ' - ')
    label = label.replace('_', ' ')
    return label.title()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada file yang diunggah!')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='Pilih file terlebih dahulu!')
    
    if not allowed_file(file.filename):
        return render_template('index.html', error='Format file harus PNG, JPG, atau JPEG!')
    
    # Simpan file dengan nama unik
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Prediksi
    label, confidence, top3 = predict_image(filepath)
    
    # Tentukan status kesehatan
    is_healthy = 'healthy' in label.lower()
    
    return render_template('result.html',
        filename=filename,
        label=label,
        confidence=confidence,
        top3=top3,
        is_healthy=is_healthy
    )


if __name__ == '__main__':
    app.run(debug=True)