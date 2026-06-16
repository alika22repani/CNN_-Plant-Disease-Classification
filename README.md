# 🌿 Plant Disease Classification Web App

Aplikasi berbasis web untuk klasifikasi penyakit daun tanaman menggunakan Convolutional Neural Network (CNN) dan Flask.

## 📌 Deskripsi

Proyek ini dibuat untuk mengidentifikasi jenis penyakit pada daun tanaman berdasarkan gambar yang diunggah oleh pengguna. Model CNN dilatih menggunakan dataset PlantVillage dan diintegrasikan ke dalam aplikasi web menggunakan Flask.

## ✨ Fitur

* Upload gambar daun tanaman
* Prediksi penyakit daun menggunakan CNN
* Menampilkan tingkat kepercayaan (confidence)
* Menampilkan Top 3 hasil prediksi
* Antarmuka web yang mudah digunakan

## 🛠️ Teknologi yang Digunakan

* Python
* Flask
* TensorFlow / Keras
* NumPy
* Pillow
* HTML
* CSS

## 📂 Struktur Project

```text
CNN/
├── app.py
├── best_plant_model.keras
├── class_names.txt
├── requirements.txt
├── Procfile
├── runtime.txt
├── static/
├── templates/
└── README.md
```

## 🚀 Cara Menjalankan

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python app.py
```

Buka browser:

```text
http://127.0.0.1:5000
```

## 📊 Dataset

Dataset yang digunakan adalah PlantVillage Dataset yang berisi berbagai jenis penyakit daun tanaman dan daun sehat.

## 👩‍💻 Author

Alika
