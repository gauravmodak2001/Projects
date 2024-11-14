# app.py
from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
from keras.models import load_model
import base64
from io import BytesIO
import soundfile as sf

app = Flask(__name__)
model = load_model('model\trained_model_trail_1_GCP.h5')

CHORD_LABELS = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']
SR = 16000  # Sampling rate
N_MFCC = 80

def extract_mfcc(audio):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def classify_chord(audio_data):
    mfcc_features = extract_mfcc(audio_data)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    prediction = model.predict(mfcc_features)
    predicted_class = np.argmax(prediction, axis=1)
    return CHORD_LABELS[predicted_class[0]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        audio_file = request.files['file']
        audio, _ = librosa.load(audio_file, sr=SR)
    else:
        audio_data = request.json['audio_data']
        audio_bytes = BytesIO(base64.b64decode(audio_data))
        audio, _ = sf.read(audio_bytes)

    predicted_chord = classify_chord(audio)
    return jsonify({'predicted_chord': predicted_chord})

if __name__ == '__main__':
    app.run(debug=True)
