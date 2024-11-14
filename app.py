import keras
import librosa
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from matplotlib.animation import FuncAnimation
import librosa.display
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings, WebRtcMode
import base64

# Replace 'background.jpg' with the path to your image file
with open("image.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Define constants
sr = 16000  # Sampling rate
n_mfcc = 80  # Number of MFCC features
N_CLASSES = 8  # Number of classes
CHORD_LABELS = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']  # Update as needed

# Load the trained model
model_path = 'model/trained_model_trail_1_GCP.h5'
model = load_model(model_path)

# Function to extract MFCC features
def extract_mfcc(audio, sr):
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Function to classify uploaded audio
def classify_chord(audio):
    mfcc_features = extract_mfcc(audio, sr)
    mfcc_features = np.expand_dims(mfcc_features, axis=0)
    prediction = model.predict(mfcc_features)
    predicted_class = np.argmax(prediction, axis=1)
    return CHORD_LABELS[predicted_class[0]]

# Function to plot the spectrogram
def plot_spectrogram(audio, sr):
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    st.pyplot(plt)

# Audio processor to capture and process audio frames
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv_audio(self, frame):
        audio = frame.to_ndarray()
        self.audio_buffer.append(audio)
        return frame

st.markdown(
    f"""
    <style>
        body {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            color: white;
        }}
        .chord-button {{
            display: inline-block;
            margin: 5px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            border: 2px solid #4CAF50;
            background-color: #f1f1f1;
            color: black;
            transition: background-color 0.3s, color 0.3s;
        }}
        .chord-button.highlight {{
            background-color: #4CAF50;
            color: white;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
            100% {{ transform: scale(1); }}
        }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🎸 Guitar Chord Classifier")
st.write("Upload or record a WAV file to classify the guitar chord and visualize its spectrogram.")

# Audio recording option
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True}
    ),
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx and webrtc_ctx.audio_processor:
    audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_buffer, axis=0)
    if len(audio_data) > 0:
        audio_data = audio_data.astype(np.float32)
        audio_data /= np.max(np.abs(audio_data), axis=0)
        st.audio(audio_data, format='audio/wav')

        # Classify chord
        predicted_chord = classify_chord(audio_data)
        st.write(f"Predicted Chord: **{predicted_chord}**")
        
        # Display chord buttons
        st.write("Chord Prediction:")
        cols = st.columns(4)
        for i, chord in enumerate(CHORD_LABELS):
            is_predicted = (chord == predicted_chord)
            button_style = "highlight" if is_predicted else ""
            cols[i % 4].markdown(f"<div class='chord-button {button_style}'>{chord}</div>", unsafe_allow_html=True)

        # Plot and display spectrogram
        st.write("Spectrogram of the recorded audio:")
        plot_spectrogram(audio_data, sr)

# File upload for classification
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    # Load audio file
    audio, _ = librosa.load(uploaded_file, sr=sr)
    st.audio(uploaded_file, format='audio/wav')
    
    # Classify chord
    predicted_chord = classify_chord(audio)
    
    # Display buttons for each chord label
    st.write("Chord Prediction:")
    cols = st.columns(4)
    for i, chord in enumerate(CHORD_LABELS):
        is_predicted = (chord == predicted_chord)
        button_style = "highlight" if is_predicted else ""
        cols[i % 4].markdown(f"<div class='chord-button {button_style}'>{chord}</div>", unsafe_allow_html=True)
    
    # Plot and display spectrogram
    st.write("Spectrogram of the uploaded audio:")
    plot_spectrogram(audio, sr)

    # Display the predicted chord output
    st.write(f"Predicted Chord: **{predicted_chord}**")
