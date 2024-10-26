import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import librosa

# Load the pre-trained model
model = tf.keras.models.load_model('/Users/neelpatel/Desktop/SER/SER_model.h5')


# Define the emotions (adjust based on your model's output)
emotions = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

# Function to extract MFCC features from audio file
def extract_features(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path, sr=None)
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Return the mean of the MFCC features
    return np.mean(mfccs.T, axis=0)

# Function to handle file upload and prediction
def upload_file():
    # Open file dialog to select a .wav file
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        # Display the selected file path
        label_file_explorer.config(text="File Opened: " + file_path)
        
        # Extract features from the uploaded audio file
        features = extract_features(file_path)
        
        # Reshape the features to match the input shape of the model
        features = features.reshape(1, -1)
        
        # Perform prediction using the pre-trained model
        predictions = model.predict(features)
        
        # Get the emotion with the highest probability
        predicted_emotion = emotions[np.argmax(predictions)]
        
        # Display the predicted emotion
        label_result.config(text="Emotion Detected: " + predicted_emotion)

# Create the main window
window = tk.Tk()
window.title('Speech Emotion Detector')
window.geometry('500x300')

# Create a label for file explorer
label_file_explorer = tk.Label(window, text="Upload a .wav file", width=50, height=4)
label_file_explorer.grid(column=1, row=1)

# Create a button to browse files
button_explore = tk.Button(window, text="Browse Files", command=upload_file)
button_explore.grid(column=1, row=2)

# Create a label for displaying the result
label_result = tk.Label(window, text="", width=50, height=4)
label_result.grid(column=1, row=3)

# Start the GUI event loop
window.mainloop()
