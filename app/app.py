from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import time

print("--- Script starting, imports are complete. ---")

# --- Initialize Flask App ---
app = Flask(__name__)
# Create a temporary folder for uploads if it doesn't exist
if not os.path.exists('app/uploads'):
    os.makedirs('app/uploads')
app.config['UPLOAD_FOLDER'] = 'app/uploads'


# --- Load Model and Labels ---
print("--- Loading Keras model... (This may take a moment) ---")
start_time = time.time()
try:
    MODEL = load_model('app/best_music_model.keras')
    end_time = time.time()
    print(f"--- Model loaded successfully in {end_time - start_time:.2f} seconds. ---")
except Exception as e:
    print(f"!!! ERROR: Could not load model. Please check the file path. Error: {e}")
    MODEL = None

print("--- Loading label list... ---")
try:
    with open('app/label_cols.txt', 'r') as f:
        LABEL_COLS = [line.strip() for line in f.readlines()]
    print("--- Labels loaded successfully. ---")
except Exception as e:
    print(f"!!! ERROR: Could not load label_cols.txt. Please run data_processing.py. Error: {e}")
    LABEL_COLS = []


# --- Helper Functions for Prediction ---
def create_spectrogram_for_chunk(audio_chunk, sr, n_mels=96):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=n_mels, fmax=sr/2)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def analyze_audio_file(audio_path, window_sec=29, overlap_sec=10):
    if MODEL is None or not LABEL_COLS:
        return {"error": "Model or labels not loaded."}

    predictions = []
    # This function is already wrapped in a try/except in the route, which is good.
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    window_samples = int(window_sec * sr)
    overlap_samples = int(overlap_sec * sr)
    step_samples = window_samples - overlap_samples

    for i in range(0, len(y) - window_samples + 1, step_samples):
        chunk = y[i:i + window_samples]
        
        spec = create_spectrogram_for_chunk(chunk, sr)
        spec_for_pred = np.expand_dims(np.expand_dims(spec, axis=0), axis=-1)
        
        chunk_pred = MODEL.predict(spec_for_pred, verbose=0)[0] # Added verbose=0 for cleaner logs
        predictions.append(chunk_pred)

    if not predictions: # Handle very short audio clips
        return {"tags": []} # Return empty list instead of error for UI consistency

    avg_probabilities = np.mean(predictions, axis=0)
    
    top_k = 10
    min_confidence = 0.2
    
    descending_indices = np.argsort(-avg_probabilities)
    top_k_indices = descending_indices[:top_k]
    
    final_tags = []
    for i in top_k_indices:
        if avg_probabilities[i] > min_confidence:
            tag_info = {
                "tag": LABEL_COLS[i],
                "confidence": f"{avg_probabilities[i] * 100:.1f}%"
            }
            final_tags.append(tag_info)
    
    return {"tags": final_tags}


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print(f"Received request.files: {request.files}")

    if 'audio_file' not in request.files or request.files['audio_file'].filename == '':
        print("!!! ERROR: No file part or no selected file.")
        return jsonify({"error": "No file selected."}), 400
    
    file = request.files['audio_file']
    filepath = None  # Initialize filepath to None

    try:
        # Save the file securely
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.mp3')
        file.save(filepath)
        print(f"--- File saved temporarily to {filepath} ---")
        
        # Run the analysis
        results = analyze_audio_file(filepath)
        print("--- Analysis complete. ---")
        
        return jsonify(results)

    except Exception as e:
        # This will catch errors from file.save() or predict_long_audio()
        print(f"!!! AN UNEXPECTED ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc() # Print the full error traceback for detailed debugging
        return jsonify({"error": "An internal error occurred during analysis."}), 500

    finally:
        # This block will run whether there was an error or not
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            print(f"--- Temporary file {filepath} removed. ---")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for cleaner production-like testing

