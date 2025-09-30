import librosa
import numpy as np
import os
import time
from tqdm import tqdm
import pandas as pd
from src.config import MP3_DIR, ANNOTATIONS_PATH, SPECTROGRAMS_DIR, APP_DIR

#Get full path from mp3 directory
def get_full_path(relative_path_col):
    #Join the main MP3 directory with the path from the CSV
    return os.path.join(MP3_DIR, relative_path_col)

#Create spectrogram
def create_mel_spectrogram(audio_path, duration=29):
    """
    Generates a standardized Mel-spectrogram from an audio file.
    """
    try:
        y, sr = librosa.load(audio_path, duration=duration, sr=None)
        target_length = int(duration * sr)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=sr/2) #n_mels changed from 128 to 96 to speed up training, but ideally will be changed back
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error processing {os.path.basename(audio_path)}: {e}")
        return None

#Process annotations file from MagnaTagATune
def load_and_prepare_data(annotations_path=ANNOTATIONS_PATH):
    """
    Loads and parses the annotations file, adds the full file path,
    and returns a clean DataFrame. This is the function you will import.
    """
    print("Loading and preparing annotations data...")
    # (Your manual parsing logic from the notebook goes here)
    processed_data = []
    with open(annotations_path, 'r') as f:
        header_line = f.readline()
        column_names = [name.strip('"') for name in header_line.strip().split('\t')]
        for line in f:
            if not line.strip(): continue
            values = [val.strip('"') for val in line.strip().split('\t')]
            row_dict = {column_names[i]: val for i, val in enumerate(values)}
            processed_data.append(row_dict)
    
    df = pd.DataFrame(processed_data)
    if column_names:
        df = df[column_names]

    #Defragmenting DataFrame
    df = df.copy()

    #----------------------------------------------
    print("Saving label columns for the web app...")

    non_label_cols = ['clip_id', 'mp3_path']
    label_cols = [col for col in df.columns if col not in non_label_cols]
    
    labels_output_path = os.path.join(APP_DIR, 'label_cols.txt')
    #os.makedirs(APP_DIR, exist_ok=True)
    
    with open(labels_output_path, 'w') as f:
        for label in label_cols:
            f.write(f"{label}\n")
    print(f"Successfully saved {len(label_cols)} labels to {labels_output_path}")
    #----------------------------------------------

    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    #Defragmenting DataFrame
    df = df.copy()

    #Add the full path to each audio file
    df['path'] = df['mp3_path'].apply(get_full_path)
    
    print("DataFrame prepared successfully.")
    return df

#Create df from annotations and generate spectrograms
# default limit for spectrograms set to the first 10k due to storage restrictions
def generate_spectrograms(df, limit=10000):
    """
    Takes a DataFrame and generates spectrograms for the files listed in it.
    """
    print(f"\nProcessing audio and saving spectrograms to '{SPECTROGRAMS_DIR}'...")
    start_time = time.time()
    
    # Use df.head(limit) to process only the first 'limit' rows
    for index, row in tqdm(df.head(limit).iterrows(), total=limit, desc="Generating Spectrograms"):
        audio_path = row['path']
        clip_id = row['clip_id']
        output_path = os.path.join(SPECTROGRAMS_DIR, f"{clip_id}.npy")
        
        if os.path.exists(output_path):
            continue
            
        mel_spec = create_mel_spectrogram(audio_path)
        if mel_spec is not None:
            np.save(output_path, mel_spec)

    end_time = time.time()
    print("\nProcessing complete!")
    print(f"Spectrogram generation took {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    #Run this by doing python src/data_processing.py in console
    #Get the clean DataFrame
    main_df = load_and_prepare_data()
    
    #Use the DataFrame to generate the spectrograms
    generate_spectrograms(main_df, limit=10000)