import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model
from src.config import PROJ_DIR, SPECTROGRAMS_DIR
from data_processing import load_and_prepare_data

#Get the dataframe using the imported function from data_processing.py
df = load_and_prepare_data()

#Reorganize df to only include the samples that have a corresponding spectrogram
# (ideally temporary section, hopefully all 25863 mp3s end up being converted)
print(f"Original number of samples: {len(df)}")
def spectrogram_exists(clip_id):
    path = os.path.join(SPECTROGRAMS_DIR, f"{clip_id}.npy")
    return os.path.exists(path)
df['spec_exists'] = df['clip_id'].apply(spectrogram_exists)
df = df[df['spec_exists']].copy()
df.drop(columns=['spec_exists'], inplace=True)
print(f"Found {len(df)} existing spectrograms for training.")

# --- Build the data pipeline ---
#Create lists of file paths and their corresponding labels
file_paths = [os.path.join(SPECTROGRAMS_DIR, f"{clip_id}.npy") for clip_id in df['clip_id']]

#Drop 'path' in addition to the other two, as it was the helper column that was used for spectrogram generation
non_label_cols = ['clip_id', 'mp3_path', 'path']
label_cols = [col for col in df.columns if col not in non_label_cols]
labels = df[label_cols].values.astype(np.float32)

#Create a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

def load_and_preprocess(file_path, label):
    spectrogram = np.load(file_path.numpy())
    spectrogram = tf.expand_dims(spectrogram, axis=-1) #Add channel dimension
    return spectrogram, label

def map_func(file_path, label):
    spectrogram, label = tf.py_function(
        load_and_preprocess,
        [file_path, label],
        [tf.float32, tf.float32]
    )

    num_classes = label.shape[0] 
    spectrogram.set_shape([96, None, 1]) #Shape: (height, width, channels)  |  NOTE: height should always correspond to n_mels value in create_mel_spectrogram
    label.set_shape([num_classes])       #Shape: (number_of_tags,)

    return spectrogram, label

#Build the final pipeline

#Shuffle and map the entire dataset of individual files
DATASET_SIZE = len(df) # len(df) is the number of samples
dataset = dataset.shuffle(buffer_size=DATASET_SIZE)
dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)

#Split the dataset into training and validation sets
train_size = int(0.8 * DATASET_SIZE)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

#Batch and prefetch the training and validation sets separately
batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

print("TensorFlow training and validation pipelines created successfully.")

# --- Callback ---

#The file path where the best model will be saved
checkpoint_filepath = os.join(PROJ_DIR, "app", "best_music_model.keras")

#Create the ModelCheckpoint callback
#It will monitor the validation loss and save only the best model
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',      # Monitor the loss on the validation set
    mode='min',              # We want to minimize the loss
    save_best_only=True      # Only save the model if val_loss has improved
)

#Create the EarlyStopping callback
#It will monitor the validation loss and stop model training if 
# there are 5 consecutive epochs with no improvement.

#restore_best_weights: If True, the model weights from the epoch with the best
# value of the monitored quantity will be restored at the end of training.
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,  #Stop training if `val_loss` doesn't improve for 5 consecutive epochs.
    verbose=1,
    restore_best_weights=True
)

# --- Train the model ---
input_shape = (96, None, 1)
num_classes = labels.shape[1]

model = build_model(input_shape, num_classes)

#Compile the model
#No "accuracy" metric to avoid misleading information
#Example: A lazy model that predicts 0 for every label would get a high accuracy 
# because only 3 labels are present out of the 188 anyways
model.compile(optimizer='adam',
              loss='binary_crossentropy', #Loss for multi-label
              metrics=[
                'AUC',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')])

print("\nStarting model training...")

model_training_time_start = time.time()
history = model.fit(
    train_dataset,
    epochs=100, #High number because EarlyStopping will handle the rest
    validation_data=val_dataset,
    callbacks=[checkpoint_callback, early_stopping_callback]  #Callbacks passed here
)
model_training_time_end = time.time()
total_model_training_time = model_training_time_end - model_training_time_start

print(f"\nTraining complete! The best model was saved to {checkpoint_filepath}")
print(f"Total model training time was: {total_model_training_time} seconds.")


