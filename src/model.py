from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    #CNN layers go from 16, 32, 64..., but ideally will be reverted to 32, 63, 128...
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(), #Using GlobalAveragePooling instead of Flatten because we just need to detect the presence of an audio feature, not exactly where it is in the song
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid') #Sigmoid for multi-label
    ])

    return model