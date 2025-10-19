"""
Quick fine-tune script for jaundice model
Requires: dataset/ with train/valid/test structure and classes folders
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

DATA_DIR = 'dataset'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
MODEL_PATH = os.path.join('..', 'models', 'jaunenet_full_model.h5')
SAVE_PATH = os.path.join('..', 'models', 'jaunenet_full_model_finetuned.h5')

# Load model with custom objects if available
models_path = os.path.join(os.getcwd(), 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)
try:
    from ConvNeXt import LayerScale, StochasticDepth
    custom_objects = {'LayerScale': LayerScale, 'StochasticDepth': StochasticDepth}
except Exception:
    custom_objects = None

print('Preparing data generators...')
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
valid_generator = valid_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'valid'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print('Loading model...')
if os.path.exists(MODEL_PATH):
    if custom_objects:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    else:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
else:
    print('Base model not found:', MODEL_PATH)
    sys.exit(1)

# Unfreeze some top layers
for layer in model.layers:
    layer.trainable = False
# Make last block trainable (heuristic)
for layer in model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

print('Starting fine-tuning...')
history = model.fit(train_generator,
                    validation_data=valid_generator,
                    epochs=EPOCHS,
                    callbacks=callbacks)

print('Fine-tune complete. Model saved to', SAVE_PATH)
