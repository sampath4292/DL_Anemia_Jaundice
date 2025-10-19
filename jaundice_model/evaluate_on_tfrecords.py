"""
Evaluate jaundice model on TFRecord test set
Produces confusion matrix and per-class metrics
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Add models path for custom layers
models_path = os.path.join(os.getcwd(), 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

try:
    from ConvNeXt import LayerScale, StochasticDepth
except Exception:
    LayerScale = None
    StochasticDepth = None

# Config
TEST_TFRECORD = 'dataset/test.tfrecord'
MODEL_PATH = os.path.join('..', 'models', 'jaunenet_full_model.h5')
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3
BATCH_SIZE = 16


def parse_example(serialized_example):
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'meta': tf.io.FixedLenFeature([9], tf.float32)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_image(example['image_raw'], channels=CHANNELS, dtype=tf.dtypes.float32)
    # Preprocess exactly as training
    h = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    image = tf.image.resize_with_crop_or_pad(image, h, h)
    zoom_rate = 1.05
    if h < IMAGE_WIDTH:
        image = tf.image.resize_with_crop_or_pad(image, int(IMAGE_WIDTH * zoom_rate), int(IMAGE_WIDTH * zoom_rate))
    else:
        image = tf.image.resize(image, (int(IMAGE_WIDTH * zoom_rate), int(IMAGE_WIDTH * zoom_rate)))
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    label = example['label']
    return image, label


def load_dataset(tfrecord_path):
    if not os.path.exists(tfrecord_path):
        print(f"TFRecord not found: {tfrecord_path}")
        return None
    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    print("Evaluating jaundice model on TFRecord test set")
    print("Looking for:", TEST_TFRECORD)

    ds = load_dataset(TEST_TFRECORD)
    if ds is None:
        print("No TFRecord test set found. Please create TFRecords or restore `dataset/` directory.")
        return 1

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return 2

    print("Loading model... this may take a moment")
    try:
        if LayerScale is not None and StochasticDepth is not None:
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'LayerScale': LayerScale, 'StochasticDepth': StochasticDepth}, compile=False)
        else:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 3

    y_true = []
    y_pred = []

    for batch in ds:
        X_batch, y_batch = batch
        preds = model.predict(X_batch, verbose=0)
        preds_idx = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds_idx.tolist())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    return 0

if __name__ == '__main__':
    sys.exit(main())
