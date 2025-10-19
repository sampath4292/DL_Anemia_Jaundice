# -*- coding: utf-8 -*-
"""
Author       : Yuanting Ma (adapted)
Description  : Training script for JaunENet (Anemia & Jaundice detection)
"""

from __future__ import absolute_import, division, print_function
import os
import random
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet, densenet, VGG19, ResNet50V2, InceptionV3, MobileNetV3Large, Xception
from models.ConvNeXt import ConvNeXtLarge
from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras import activations
# from models.VisonTransformer import create_vit_model
from prepare_data import generate_datasets, load_and_preprocess_image

# GPU setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0] 
    tf.config.experimental.set_memory_growth(gpu0, True) 
    tf.config.set_visible_devices([gpu0], "GPU")

# Seed
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Add last layer
def add_last_layer(base_model, NUM_CLASSES):
    x = base_model.layers[-2].output
    pred = Dense(NUM_CLASSES, activation=activations.softmax)(x)  # set activation directly
    from tensorflow.keras.models import Model
    return Model(inputs=base_model.input, outputs=pred)

# Freeze layers
def freeze(model_, freeze_num):
    for layer in model_.layers[:freeze_num]:
        layer.trainable = False
    return model_

# Learning rate decay
def lr_decay(learning_rate, epoch):
    if epoch <= 50:
        return learning_rate
    elif epoch <= 100:
        return learning_rate * 0.5
    elif epoch <= 300:
        return learning_rate * 0.1
    else:
        return learning_rate * 0.05

# Initialize model
def init_way(init_weight, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, NUM_CLASSES, show_summary=True):
    freeze_num = 0

    if init_weight == 'ConvNeXt':
        model = ConvNeXtLarge(weights=None, model_name='convnext_large', include_top=True,
                              input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                              classes=NUM_CLASSES, classifier_activation='softmax')
    elif init_weight == 'Vit':
        model = create_vit_model(image_size=128, num_classes=NUM_CLASSES)
        return model
    elif init_weight == 'JaunENet':
        # For JaunENet, we will use ConvNeXtLarge as default backbone
        model = ConvNeXtLarge(weights=None, model_name='convnext_large', include_top=True,
                              input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                              classes=NUM_CLASSES, classifier_activation='softmax')
    else:
        raise ValueError("Unsupported init_weight: {}".format(init_weight))

    model = add_last_layer(model, NUM_CLASSES)
    model = freeze(model, freeze_num)
    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    if show_summary:
        model.summary()
    return model

# Process features
def process_features(features, data_augmentation): 
    image_raw = features['image_raw'].numpy()
    image_tensor_list = []
    for image in image_raw:
        image_tensor = load_and_preprocess_image(image, data_augmentation=data_augmentation)
        image_tensor_list.append(image_tensor)
    images = tf.stack(image_tensor_list, axis=0)
    labels = features['label'].numpy()
    return images, labels

if __name__ == '__main__':
    # ================= CONFIG =================
    IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS = 128, 128, 3
    EPOCHS = 500
    BATCH_SIZE = 48
    SAVE_EVERY_N_EPOCH = 20
    INIT_LR = 2e-5
    NUM_CLASSES = 3
    PATIENCE = 80
    INIT_WAY = 'JaunENet'
    SAVE_MODEL_PATH = r"K:\DL_Anemia_Jaundice\models\jaunenet_full_model.h5"
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

    print("Loading dataset...")
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets(batch=BATCH_SIZE)

    print("Initializing model...")
    model = init_way(INIT_WAY, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, NUM_CLASSES)

    # Loss & metrics
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    # Training & validation steps
    @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch, training=True)
            loss = loss_object(label_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
        train_accuracy.update_state(label_batch, predictions)

    @tf.function
    def valid_step(image_batch, label_batch):
        predictions = model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)
        valid_loss.update_state(v_loss)
        valid_accuracy.update_state(label_batch, predictions)

    # ================= TRAINING =================
    valid_acc_sub = 0.0
    patience_marker = 0

    for epoch in range(EPOCHS):
        img_aug = True if epoch > 50 else False
        lr = lr_decay(INIT_LR, epoch)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        step = 0
        train_dataset.shuffle(100)

        # Training loop
        for features in train_dataset:
            step += 1
            X, labels = process_features(features, data_augmentation=img_aug)
            train_step(X, labels)
            print(f"\rEpoch: {epoch+1}/{EPOCHS}, step: {step}/{math.ceil(train_count/BATCH_SIZE)}, "
                  f"loss: {train_loss.result().numpy():.5f}, acc: {train_accuracy.result().numpy():.5f}", end="", flush=True)

        # Validation loop
        for features in valid_dataset:
            X_val, y_val = process_features(features, data_augmentation=False)
            valid_step(X_val, y_val)

        print(f"\nEpoch: {epoch+1}/{EPOCHS}, valid_loss: {valid_loss.result().numpy():.5f}, "
              f"valid_acc: {valid_accuracy.result().numpy():.5f}")

        # Save best model (full model)
        if valid_accuracy.result().numpy() >= valid_acc_sub:
            valid_acc_sub = valid_accuracy.result().numpy()
            model.save(SAVE_MODEL_PATH)
            patience_marker = 0
            print(f"Saved best model to: {SAVE_MODEL_PATH}")
        else:
            patience_marker += 1
            print(f"Patience step: {patience_marker}")

        if patience_marker > PATIENCE:
            print("Early stopping triggered.")
            break

        # Reset metrics
        train_loss.reset_state()
        train_accuracy.reset_state()
        valid_loss.reset_state()
        valid_accuracy.reset_state()

    # Save final model
    model.save(SAVE_MODEL_PATH)
    print(f"Final model saved at: {SAVE_MODEL_PATH}")
