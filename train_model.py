# train_model.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import config
from data_prep import make_datasets
from utils import vlog

def build_model(input_shape=(224,224,3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model, base_model

def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    train_ds, val_ds = make_datasets(img_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE)
    model, base_model = build_model(input_shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3))

    checkpoint = ModelCheckpoint(config.MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

    vlog("Starting initial training (frozen base)...")
    model.fit(train_ds, validation_data=val_ds, epochs=config.INITIAL_EPOCHS, callbacks=[checkpoint, early])

    # Fine-tune: unfreeze some layers
    vlog("Starting fine-tuning...")
    base_model.trainable = True

    # Freeze first X% of layers
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(optimizer=Adam(config.FINE_TUNE_LR), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=config.FINE_TUNE_EPOCHS, callbacks=[checkpoint, early])

    vlog(f"Final model saved to {config.MODEL_PATH}")
    return model

if __name__ == "__main__":
    train()
