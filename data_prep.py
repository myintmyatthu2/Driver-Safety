# data_prep.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import config

def make_datasets(data_dir=config.DATA_DIR, img_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE, seed=config.SEED):
    """
    Creates train and validation tf.data.Dataset objects using
    image_dataset_from_directory. Expects subfolders under data_dir
    such as 'open' and 'closed' (class names inferred).
    Returns (train_ds, val_ds).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} not found. Create dataset folder with subfolders 'open' and 'closed'.")

    train_ds = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

if __name__ == "__main__":
    train_ds, val_ds = make_datasets()
    print("Training batches:", len(list(train_ds)))
    print("Validation batches:", len(list(val_ds)))
