"""
train_fusion_dr.py

Complete training script for a fusion model (DenseNet121, ResNet50, VGG16)
with a custom attention module and LocallyConnected1D fusion for APTOS-2019 DR classification.

Usage example:
    python train_fusion_dr.py --csv data/train.csv --img_dir data/train_images --epochs 40 --batch_size 8

Dependencies:
    pip install tensorflow==2.11.0 opencv-python pandas scikit-learn

"""
import os
import argparse
import math
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def preprocess_image(img):
    """Preprocessing function applied to each image array (H,W,3, RGB):
    - Gaussian blur
    - Convert to grayscale
    - Circular crop (mask)
    - Resize to 224x224

    This function returns an RGB uint8 image array sized (224,224,3).
    It is intended to be passed as the `preprocessing_function` to
    `ImageDataGenerator` and will be scaled afterwards by rescale=1./255.
    """
    # Ensure uint8
    img = img.astype(np.uint8)
    # ImageDataGenerator provides RGB arrays; convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Gaussian blur
    img_blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)

    # Convert to grayscale
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # Circular crop mask
    h, w = gray.shape
    center = (w // 2, h // 2)
    radius = int(min(center) * 0.95)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    gray_masked = (gray * mask).astype(np.uint8)

    # Convert back to 3-channel BGR (so base models get 3 channels)
    img_gray3 = cv2.cvtColor(gray_masked, cv2.COLOR_GRAY2BGR)

    # Resize to 224x224
    img_resized = cv2.resize(img_gray3, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert back to RGB and cast to float32 so rescale won't fail in-place
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32)


def get_generators(csv_path, img_dir, batch_size=8, val_split=0.2, seed=42):
    df = pd.read_csv(csv_path)

    # Detect common column names from APTOS / Kaggle style CSV
    if 'id_code' in df.columns:
        id_col = 'id_code'
    elif 'image' in df.columns:
        id_col = 'image'
    else:
        id_col = df.columns[0]

    if 'diagnosis' in df.columns:
        label_col = 'diagnosis'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        label_col = df.columns[-1]

    # Build filepath column (assume PNG; adjust if needed)
    def _try_paths(base, img_name):
        candidates = [f"{img_name}.png", f"{img_name}.jpg", f"{img_name}.jpeg"]
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.exists(p):
                return p
        # fallback default path (png)
        return os.path.join(base, f"{img_name}.png")

    df['filepath'] = df[id_col].astype(str).apply(lambda x: _try_paths(img_dir, x))
    df['label'] = df[label_col].astype(str)

    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['label'], random_state=seed)

    # ImageDataGenerators with augmentation; preprocessing_function is applied before rescale
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rescale=1.0 / 255.0
    )

    target_size = (224, 224)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filepath',
        y_col='label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen


@tf.keras.utils.register_keras_serializable()
def scale_fn(inputs):
    """Rescale pooled output proportional to non-zero pixel counts in feature maps."""
    fmap, pooled = inputs
    mask = tf.reduce_any(tf.not_equal(fmap, 0.0), axis=-1)  # (batch, H, W)
    pixel_count = tf.reduce_sum(tf.cast(mask, tf.float32), axis=[1, 2])  # (batch,)
    total = tf.cast(tf.shape(mask)[1] * tf.shape(mask)[2], tf.float32)
    scale = pixel_count / (total + 1e-6)
    scale = tf.expand_dims(scale, -1)  # (batch, 1)
    return pooled * scale


def build_base(name, input_shape=(224, 224, 3)):
    if name == 'densenet':
        base = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    elif name == 'resnet':
        base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    elif name == 'vgg':
        base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError('Unknown base model name')

    base.trainable = True
    return base


def attention_module(feature_map, filters=256):
    """Complex parallel attention matching Figure 4 of the paper.

    Structure:
    - BatchNorm on input
    - Multiple parallel 1x1 conv branches (channel down-projection)
    - Element-wise multiplication of branches
    - GAP on multiplied map
    - Rescale by pixel-count proportion via Lambda
    - Final Dense layer to produce feature vector
    """
    x_bn = layers.BatchNormalization()(feature_map)

    # Parallel 1x1 conv branches (downsample channels)
    b1 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x_bn)
    b2 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x_bn)
    b3 = layers.Conv2D(filters, (1, 1), activation='relu', padding='same')(x_bn)

    # Optionally apply small separable convs to add diversity
    b1 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(b1)
    b2 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(b2)
    b3 = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(b3)

    # Branch multiplication (element-wise)
    merged = layers.Multiply()([b1, b2, b3])

    # Global Average Pooling to highlight ROI
    gap = layers.GlobalAveragePooling2D()(merged)

    # Rescale pooled output using the global scale_fn (proportional to non-zero pixel counts)
    scaled = layers.Lambda(scale_fn)([feature_map, gap])

    # Final dense projection
    dense_out = layers.Dense(filters, activation='relu')(scaled)
    return dense_out


def build_fusion_model(input_shape=(224, 224, 3), num_classes=5):
    # Inputs
    inp = layers.Input(shape=input_shape)

    # Base models
    densenet = build_base('densenet', input_shape)
    resnet = build_base('resnet', input_shape)
    vgg = build_base('vgg', input_shape)

    # Get feature maps
    f1 = densenet(inp)
    f2 = resnet(inp)
    f3 = vgg(inp)

    # Attention-modified GAP outputs
    a1 = attention_module(f1, filters=256)
    a2 = attention_module(f2, filters=256)
    a3 = attention_module(f3, filters=256)

    # Apply independent Dense layers to each attention output (simulate unshared weights)
    d1 = layers.Dense(128, activation='relu')(a1)
    d2 = layers.Dense(128, activation='relu')(a2)
    d3 = layers.Dense(128, activation='relu')(a3)

    # Concatenate the three independent projections
    concat = layers.Concatenate()([d1, d2, d3])

    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out, name='fusion_dr_model')
    return model


class CategoricalFocalLoss(tf.keras.losses.Loss):
    """Focal loss for handling class imbalance in multi-class classification."""
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss_tensor = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss_tensor, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class WarmUpCallback(callbacks.Callback):
    """Simple warmup: use `warmup_lr` for the first `warmup_epochs`, then `base_lr` afterwards."""
    def __init__(self, warmup_epochs=2, warmup_lr=1e-3, base_lr=1e-4):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr
        else:
            lr = self.base_lr
        self.model.optimizer.learning_rate.assign(lr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/train.csv')
    parser.add_argument('--img_dir', type=str, default='data/train_images')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='/content/drive/MyDrive/best_fusion_model.keras')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to .keras checkpoint to resume from')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Starting epoch for training (used when resuming)')
    return parser.parse_args()


def main():
    args = parse_args()

    train_gen, val_gen = get_generators(args.csv, args.img_dir, batch_size=args.batch_size)

    # Load model from checkpoint if provided, otherwise build from scratch
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print(f"Loading model from checkpoint: {args.resume_from}")
        try:
            model = tf.keras.models.load_model(
                args.resume_from,
                custom_objects={
                    'CategoricalFocalLoss': CategoricalFocalLoss,
                    'scale_fn': scale_fn
                },
                safe_mode=False
            )
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e}). Building new model from scratch.")
            model = build_fusion_model(input_shape=(224, 224, 3), num_classes=5)
    else:
        if args.resume_from is not None:
            print(f"Checkpoint file not found: {args.resume_from}. Building new model from scratch.")
        else:
            print("Building new model from scratch")
        model = build_fusion_model(input_shape=(224, 224, 3), num_classes=5)

    # Optimizer and loss
    base_lr = 1e-4
    warmup_lr = 1e-3
    opt = optimizers.Adam(learning_rate=base_lr)
    loss_fn = CategoricalFocalLoss(gamma=2.0, alpha=0.25)

    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    # Callbacks
    warmup_cb = WarmUpCallback(warmup_epochs=2, warmup_lr=warmup_lr, base_lr=base_lr)
    reduce_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
    ckpt_cb = callbacks.ModelCheckpoint(args.output, monitor='val_loss', save_best_only=True, verbose=1)

    steps_per_epoch = math.ceil(train_gen.n / train_gen.batch_size)
    validation_steps = math.ceil(val_gen.n / val_gen.batch_size)

    model.summary()

    model.fit(
        train_gen,
        epochs=args.epochs,
        initial_epoch=args.initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[warmup_cb, reduce_cb, ckpt_cb]
    )


if __name__ == '__main__':
    main()
