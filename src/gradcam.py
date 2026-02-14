import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
from PIL import Image


def find_last_conv_layer(model):
    """
    Automatically find last Conv2D layer
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def generate_gradcam(
    model,
    img_array,
    last_conv_layer_name=None,
    class_index=None
):
    """
    Generate Grad-CAM heatmap
    """

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image
    """

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if original_img.max() <= 1:
        original_img = (original_img * 255).astype(np.uint8)

    superimposed = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return superimposed


def image_to_base64(img):
    """
    Convert image to base64 for API response
    """

    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode()