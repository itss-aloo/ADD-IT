import cv2
import numpy as np

from layers.rendering.rendering_layer import (
    render_logo_on_image,
)

if __name__ == "__main__":
    dst_points = np.array([
        [322, 199],
        [429, 191],
        [439, 295],
        [345, 297],
    ], dtype=np.float32)

    result = render_logo_on_image(
        "data/images/07_image.jpg",
        "data/logos/10_logo.png",
        dst_points,
    )

    print("result:", result.shape)

    cv2.imwrite("render_logo_on_image_test.png", result)
