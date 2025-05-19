import base64
import os
import subprocess
from io import BytesIO

import numpy as np
from PIL import Image

working_dir = os.path.dirname(os.path.abspath(__file__))


def render_threejs_scene_to_numpy(width, height, data_str, print_err=False):
    result = subprocess.run(
        ["/root/.nvm/versions/node/v22.14.0/bin/node", "render.js", str(width), str(height), data_str],
        cwd=working_dir,
        capture_output=True,
        text=True,
    )

    if print_err:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("Render error: " + result.stderr)

    base64_image = result.stdout.strip()
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return np.array(image)[:, :, ::-1]


if __name__ == "__main__":
    data_str = "red"
    image = render_threejs_scene_to_numpy(100, 100, data_str)