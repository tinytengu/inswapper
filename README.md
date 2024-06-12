# inswapper

This is a fork of [inswapper](https://github.com/haofanwang/inswapper) project by [haofanwang](https://github.com/haofanwang) modified to my needs.

## Changes
- Moved [CodeFormer](https://github.com/sczhou/CodeFormer) to a Git submodule. No need to clone it separately.
- Added [Bye-lemon](https://github.com/Bye-lemon)'s for ok [py-lmdb](https://github.com/Bye-lemon/py-lmdb). No more `Building py-lmdb from source on Windows requires the "patch-ng" python module` error on newer python versions.

## Installation
### 0. Clone this repository
```bash
git clone https://github.com/tinytengu/inswapper.git
cd inswapper
```

### 1. Activate a Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install py-lmdb
```bash
pip install ./py-lmdb/
```

### 3. CodeFormer dependencies
```bash
pip install -r CodeFormer/requirements.txt
```

### 4. Inswapper dependencies
```bash
pip install -r requirements.txt
```

### 5. (Optional) Onnxruntime GPU support
```bash
pip install onnxruntime-gpu
```

## Download Checkpoints
Inswapper requires a [Face Swap Model](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx) to run. You can download it from the [release page](https://github.com/facefusion/facefusion-assets/releases/tag/models). To obtain better result, it is highly recommended to improve image quality with face restoration model. Here, we use [CodeFormer](https://github.com/sczhou/CodeFormer). You can finish all as following, required models will be downloaded automatically when you first run the inference.

```bash
wget -O ./checkpoints/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
```


## Quick Inference

```bash
from swapper import *

source_img = [Image.open("./data/man1.jpeg"),Image.open("./data/man2.jpeg")]
target_img = Image.open("./data/mans1.jpeg")

model = "./checkpoints/inswapper_128.onnx"
result_image = process(source_img, target_img, -1, -1, model)
result_image.save("result.png")
```

To improve to quality of face, we can further do face restoration as shown in the full script.

```bash
python swapper.py \
--source_img="./data/man1.jpeg;./data/man2.jpeg" \
--target_img "./data/mans1.jpeg" \
--face_restore \
--background_enhance \
--face_upsample \
--upscale=2 \
--codeformer_fidelity=0.5
```
You will obtain the exact result as above.