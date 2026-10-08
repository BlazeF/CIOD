"""Run the trained multitask TFLite models on one RGB image."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if _SCRIPT.parent.name == "scripts" else _SCRIPT.parents[1]
TRAINED = ROOT / "models" / "trained"
MOBILE = "mobilenet_v3_large"
VIT = "vit_tiny_patch16_224_augreg_in21k_ft_in1k"

def preprocess(path, model):
    image = Image.open(path).convert("RGB")
    if model == MOBILE:
        scale = 256 / min(image.size)
        size = (int(round(image.width * scale)), int(round(image.height * scale)))
        image = image.resize(size, Image.Resampling.BILINEAR)
        left = (image.width - 224) // 2; top = (image.height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - np.array([.485,.456,.406],np.float32)) / np.array([.229,.224,.225],np.float32)
    else:
        image = image.resize((248,248), Image.Resampling.BICUBIC)
        left = (248 - 224) // 2; top = left
        image = image.crop((left,top,left+224,top+224))
        x = np.asarray(image,dtype=np.float32)/127.5-1.0
    return x[None,...]

def topk(logits, labels, k):
    z = logits.astype(np.float64).reshape(-1)
    z = np.exp(z - np.max(z)); z /= z.sum()
    indices = np.argsort(z)[-k:][::-1]
    return [{"label": labels[int(i)], "probability": float(z[i])} for i in indices]

def predict(image_path, model):
    label_data = json.loads((TRAINED / "labels.json").read_text(encoding="utf-8"))
    name = MOBILE if model == "mobilenet" else VIT
    model_path = TRAINED / f"{name}.tflite"
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    tensor = preprocess(image_path,name)
    expected = tuple(int(v) for v in inp["shape"])
    if tensor.shape != expected and tensor.transpose(0,1,3,2).shape == expected:
        tensor = tensor.transpose(0,1,3,2).copy()
    if tensor.shape != expected:
        raise ValueError(f"Preprocessed input {tensor.shape} does not match TFLite input {expected}")
    interpreter.set_tensor(inp["index"], tensor)
    interpreter.invoke()
    outputs = {}
    for detail in interpreter.get_output_details():
        shape = int(detail["shape"][-1])
        outputs[shape] = interpreter.get_tensor(detail["index"])[0]
    return {
        "model": name,
        "image": str(Path(image_path).resolve()),
        "semantic_top2": topk(outputs[len(label_data["semantic_classes"])],label_data["semantic_classes"],2),
        "fine_top3": topk(outputs[len(label_data["fine_classes"])],label_data["fine_classes"],3),
        "imagenet_top2": topk(outputs[len(label_data["imagenet_classes"])],label_data["imagenet_classes"],2),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model",choices=["mobilenet","vit","both"],default="both")
    args=ap.parse_args()
    models=["mobilenet","vit"] if args.model=="both" else [args.model]
    print(json.dumps([predict(args.image,m) for m in models],indent=2))

if __name__=="__main__": main()
