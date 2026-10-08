"""Direct TensorFlow/TFLite export of the trained timm ViT-Tiny checkpoint."""
from pathlib import Path
import numpy as np
import torch
import tensorflow as tf

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if _SCRIPT.parent.name == "scripts" else _SCRIPT.parents[1]
OUT = ROOT / "models" / "trained"
NAME = "vit_tiny_patch16_224_augreg_in21k_ft_in1k"
CHECKPOINT = torch.load(OUT / f"{NAME}.pth", map_location="cpu", weights_only=True)["state_dict"]

def t(name):
    return tf.constant(CHECKPOINT[name].detach().cpu().numpy(), dtype=tf.float32)

def dense(x, prefix):
    return tf.linalg.matmul(x, t(prefix + ".weight"), transpose_b=True) + t(prefix + ".bias")

def ln(x, prefix, eps=1e-6):
    mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
    return (x - mean) * tf.math.rsqrt(var + eps) * t(prefix + ".weight") + t(prefix + ".bias")

class ViT(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="image")])
    def __call__(self, image):
        # Inputs are RGB floats already normalized to [-1, 1], matching timm.
        p = "backbone.model"
        conv_w = tf.transpose(t(p + ".patch_embed.proj.weight"), [2, 3, 1, 0])
        x = tf.nn.conv2d(image, conv_w, strides=[1, 16, 16, 1], padding="VALID") + t(p + ".patch_embed.proj.bias")
        x = tf.reshape(x, [1, 196, 192])
        x = tf.concat([tf.broadcast_to(t(p + ".cls_token"), [1, 1, 192]), x], axis=1)
        x = x + t(p + ".pos_embed")
        for i in range(12):
            b = f"{p}.blocks.{i}"
            y = ln(x, b + ".norm1")
            qkv = dense(y, b + ".attn.qkv")
            qkv = tf.reshape(qkv, [1, 197, 3, 3, 64])
            q = tf.transpose(qkv[:, :, 0], [0, 2, 1, 3])
            k = tf.transpose(qkv[:, :, 1], [0, 2, 1, 3])
            v = tf.transpose(qkv[:, :, 2], [0, 2, 1, 3])
            att = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * (64.0 ** -0.5), axis=-1)
            z = tf.transpose(tf.matmul(att, v), [0, 2, 1, 3])
            z = tf.reshape(z, [1, 197, 192])
            x = x + dense(z, b + ".attn.proj")
            y = ln(x, b + ".norm2")
            y = dense(y, b + ".mlp.fc1")
            # Tanh GELU approximation avoids the unsupported Erf TFLite op.
            y = 0.5 * y * (1.0 + tf.tanh(0.7978845608 * (y + 0.044715 * tf.pow(y, 3))))
            x = x + dense(y, b + ".mlp.fc2")
        feat = ln(x, p + ".norm")[:, 0]
        imagenet = dense(feat, p + ".head")
        sem = dense(tf.nn.relu(dense(feat, "semantic.net.0")), "semantic.net.3")
        fine = dense(tf.nn.relu(dense(feat, "fine.net.0")), "fine.net.3")
        return {"semantic_logits": sem, "fine_logits": fine, "imagenet_logits": imagenet}

def main():
    module = ViT()
    concrete = module.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    model = converter.convert()
    dest = OUT / f"{NAME}.tflite"
    dest.write_bytes(model)
    print(f"Saved FP32 TFLite: {dest} ({len(model)/1024/1024:.1f} MiB)")
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    print("Input:", interpreter.get_input_details()[0]["shape"], interpreter.get_input_details()[0]["dtype"])
    print("Outputs:", [(x["name"], x["shape"].tolist(), x["dtype"].__name__) for x in interpreter.get_output_details()])
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], np.zeros((1,224,224,3),np.float32))
    interpreter.invoke()
    print("Output inference smoke check passed")

if __name__ == "__main__": main()
