# Model scripts

These scripts operate on the dataset in the project root. Run them from the project root (`E:\semantic_datasets`) or any working directory; they locate the root from their own path. The initial pretrained weights and ImageNet label map needed for retraining are included here: `mobilenet_v3_large-8738ca79.pth`, `vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors`, and `imagenet_classes.txt`.

## Train / retrain both models

The training script scans the dataset folders, creates reproducible train/validation/test splits, trains the two dataset-specific heads, saves PyTorch checkpoints and metrics, and exports both FP32 TFLite models.

```powershell
python models\scripts\train_models.py --epochs 24 --batch-size 32
```

Use `--skip-export` to train and save checkpoints without converting TFLite, or `--keep-cache` to retain the temporary feature caches. A CUDA-enabled PyTorch installation is recommended. The original pretrained weights must remain in `models`.

## Export ViT TFLite from its saved checkpoint

```powershell
python models\scripts\export_vit_tflite.py
```

This reads the saved ViT checkpoint from `models\trained` and writes its FP32 TFLite model there.

## Run image inference

```powershell
python models\scripts\predict_tflite.py path\to\photo.jpg --model both
```

Use `--model mobilenet` or `--model vit` for a single model. Inference uses the TFLite files and labels under `models\trained`.

Install inference packages from `requirements-inference.txt`; use `requirements-training.txt` for the training/export dependencies. The training TFLite exporter also uses `onnx2tf` for MobileNet.
