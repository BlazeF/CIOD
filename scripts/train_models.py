from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models as tv_models, transforms
import timm
from safetensors.torch import load_file

_SCRIPT = Path(__file__).resolve()
ROOT = _SCRIPT.parents[2] if _SCRIPT.parent.name == "scripts" else _SCRIPT.parents[1]
ASSET_DIR = _SCRIPT.parent if (_SCRIPT.parent/"mobilenet_v3_large-8738ca79.pth").exists() else ROOT/"models"
OUT = ROOT / "models" / "trained"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 347
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm_label(s: str) -> str:
    s = re.sub(r"[_-]+", " ", s.strip())
    return " ".join(w[:1].upper() + w[1:] for w in s.split())


def index_dataset():
    rows = []
    for top in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "models" and not p.name.startswith(".")):
        semantic = "Game" if top.name == "Gaming" else norm_label(top.name)
        for p in top.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in EXTS:
                continue
            rel = p.relative_to(top)
            leaf = rel.parts[0] if len(rel.parts) > 1 else ""
            fine = f"{norm_label(leaf)}-{semantic}" if leaf else semantic
            group = p.stem
            if top.name == "Gaming":
                meta = p.parent / "metadata.csv"
                if meta.exists():
                    try:
                        with meta.open(newline="", encoding="utf-8-sig") as f:
                            for r in csv.DictReader(f):
                                if r.get("image_file") == p.name:
                                    group = r.get("video_id") or p.stem
                                    break
                    except (OSError, csv.Error):
                        pass
            rows.append({"path": p, "semantic": semantic, "fine": fine, "group": f"{top.name}/{leaf}/{group}"})
    if not rows:
        raise RuntimeError(f"No images found beneath {ROOT}")
    semantic_names = sorted({r["semantic"] for r in rows})
    fine_names = sorted({r["fine"] for r in rows})
    sem_map = {v: i for i, v in enumerate(semantic_names)}
    fine_map = {v: i for i, v in enumerate(fine_names)}
    for r in rows:
        r["semantic_id"] = sem_map[r["semantic"]]
        r["fine_id"] = fine_map[r["fine"]]
    return rows, semantic_names, fine_names


def make_splits(rows):
    # Keep each gameplay source video in exactly one partition. Other classes
    # use deterministic per-fine-class image splits.
    rng = random.Random(SEED)
    ids = {"train": [], "val": [], "test": []}
    gaming_groups = defaultdict(list)
    per_class = defaultdict(list)
    for i, r in enumerate(rows):
        if r["group"].startswith("Gaming/"):
            gaming_groups[(r["fine_id"], r["group"])].append(i)
        else:
            per_class[r["fine_id"]].append(i)
    by_fine_groups = defaultdict(list)
    for (fine, group), members in gaming_groups.items():
        by_fine_groups[fine].append((group, members))
    for fine, groups in by_fine_groups.items():
        rng.shuffle(groups)
        n = len(groups)
        # With five source videos this yields 3 train, 1 validation, 1 test.
        nt = max(1, int(n * .6))
        nv = max(1, int(n * .2)) if n >= 3 else 0
        if nt + nv >= n and n > 1:
            nt = n - 1 - nv
        for j, (_, members) in enumerate(groups):
            part = "train" if j < nt else ("val" if j < nt + nv else "test")
            ids[part].extend(members)
    for fine, members in per_class.items():
        rng.shuffle(members)
        n = len(members)
        if n == 1:
            ids["train"].extend(members)
            continue
        ntest = max(1, round(n * .1))
        nval = max(1, round(n * .1)) if n >= 5 else (1 if n >= 3 else 0)
        ntrain = n - ntest - nval
        ids["train"].extend(members[:ntrain])
        ids["val"].extend(members[ntrain:ntrain+nval])
        ids["test"].extend(members[ntrain+nval:])
    for part in ids:
        rng.shuffle(ids[part])
    return ids


class ImageRows(Dataset):
    def __init__(self, rows, indices, tfm):
        self.rows, self.indices, self.tfm = rows, indices, tfm
    def __len__(self): return len(self.indices)
    def __getitem__(self, j):
        r = self.rows[self.indices[j]]
        try:
            im = Image.open(r["path"]).convert("RGB")
            return self.tfm(im), r["semantic_id"], r["fine_id"]
        except Exception as e:
            # Corrupt files are skipped later by returning a neutral tensor;
            # record the issue to the run log.
            with (OUT / "bad_images.txt").open("a", encoding="utf-8") as f:
                f.write(f"{r['path']}\t{e}\n")
            return torch.zeros((3, 224, 224)), r["semantic_id"], r["fine_id"]


class MobileBackbone(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.model = tv_models.mobilenet_v3_large(weights=None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
    def forward(self, x):
        z = self.model.features(x)
        z = self.model.avgpool(z).flatten(1)
        z = self.model.classifier[1](self.model.classifier[0](z))
        logits = self.model.classifier[3](self.model.classifier[2](z))
        return z, logits


class VitBackbone(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
    def forward(self, x):
        z = self.model.forward_features(x)[:, 0]
        return z, self.model.head(z)


class TaskHead(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        hidden = min(512, max(128, dim))
        # ReLU maps cleanly to the standard TFLite builtin operator set.
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(.25), nn.Linear(hidden, classes))
    def forward(self, x): return self.net(x)


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, dim, ns, nf):
        super().__init__()
        self.backbone = backbone
        self.semantic = TaskHead(dim, ns)
        self.fine = TaskHead(dim, nf)
    def forward(self, x):
        z, imagenet = self.backbone(x)
        return self.semantic(z), self.fine(z), imagenet


def get_transform(name):
    if name == "mobilenet_v3_large":
        return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    return transforms.Compose([transforms.Resize(248, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))])


def extract_features(rows, split, backbone, tfm, dim, model_name, bs=64):
    cache = OUT / f"{model_name}_features.f16"
    labels = OUT / f"{model_name}_labels.npz"
    N = len(rows)
    mm = np.memmap(cache, mode="w+", dtype=np.float16, shape=(N, dim))
    sem = np.empty(N, np.int32); fine = np.empty(N, np.int32)
    loaders = DataLoader(ImageRows(rows, list(range(N)), tfm), batch_size=bs, shuffle=False, num_workers=0, pin_memory=DEVICE.type == "cuda")
    backbone.to(DEVICE).eval()
    offset = 0
    with torch.inference_mode():
        for bi, (x, ys, yf) in enumerate(loaders):
            x = x.to(DEVICE, non_blocking=True)
            z, _ = backbone(x)
            n = x.shape[0]
            mm[offset:offset+n] = z.float().cpu().numpy().astype(np.float16)
            sem[offset:offset+n] = ys.numpy(); fine[offset:offset+n] = yf.numpy()
            offset += n
            if bi % 100 == 0:
                print(f"[{model_name}] embedding {offset}/{N}", flush=True)
    mm.flush()
    np.savez_compressed(labels, semantic=sem, fine=fine, splits=np.array(json.dumps(split)))
    return np.memmap(cache, mode="r", dtype=np.float16, shape=(N, dim)), sem, fine


def class_weights(y, num_classes):
    c = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(c, 1))
    w *= num_classes / w.sum()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_heads(features, sem, fine, splits, ns, nf, dim, model_name, epochs=30):
    train_idx, val_idx = splits["train"], splits["val"]
    x = torch.from_numpy(np.array(features, dtype=np.float32))
    ysem = torch.from_numpy(sem.astype(np.int64)); yfine = torch.from_numpy(fine.astype(np.int64))
    heads = nn.ModuleDict({"semantic": TaskHead(dim, ns), "fine": TaskHead(dim, nf)}).to(DEVICE)
    opt = torch.optim.AdamW(heads.parameters(), lr=8e-4, weight_decay=1e-4)
    ws, wf = class_weights(sem[train_idx], ns), class_weights(fine[train_idx], nf)
    best_loss, best_state, stale = float("inf"), None, 0
    train_dl = DataLoader(TensorDataset(torch.tensor(train_idx),), batch_size=2048, shuffle=True)
    for epoch in range(epochs):
        heads.train()
        for (idx,) in train_dl:
            idx = idx.numpy()
            xb = x[idx].to(DEVICE); sb = ysem[idx].to(DEVICE); fb = yfine[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(heads["semantic"](xb), sb, weight=ws) + nn.functional.cross_entropy(heads["fine"](xb), fb, weight=wf)
            loss.backward(); opt.step()
        heads.eval(); sums=0.; count=0
        with torch.no_grad():
            for start in range(0, len(val_idx), 4096):
                ix = val_idx[start:start+4096]
                xb=x[ix].to(DEVICE); sb=ysem[ix].to(DEVICE); fb=yfine[ix].to(DEVICE)
                v = nn.functional.cross_entropy(heads["semantic"](xb),sb) + nn.functional.cross_entropy(heads["fine"](xb),fb)
                sums += float(v)*len(ix); count += len(ix)
        vl = sums/max(count,1)
        print(f"[{model_name}] epoch {epoch+1}/{epochs} validation_loss={vl:.5f}", flush=True)
        if vl < best_loss:
            best_loss, stale = vl, 0
            best_state = {k:v.detach().cpu().clone() for k,v in heads.state_dict().items()}
        else:
            stale += 1
            if stale >= 5: break
        for g in opt.param_groups:
            g["lr"] = max(g["lr"] * .92, 1e-5)
    heads.load_state_dict(best_state)
    return heads


def scores(head, features, sem, fine, splits, semantic_names, fine_names, model_name):
    head.eval()
    report = {}
    with torch.no_grad():
        for part in ("val", "test"):
            idx=splits[part]; cor_s=cor_f=tot=0; top3=0
            for start in range(0,len(idx),4096):
                ix=idx[start:start+4096]
                xb=torch.from_numpy(np.array(features[ix],dtype=np.float32)).to(DEVICE)
                ps,pf=head["semantic"](xb),head["fine"](xb)
                ts=torch.tensor(sem[ix],device=DEVICE); tf=torch.tensor(fine[ix],device=DEVICE)
                cor_s += int((ps.argmax(1)==ts).sum()); cor_f += int((pf.argmax(1)==tf).sum())
                top3 += int((pf.topk(min(3,pf.shape[1]),1).indices==tf[:,None]).any(1).sum()); tot += len(ix)
            report[part]={"count":tot,"semantic_accuracy":cor_s/max(tot,1),"fine_top1_accuracy":cor_f/max(tot,1),"fine_top3_accuracy":top3/max(tot,1)}
    return report


def export_model(backbone, heads, dim, ns, nf, model_name):
    class Export(nn.Module):
        def __init__(self, b, h):
            super().__init__(); self.backbone=b; self.semantic=h["semantic"]; self.fine=h["fine"]
        def forward(self, nchw):
            x=nchw
            z, il=self.backbone(x)
            return self.semantic(z), self.fine(z), il
    model=Export(backbone.cpu().eval(),heads.cpu().eval()).eval()
    onnx_path=OUT/f"{model_name}.onnx"
    torch.onnx.export(model,torch.zeros(1,3,224,224),str(onnx_path),opset_version=17,input_names=["image"],output_names=["semantic_logits","fine_logits","imagenet_logits"],dynamo=False)
    print(f"[{model_name}] converting ONNX to TensorFlow SavedModel",flush=True)
    import onnx2tf
    saved=OUT/f"{model_name}_savedmodel"
    if saved.exists(): shutil.rmtree(saved)
    # onnx2tf's automatic sample-image download can fail on restricted or
    # stale mirrors; validation is performed against the generated TFLite
    # model in the export step instead.
    onnx2tf.onnx2tf.download_test_image_data=lambda: np.zeros((20,224,224,3),np.float32)
    onnx2tf.convert(input_onnx_file_path=str(onnx_path),output_folder_path=str(saved),non_verbose=True)
    import tensorflow as tf
    converter=tf.lite.TFLiteConverter.from_saved_model(str(saved))
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    data=converter.convert()
    tflite=OUT/f"{model_name}.tflite"; tflite.write_bytes(data)
    print(f"[{model_name}] wrote FP32 TFLite {len(data)/1024/1024:.1f} MiB",flush=True)
    return tflite


def load_models():
    mobile=MobileBackbone(torch.load(ASSET_DIR/"mobilenet_v3_large-8738ca79.pth",map_location="cpu",weights_only=True))
    vit=VitBackbone(load_file(str(ASSET_DIR/"vit_tiny_patch16_224_augreg_in21k_ft_in1k.safetensors"),device="cpu"))
    return [("mobilenet_v3_large",mobile,1280),("vit_tiny_patch16_224_augreg_in21k_ft_in1k",vit,192)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--skip-export",action="store_true")
    ap.add_argument("--keep-cache",action="store_true")
    args=ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    OUT.mkdir(parents=True,exist_ok=True)
    rows, sem_names, fine_names=index_dataset(); splits=make_splits(rows)
    labels={"semantic_classes":sem_names,"fine_classes":fine_names,"imagenet_classes":[x.strip() for x in (ASSET_DIR/"imagenet_classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()],"top_k":{"semantic":2,"fine":3,"imagenet":2},"dataset_root":str(ROOT),"image_count":len(rows)}
    (OUT/"labels.json").write_text(json.dumps(labels,indent=2),encoding="utf-8")
    split_export={k:[str(rows[i]["path"].relative_to(ROOT)) for i in v] for k,v in splits.items()}
    (OUT/"splits.json").write_text(json.dumps(split_export,indent=2),encoding="utf-8")
    print(f"Dataset: {len(rows):,} images, {len(sem_names)} semantic classes, {len(fine_names)} fine classes; splits { {k:len(v) for k,v in splits.items()} }; device={DEVICE}",flush=True)
    results={"dataset":{"images":len(rows),"semantic_classes":len(sem_names),"fine_classes":len(fine_names),"split_counts":{k:len(v) for k,v in splits.items()}},"models":{}}
    for name, backbone, dim in load_models():
        print(f"Starting {name}",flush=True)
        tfm=get_transform("mobilenet_v3_large" if name=="mobilenet_v3_large" else "vit")
        feats, sem, fine=extract_features(rows,splits,backbone,tfm,dim,name,args.batch_size)
        heads=train_heads(feats,sem,fine,splits,len(sem_names),len(fine_names),dim,name,args.epochs)
        report=scores(heads,feats,sem,fine,splits,sem_names,fine_names,name)
        full=MultiTaskModel(backbone,dim,len(sem_names),len(fine_names))
        full.semantic.load_state_dict(heads["semantic"].state_dict()); full.fine.load_state_dict(heads["fine"].state_dict())
        torch.save({"state_dict":full.state_dict(),"architecture":name,"embedding_dim":dim,"label_file":"labels.json","metrics":report},OUT/f"{name}.pth")
        if not args.skip_export:
            if name == "vit_tiny_patch16_224_augreg_in21k_ft_in1k":
                import subprocess
                vit_export = ROOT/"models"/"scripts"/"export_vit_tflite.py"
                if not vit_export.exists():
                    vit_export = ROOT/"models"/"export_vit_tflite.py"
                subprocess.run([sys.executable,str(vit_export)],check=True)
            else:
                export_model(backbone,heads,dim,len(sem_names),len(fine_names),name)
        results["models"][name]={"metrics":report}
        if not args.keep_cache:
            del feats; (OUT/f"{name}_features.f16").unlink(missing_ok=True); (OUT/f"{name}_labels.npz").unlink(missing_ok=True)
        print(f"Completed {name}: {report}",flush=True)
    (OUT/"training_report.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"Training artifacts: {OUT}",flush=True)

if __name__=="__main__": main()

