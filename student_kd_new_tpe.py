#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
student_kd.py

Lightning training script for Knowledge Distillation:
- Teacher: pretrained ResNet-50 (SSL weights if provided)
- Student: ResNet-18 (standard final FC for num_classes)
- Exposes config via JSON file (same as prior code)
- Short mode for cheap runs (useful in FunBO inner loops)
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="path to json config")
parser.add_argument("--short", action="store_true", help="short (cheap) mode")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

pl.seed_everything(args.seed)

# default hyperparams (overridden by JSON config)
hp = {
    "batch_size": 64,
    "epochs": 3,
    "learning_rate": 1e-2,
    "weight_decay": 1e-4,
    "alpha": 0.7,
    "temperature": 2.0,
    "teacher_ckpt": None,
    "num_classes": 10,
    "limit_train_frac": 1.0,
    "num_workers": 4,
}

if args.config and os.path.exists(args.config):
    try:
        with open(args.config, "r") as f:
            hp.update(json.load(f))
    except Exception:
        pass

if args.short:
    hp["epochs"] = max(1, int(hp.get("epochs", 3)))
    hp["limit_train_frac"] = min(hp.get("limit_train_frac", 1.0), 0.12)
    # reduce batch size in short tests if needed
    hp["batch_size"] = min(hp["batch_size"], 32)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Student (ResNet-18)
def create_student(num_classes=10):
    r18 = torch.hub.load("pytorch/vision:v0.14.0", "resnet18", pretrained=False)
    backbone = nn.Sequential(*list(r18.children())[:-1])
    head = nn.Linear(r18.fc.in_features, num_classes)
    model = nn.Sequential(backbone, nn.Flatten(1), head)
    return model

# Load teacher (ResNet-50). If teacher_ckpt provided, try load; else use pretrained weights.
def load_teacher(num_classes=10, ckpt_path=None):
    try:
        t = torch.hub.load("pytorch/vision:v0.14.0", "resnet50", pretrained=True)
        t.fc = nn.Linear(t.fc.in_features, num_classes)
        if ckpt_path:
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                state = sd["state_dict"]
                # attempt to adapt keys
                new_state = {k.replace("module.", ""): v for k, v in state.items()}
                t.load_state_dict(new_state, strict=False)
            else:
                t.load_state_dict(sd, strict=False)
        t.eval().to(device)
        return t
    except Exception as e:
        print("Warning loading teacher:", e)
        raise

class LightningKD(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.save_hyperparameters(hp)
        self.student = create_student(num_classes=hp["num_classes"])
        self.teacher = None

    def training_step(self, batch, batch_idx):
        x, y = batch
        s_logits = self.student(x)
        with torch.no_grad():
            t_logits = self.teacher(x)
        s_logits_f = s_logits.float()
        t_logits_f = t_logits.float()

        T = float(self.hparams.temperature)
        kd_loss = F.kl_div(
            F.log_softmax(s_logits_f / T, dim=1),
            F.softmax(t_logits_f / T, dim=1),
            reduction="batchmean",
        ) * (T * T)
        ce = F.cross_entropy(s_logits_f, y)
        loss = self.hparams.alpha * kd_loss + (1.0 - self.hparams.alpha) * ce
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.student(x)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_acc": acc}

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.student.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[int(max(1, self.trainer.max_epochs * 0.6))],
            gamma=0.1,
        )
        return [opt], [sched]


# Data transforms (CIFAR-10 default pipeline resized to 224)
transform_train = transforms = transforms = transforms = transforms = transforms = transforms = None
from torchvision import transforms as T

transform_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform_test = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def main():
    # prepare data
    DATA_DIR = os.getcwd()
    train_full = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

    if hp.get("limit_train_frac", 1.0) < 1.0:
        n = int(len(train_full) * hp["limit_train_frac"])
        train_subset = torch.utils.data.Subset(train_full, list(range(n)))
    else:
        train_subset = train_full

    train_loader = DataLoader(
        train_subset,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=hp.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=hp.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=False,
    )

    teacher_model = load_teacher(num_classes=hp["num_classes"], ckpt_path=hp.get("teacher_ckpt"))
    model = LightningKD(hp)
    model.teacher = teacher_model

    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, dirpath=os.getcwd(), filename="studentkd_best")
    trainer = pl.Trainer(
        max_epochs=hp["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
        limit_train_batches=1.0,
    )

    trainer.fit(model, train_loader, val_loader)

    metrics = trainer.callback_metrics
    val_acc = None
    if "val_acc" in metrics:
        val_acc = metrics["val_acc"].item()
    else:
        res = trainer.validate(model, val_loader, verbose=False)
        if isinstance(res, list) and len(res) > 0 and "val_acc" in res[0]:
            val_acc = float(res[0]["val_acc"])

    out_dir = os.path.dirname(args.config) if (args.config and os.path.dirname(args.config)) else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"val_acc": val_acc}, f)

    print("RESULT", {"val_acc": val_acc})

if __name__ == "__main__":
    main()
