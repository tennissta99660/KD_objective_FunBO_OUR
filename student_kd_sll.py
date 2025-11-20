#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
student_kd.py

Student-teacher KD training script.
- Student: ResNet-18
- Teachers: Ensemble of 3 ResNet-50 (attempt to load SSL checkpoints; fallback to ImageNet pretrained)
- Writes result.json with {"val_acc": <float>}
- Accepts --config JSON same as earlier format.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="path to json config")
parser.add_argument("--short", action="store_true", help="short (cheap) mode")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# --- default hyperparams (overridden by config) ---
hp = {
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 1e-2,
    "weight_decay": 1e-4,
    "alpha": 0.7,
    "temperature": 0.5,
    "teacher_ckpt_1": None,
    "teacher_ckpt_2": None,
    "teacher_ckpt_3": None,
    "limit_train_frac": 1.0,
    "num_classes": 10,
    "img_size": 224
}
if args.config and os.path.exists(args.config):
    try:
        hp.update(json.load(open(args.config)))
    except Exception:
        pass

if args.short:
    hp["epochs"] = max(1, int(hp.get("epochs", 3)))
    hp["limit_train_frac"] = min(hp.get("limit_train_frac", 1.0), 0.15)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- models ---
class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        res = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(res.children())[:-1])
        self.head = nn.Linear(res.fc.in_features, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head(x)

def try_load_ssl_resnet50(checkpoint_path=None, device="cpu"):
    """
    Flexible loader: try common SSL sources, then checkpoint path, then fall back to ImageNet pretrained.
    Returns a ResNet50 model on device in eval() mode.
    """
    # 1) If user provided a checkpoint path, try to load
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            sd = torch.load(checkpoint_path, map_location=device)
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, hp["num_classes"])
            # attempt a few common shapes of saved checkpoint
            if isinstance(sd, dict) and "state_dict" in sd:
                state = sd["state_dict"]
                # strip common prefixes
                try:
                    model.load_state_dict(state, strict=False)
                except Exception:
                    new_state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
                    model.load_state_dict(new_state, strict=False)
            elif isinstance(sd, dict):
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    pass
            else:
                # sd might be raw state_dict
                try:
                    model.load_state_dict(sd, strict=False)
                except Exception:
                    pass
            model.to(device).eval()
            print(f"Loaded SSL checkpoint from: {checkpoint_path}")
            return model
        except Exception as e:
            print(f"Warning: failed to load provided checkpoint {checkpoint_path}: {e}")

    # 2) Try common hub sources (MoCo) — best-effort, may require internet & hub availability
    try:
        # example: facebookresearch/moco: moco_v2_800ep_pretrained
        moco = torch.hub.load("facebookresearch/moco:main", "moco_v2_800ep_pretrained")
        # moco may return a wrapper; attempt to extract backbone
        if hasattr(moco, "state_dict"):
            # attempt to adapt
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, hp["num_classes"])
            state = moco.state_dict()
            try:
                model.load_state_dict(state, strict=False)
                model.to(device).eval()
                print("Loaded MoCo V2 pretrained ResNet50 via torch.hub")
                return model
            except Exception:
                pass
    except Exception:
        pass

    # 3) Fallback: ImageNet pretrained resnet50 (supervised)
    print("Falling back to ImageNet-pretrained ResNet50 (not SSL). If you want SSL teachers, provide checkpoints.)")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, hp["num_classes"])
    model.to(device).eval()
    return model

class LightningKD(pl.LightningModule):
    def __init__(self, hp, teachers):
        super().__init__()
        self.save_hyperparameters(hp)
        self.student = StudentNet(num_classes=hp["num_classes"])
        self.teachers = teachers  # list of teacher nn.Modules in eval mode
        # note: teachers on device already
        # Access hparams via self.hparams

    def training_step(self, batch, batch_idx):
        x,y = batch
        s_logits = self.student(x)
        # ensemble teacher logits (no grad)
        with torch.no_grad():
            teacher_logits_list = [t(x) for t in self.teachers]
            # average logits
            t_logits = torch.mean(torch.stack(teacher_logits_list, dim=0), dim=0)

        s_logits_f = s_logits.float()
        t_logits_f = t_logits.float()

        T = float(self.hparams.temperature)
        kd_loss = F.kl_div(F.log_softmax(s_logits_f / T, dim=1),
                           F.softmax(t_logits_f / T, dim=1),
                           reduction="batchmean") * (T * T)
        ce = F.cross_entropy(s_logits_f, y)
        loss = self.hparams.alpha * kd_loss + (1.0 - self.hparams.alpha) * ce
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self.student(x)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return {"val_acc": acc}

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.student.parameters(),
                              lr=self.hparams.learning_rate,
                              weight_decay=self.hparams.weight_decay,
                              momentum=0.9)
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[int(max(1, self.trainer.max_epochs * 0.6))],
            gamma=0.1
        )
        return [opt], [sched]

# --- data ---
img_size = hp.get("img_size", 224)
transform_train = transforms = transforms = __import__("torchvision.transforms", fromlist=["Compose"]).Compose([
    __import__("torchvision.transforms", fromlist=["Resize"]).Resize((img_size, img_size)),
    __import__("torchvision.transforms", fromlist=["RandomHorizontalFlip"]).RandomHorizontalFlip(),
    __import__("torchvision.transforms", fromlist=["ToTensor"]).ToTensor(),
    __import__("torchvision.transforms", fromlist=["Normalize"]).Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
transform_test = transforms = __import__("torchvision.transforms", fromlist=["Compose"]).Compose([
    __import__("torchvision.transforms", fromlist=["Resize"]).Resize((img_size, img_size)),
    __import__("torchvision.transforms", fromlist=["ToTensor"]).ToTensor(),
    __import__("torchvision.transforms", fromlist=["Normalize"]).Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# --- main run ---
if __name__ == "__main__":
    train_full = datasets.CIFAR10(root=os.getcwd(), train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root=os.getcwd(), train=False, download=True, transform=transform_test)

    if hp.get("limit_train_frac", 1.0) < 1.0:
        n = int(len(train_full) * hp["limit_train_frac"])
        train_subset = torch.utils.data.Subset(train_full, list(range(n)))
    else:
        train_subset = train_full

    train_loader = DataLoader(train_subset, batch_size=hp["batch_size"], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=hp["batch_size"], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Create/load teachers (3x ResNet-50 SSL attempts)
    teachers = []
    for i in range(1, 4):
        ck_key = f"teacher_ckpt_{i}"
        ckpt_path = hp.get(ck_key, None)
        teacher = try_load_ssl_resnet50(checkpoint_path=ckpt_path, device=device)
        teacher.to(device).eval()
        teachers.append(teacher)

    model = LightningKD(hp, teachers)

    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, dirpath=os.getcwd(), filename="studentkd_best")
    trainer = pl.Trainer(
        max_epochs=hp["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
        limit_train_batches=1.0
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
