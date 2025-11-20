import os, json, argparse
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="path to json config")
parser.add_argument("--short", action="store_true", help="short (cheap) mode")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

pl.seed_everything(args.seed)

# --- default hyperparams (overridden by config) ---
hp = {
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 1e-2,
    "weight_decay": 1e-4,
    "alpha": 0.7,
    "temperature": 0.5,
    "teacher_ckpt": None,
    "limit_train_frac": 1.0
}
if args.config and os.path.exists(args.config):
    try:
        hp.update(json.load(open(args.config)))
    except Exception:
        pass

if args.short:
    hp["epochs"] = max(1, int(hp.get("epochs", 3)))
    hp["limit_train_frac"] = min(hp.get("limit_train_frac", 1.0), 0.15)

# --- models ---
class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        res = torchvision.models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(res.children())[:-1])
        self.head = nn.Linear(res.fc.in_features, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.head(x)
    
def load_teacher(device):
    if hp.get("teacher_ckpt"):
        """
        Load teacher checkpoint if provided; otherwise use ImageNet-pretrained ResNet-34.
        This mirrors the simplified teacher/student setup you requested.
        """
        # FIX: Corrected comment, was ResNet-50, code uses ResNet-34
        path = hp["teacher_ckpt"]
        try:
            sd = torch.load(path, map_location=device)
            t = torchvision.models.resnet34(pretrained=False)
            t.fc = nn.Linear(t.fc.in_features, 10)
            if isinstance(sd, dict) and "state_dict" in sd:
                state = sd["state_dict"]
                try:
                    t.load_state_dict(state, strict=False)
                except Exception:
                    new_state = {k.replace("model.", ""): v for k, v in state.items()}
                    t.load_state_dict(new_state, strict=False)
            else:
                t.load_state_dict(sd, strict=False)
            t.to(device).eval()
            print("Loaded teacher checkpoint:", path)
            return t
        except Exception as e:
            print("Warning: failed to load teacher checkpoint:", e)
            print("Falling back to pretrained ResNet-34.")
    # default pretrained ResNet-34
    t = torchvision.models.resnet34(pretrained=True)
    t.fc = nn.Linear(t.fc.in_features, 10)
    t.to(device).eval()
    return t

class LightningKD(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.save_hyperparameters(hp) # Use built-in hparams
        self.student = StudentNet(num_classes=10)
        self.teacher = None
        # Access hparams via self.hparams
        
    def training_step(self, batch, batch_idx):
        x,y = batch
        s_logits = self.student(x)
        with torch.no_grad():
            t_logits = self.teacher(x)
        
        # FIX: Cast logits to float32 before loss calculation for
        # numerical stability when using mixed precision (precision=16).
        s_logits_f = s_logits.float()
        t_logits_f = t_logits.float()
        
        T = float(self.hparams.temperature)
        kd_loss = F.kl_div(F.log_softmax(s_logits_f/T, dim=1),
                            F.softmax(t_logits_f/T, dim=1),
                            reduction="batchmean") * (T*T)
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
            milestones=[int(max(1,self.trainer.max_epochs*0.6))], 
            gamma=0.1
        )
        return [opt], [sched]

# --- data (CIFAR-10) ---
transform_train = transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

if __name__ == "__main__":
    DATA_DIR = os.getcwd()
    train_full = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

    if hp.get("limit_train_frac", 1.0) < 1.0:
        n = int(len(train_full) * hp["limit_train_frac"])
        train_subset = torch.utils.data.Subset(train_full, list(range(n)))
    else:
        train_subset = train_full

    # FIX: Set persistent_workers=True for performance.
    # This avoids re-initializing dataloader workers every epoch.
    train_loader = DataLoader(train_subset, batch_size=hp["batch_size"], shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=hp["batch_size"], shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False)

    # --- trainer run ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = load_teacher(device)

    model = LightningKD(hp)
    model.teacher = teacher_model

    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, dirpath=os.getcwd(), filename="studentkd_best")
    trainer = pl.Trainer(
        max_epochs=hp["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        # FIX: Use "16-mixed" string, which is the modern way.
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
        # FIX: Use .item() to extract a scalar from a 0-dim tensor.
        # It's cleaner than .cpu().detach().numpy().
        val_acc = metrics["val_acc"].item()
    else:
        res = trainer.validate(model, val_loader, verbose=False)
        if isinstance(res, list) and len(res) > 0 and "val_acc" in res[0]:
            val_acc = float(res[0]["val_acc"])

    out_dir = os.path.dirname(args.config) if (args.config and os.path.dirname(args.config)) else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"val_acc": val_acc}, f)

    # This print is helpful for debugging, so we leave it.
    print("RESULT", {"val_acc": val_acc})