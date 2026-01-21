import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------
# Config (easy to override)
# -----------------------------
@dataclass
class Config:
    data_dir: str = os.path.expanduser("~/.pytorch/MNIST_data/")
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3

    # Train/val split
    val_split: float = 0.2
    seed: int = 42

    # Augmentation (train only)
    rot_deg: float = 10.0
    hflip: bool = True

    # Model
    hidden: Union[str, Sequence[int]] = (256, 128, 64)
    dropout: float = 0.5

    # Scheduler
    step_size: int = 1
    gamma: float = 0.7

    # Early stopping (uses ONLY validation)
    patience: int = 3

    # Perf
    num_workers: int = 2

    # Mixed precision
    amp: Union[bool, str] = "auto"  # True/False still work; "auto" enables only on CUDA

    # Evaluation hygiene
    eval_test: bool = True

    # Confusion-matrix interpretability / actionability
    report: Union[bool, str] = "final"  # "off"/False, "final" (default), "epoch"/True
    topk_confusions: int = 10
    print_normalized_cm: bool = True


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_hidden(hidden: Union[str, Sequence[int]]) -> List[int]:
    """
    Accepts:
      - "256,128,64"
      - "256 128 64"
      - [256,128,64]
      - (256,128,64)
    """
    if isinstance(hidden, str):
        s = hidden.replace(",", " ").strip()
        parts = [p for p in s.split() if p]
        if not parts:
            raise ValueError("hidden must not be empty")
        return [int(x) for x in parts]
    return [int(x) for x in hidden]


def build_transforms(cfg: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.5,), (0.5,))

    train_ops: List[transforms.Transform] = []
    if cfg.rot_deg and cfg.rot_deg > 0:
        train_ops.append(transforms.RandomRotation(cfg.rot_deg))
    if cfg.hflip:
        train_ops.append(transforms.RandomHorizontalFlip())

    train_tf = transforms.Compose(train_ops + [transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])  # val/test: no augmentation
    return train_tf, eval_tf


def make_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    - Train: augmented transform
    - Val: deterministic transform, disjoint indices from train
    - Test: separate MNIST test split (train=False), deterministic transform
    """
    train_tf, eval_tf = build_transforms(cfg)

    base_train = datasets.MNIST(cfg.data_dir, download=True, train=True, transform=train_tf)
    base_val = datasets.MNIST(cfg.data_dir, download=True, train=True, transform=eval_tf)

    n = len(base_train)
    val_n = int(round(cfg.val_split * n))
    train_n = n - val_n

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:train_n]
    val_idx = perm[train_n:]

    train_ds = Subset(base_train, train_idx)
    val_ds = Subset(base_val, val_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
    )

    test_loader: Optional[DataLoader] = None
    if cfg.eval_test:
        test_ds = datasets.MNIST(cfg.data_dir, download=True, train=False, transform=eval_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            persistent_workers=(cfg.num_workers > 0),
        )

    return train_loader, val_loader, test_loader


def amp_enabled(cfg_amp: Union[bool, str], device: torch.device) -> bool:
    """
    Intuitive AMP control:
      - amp="auto" (default): enable only if CUDA is available
      - amp=True: same as "auto"
      - amp=False: disable
      - amp="off"/"false"/"0": disable
      - amp="cuda": enable only if CUDA is available
    """
    if cfg_amp is True:
        mode = "auto"
    elif cfg_amp is False:
        mode = "off"
    else:
        mode = str(cfg_amp).lower().strip()

    if mode in {"off", "false", "0", "no"}:
        return False

    if mode in {"auto", "cuda"}:
        return device.type == "cuda"

    return device.type == "cuda"


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    # Row-normalized: for each true class, distribution of predicted classes
    row_sums = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    return cm.astype(np.float32) / row_sums


def print_top_confusions(cm: np.ndarray, k: int = 10) -> None:
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    row_sums = np.clip(cm.sum(axis=1), 1, None)
    flat_idx = np.argsort(cm_off.ravel())[::-1]

    printed = 0
    for idx in flat_idx:
        i, j = divmod(int(idx), cm.shape[1])
        count = int(cm_off[i, j])
        if count == 0:
            break
        rate = count / row_sums[i]
        print(f"  True {i} → Pred {j}: {count} ({rate:.1%} of true {i})")
        printed += 1
        if printed >= k:
            break


def report_metrics(labels: List[int], preds: List[int], cm: np.ndarray, cfg: Config, title: str) -> None:
    print(f"{title} Classification Report:")
    print(classification_report(labels, preds, target_names=[str(i) for i in range(10)], digits=4))

    if cfg.print_normalized_cm:
        cm_norm = normalize_confusion_matrix(cm)
        print(f"{title} Confusion Matrix (row-normalized, rounded to 3):\n{np.round(cm_norm, 3)}")

    print(f"{title} Top Confusions (off-diagonal):")
    print_top_confusions(cm, k=cfg.topk_confusions)


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(
        self,
        in_features: int = 28 * 28,
        hidden: Sequence[int] = (256, 128, 64),
        out_features: int = 10,
        dropout: float = 0.5,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        activation = activation or nn.ReLU()

        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


# -----------------------------
# Early stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Returns True if should stop. Tracks best weights by lowest val loss.
        """
        if val_loss < self.best:
            self.best = val_loss
            self.bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.bad += 1
        return self.bad >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
) -> float:
    model.train()
    total = 0.0

    use_amp = (scaler is not None) and scaler.is_enabled()
    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def run_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    """
    Backwards-compatible eval: returns (avg_loss, acc, confusion_matrix).
    """
    avg_loss, acc, cm, _, _ = run_epoch_eval_with_details(model, loader, criterion, device)
    return avg_loss, acc, cm


@torch.no_grad()
def run_epoch_eval_with_details(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, List[int], List[int]]:
    """
    Detailed eval: returns (avg_loss, acc, confusion_matrix, labels, preds).
    """
    model.eval()
    total = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total += float(loss.item())

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total / max(1, len(loader))
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, acc, cm, all_labels, all_preds


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = make_loaders(cfg)

    hidden = parse_hidden(cfg.hidden)
    model = MLP(hidden=hidden, dropout=cfg.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    use_amp = amp_enabled(cfg.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    early = EarlyStopping(patience=cfg.patience)

    report_mode = str(cfg.report).lower().strip() if not isinstance(cfg.report, bool) else ("epoch" if cfg.report else "off")
    if report_mode in {"false", "0", "no"}:
        report_mode = "off"

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch_train(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()

        val_loss, val_acc, val_cm, val_labels, val_preds = run_epoch_eval_with_details(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        # Preserve existing functionality: always print count confusion matrix each epoch
        print(f"Val Confusion Matrix (counts):\n{val_cm}")

        if report_mode == "epoch":
            report_metrics(val_labels, val_preds, val_cm, cfg, title="Val")

        if early.step(val_loss, model):
            print("Early stopping!")
            break

    # Restore best validation model BEFORE final reporting
    early.restore_best(model)

    # Final metrics (best model by val loss)
    final_val_loss, final_val_acc, final_val_cm, final_val_labels, final_val_preds = run_epoch_eval_with_details(
        model, val_loader, criterion, device
    )
    print(f"Final (Best) Val Loss: {final_val_loss:.4f} | Final (Best) Val Acc: {final_val_acc:.4f}")
    print(f"Final (Best) Val Confusion Matrix (counts):\n{final_val_cm}")

    if report_mode == "final":
        report_metrics(final_val_labels, final_val_preds, final_val_cm, cfg, title="Val (Final)")

    if cfg.eval_test and test_loader is not None:
        test_loss, test_acc, test_cm, test_labels, test_preds = run_epoch_eval_with_details(
            model, test_loader, criterion, device
        )
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Test Confusion Matrix (counts):\n{test_cm}")

        if report_mode in {"final", "epoch"}:
            report_metrics(test_labels, test_preds, test_cm, cfg, title="Test")

    print("Finished Training")



'''
if __name__ == "__main__":
    # Example quick tweaks:
    # cfg = Config(
    #     epochs=15,
    #     hidden="512,256,128",
    #     dropout=0.3,
    #     lr=5e-4,
    #     amp=False,
    #     eval_test=True,
    #     report="final",      # "off" | "final" | "epoch"
    #     topk_confusions=10,
    #     print_normalized_cm=True,
    # )
    cfg = Config()
    train(cfg)
'''