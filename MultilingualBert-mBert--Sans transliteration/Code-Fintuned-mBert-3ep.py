"""
╔══════════════════════════════════════════════════════════════════╗
║   FULL FINE-TUNING  —  bert-base-multilingual-cased (mBERT)    ║
║   Tâche  : Classification binaire (Hate Speech — Darija)        ║
║   Méthode: STRATIFIED 10-FOLD CROSS-VALIDATION                  ║
║   Outputs: 10 modèles .pth, CSV history/metrics/predictions,    ║
║            PNG courbes (Train vs Cross-Val), confusion matrix   ║
║            verte, ROC-AUC, classification_report CSV            ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ─────────────────────────────────────────────────────
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Set your paths
code_location = '/content/drive/MyDrive/MultilingualBert-mBert/'
results_folder = os.path.join(code_location, 'mbert-Results-CV10')

# Create the results folder
os.makedirs(results_folder, exist_ok=True)
os.chdir(results_folder)
print(f"Current working directory: {os.getcwd()}")

# ─── CONFIGURATION ───────────────────────────────────────────────
CFG = dict(
    model_name    = "bert-base-multilingual-cased",
    train_csv     = "/content/drive/MyDrive/data/trainAvant.csv",
    val_csv       = "/content/drive/MyDrive/data/validationAvant.csv",
    test_csv      = "/content/drive/MyDrive/data/testAvant.csv",
    sep           = ";",
    text_col      = "text",
    label_col     = "label",
    max_len       = 128,
    batch_size    = 16,
    epochs        = 3,              # Increased for CV
    lr            = 2e-5,
    weight_decay  = 0.01,
    warmup_ratio  = 0.1,
    n_folds       = 10,             # Stratified 10-fold CV
    seed          = 42,
    output_dir    = "outputs_cv10",
    num_labels    = 2,
    id2label      = {0: "Non-Hate", 1: "Hate"},
)

os.makedirs(CFG["output_dir"], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device : {DEVICE}")
print(f"📁  Output : {CFG['output_dir']}/")
print(f"🔢  Folds  : {CFG['n_folds']}-fold stratified CV\n")

# ─── DATASET ─────────────────────────────────────────────────────
class DarijaDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df[CFG["text_col"]].fillna("").astype(str).tolist()
        self.labels = df[CFG["label_col"]].astype(int).tolist()
        self.tok    = tokenizer
        self.max_len= max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ─── HELPERS ─────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    m = dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
        recall    = recall_score(y_true, y_pred,    pos_label=1, average="binary", zero_division=0),
        f1        = f1_score(y_true, y_pred,        pos_label=1, average="binary", zero_division=0),
    )
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            m["roc_auc"] = float("nan")
    return m

def fmt(v): return f"{v:.4f}"

# ─── TRAINING LOOP (one epoch) ────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, all_labels, all_preds = 0.0, [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        loss    = criterion(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    m = compute_metrics(all_labels, all_preds)
    m["loss"] = avg_loss
    return m

# ─── EVALUATION ──────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        loss    = criterion(outputs.logits, labels)
        total_loss += loss.item() * len(labels)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.array(all_probs)
    m = compute_metrics(all_labels, all_preds, all_probs)
    m["loss"] = avg_loss
    return m, np.array(all_labels), np.array(all_preds), all_probs

# ─── PLOT — Train vs Cross-Validation metrics ─────────────────────
def plot_cv_metrics(fold_history, path):
    """Plot training vs cross-validation curves across folds"""
    metrics = ["loss", "accuracy", "f1", "precision", "recall"]
    titles  = ["Loss", "Accuracy", "F1 Score", "Precision", "Recall"]
    colors  = {"train": "#2ecc71", "val": "#e74c3c"}

    # Calculate mean and std across folds
    epochs = range(1, CFG["epochs"] + 1)

    fig = plt.figure(figsize=(20, 12), facecolor="#0f172a")
    fig.suptitle(
        f"mBERT Fine-Tuning — {CFG['n_folds']}-Fold Cross-Validation\nDarija Hate Speech Detection",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.90, bottom=0.08)

    axes = [fig.add_subplot(gs[r, c]) for r, c in
            [(0,0),(0,1),(0,2),(1,0),(1,1)]]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        # Collect metrics across folds
        train_vals = []
        val_vals = []
        for fold_data in fold_history:
            train_vals.append([epoch_data[f"train_{metric}"] for epoch_data in fold_data])
            val_vals.append([epoch_data[f"val_{metric}"] for epoch_data in fold_data])

        # Calculate mean and std
        train_mean = np.mean(train_vals, axis=0)
        train_std = np.std(train_vals, axis=0)
        val_mean = np.mean(val_vals, axis=0)
        val_std = np.std(val_vals, axis=0)

        # Plot with confidence intervals
        ax.plot(epochs, train_mean, color=colors["train"], lw=2.5,
                marker="o", markersize=4, label="Train (mean)")
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=colors["train"])

        ax.plot(epochs, val_mean, color=colors["val"], lw=2.5,
                marker="s", markersize=4, label="Cross-Val (mean)", linestyle="--")
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color=colors["val"])

        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.grid(axis="y", color="#334155", lw=0.7, linestyle="--")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155",
                  labelcolor="white", loc="best")

    # hide 6th cell
    fig.add_subplot(gs[1, 2]).set_visible(False)

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📈 Courbes CV sauvegardées → {path}")

# ─── PLOT — Green Confusion Matrix ────────────────────────────────
def plot_confusion_green(y_true, y_pred, path):
    """Generate a beautiful green-themed confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    # Custom green colormap
    cmap = sns.light_palette("#2ecc71", as_cmap=True)

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["Non-Hate", "Hate"],
                yticklabels=["Non-Hate", "Hate"],
                linewidths=2, linecolor="#27ae60",
                annot_kws={"size": 20, "weight": "bold", "color": "#0f172a"},
                ax=ax, cbar_kws={"shrink": 0.8, "label": "Count"})

    ax.set_title("Confusion Matrix — Test Set\nGreen Theme", color="white",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", color="#94a3b8", fontsize=12)
    ax.set_ylabel("True Label",      color="#94a3b8", fontsize=12)
    ax.tick_params(colors="#94a3b8", labelsize=11)

    # Add per-class accuracy
    for i in range(2):
        total = cm[i].sum()
        acc   = cm[i, i] / total if total > 0 else 0
        ax.text(2.15, i + 0.5, f"Class Acc\n{acc:.2%}",
                va="center", ha="center", color="#facc15",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", alpha=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Green confusion matrix → {path}")

# ─── PLOT — ROC-AUC ──────────────────────────────────────────────
def plot_roc(y_true, y_prob, auc_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")

    ax.plot(fpr, tpr, color="#2ecc71", lw=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#64748b", lw=1.5,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2ecc71")

    ax.set_title("ROC-AUC Curve — Test Set", color="white",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=11)
    ax.set_ylabel("True Positive Rate",  color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.grid(color="#334155", lw=0.6, linestyle="--")
    ax.legend(fontsize=10, facecolor="#1e293b", edgecolor="#334155",
              labelcolor="white", loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📉 ROC-AUC → {path}")

# ═══════════════════════════════════════════════════════════════════
#                         MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    TOTAL_START = time.time()

    # ── 1. Load and combine train+validation ─────────────────────
    print("📂 Chargement des données...")
    train_df = pd.read_csv(CFG["train_csv"], sep=CFG["sep"])
    val_df   = pd.read_csv(CFG["val_csv"],   sep=CFG["sep"])
    test_df  = pd.read_csv(CFG["test_csv"],  sep=CFG["sep"])

    # Combine train and validation for CV
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"   Train original    : {len(train_df):,}")
    print(f"   Validation orig.  : {len(val_df):,}")
    print(f"   Combined (CV)     : {len(combined_df):,}")
    print(f"   Test              : {len(test_df):,}")

    # Check class distribution
    print(f"\n📊 Class distribution in combined set:")
    class_dist = combined_df[CFG["label_col"]].value_counts()
    for label, count in class_dist.items():
        print(f"   {CFG['id2label'][label]}: {count} ({count/len(combined_df)*100:.1f}%)")

    # ── 2. Tokenizer ──────────────────────────────────────────────
    print(f"\n🔡 Tokenizer : {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    # ── 3. Stratified 10-Fold CV ──────────────────────────────────
    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    X = combined_df[CFG["text_col"]].values
    y = combined_df[CFG["label_col"]].values

    # Store results for each fold
    fold_results = []
    fold_history = []
    all_fold_test_predictions = []

    print(f"\n🚀 Starting {CFG['n_folds']}-Fold Stratified Cross-Validation...\n")
    CV_START = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{CFG['n_folds']}")
        print(f"{'='*60}")

        # Split data
        train_fold = combined_df.iloc[train_idx].reset_index(drop=True)
        val_fold = combined_df.iloc[val_idx].reset_index(drop=True)

        print(f"   Train size: {len(train_fold):,} ({len(train_fold)/len(combined_df)*100:.1f}%)")
        print(f"   Val size:   {len(val_fold):,} ({len(val_fold)/len(combined_df)*100:.1f}%)")

        # Create datasets
        train_ds = DarijaDataset(train_fold, tokenizer, CFG["max_len"])
        val_ds = DarijaDataset(val_fold, tokenizer, CFG["max_len"])

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2,
                                shuffle=False, num_workers=2, pin_memory=True)

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            CFG["model_name"],
            num_labels=CFG["num_labels"],
            id2label=CFG["id2label"],
            label2id={v: k for k, v in CFG["id2label"].items()},
        ).to(DEVICE)

        # Weighted loss
        counts = train_fold[CFG["label_col"]].value_counts().sort_index()
        weights = torch.tensor(
            [len(train_fold) / (CFG["num_labels"] * c) for c in counts],
            dtype=torch.float
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Optimizer & scheduler
        optimizer = AdamW(model.parameters(),
                          lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        total_steps = len(train_loader) * CFG["epochs"]
        warmup_steps = int(total_steps * CFG["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training for this fold
        fold_epoch_history = []
        best_f1 = -1.0
        patience_c = 0
        best_state = None

        for epoch in range(1, CFG["epochs"] + 1):
            ep_start = time.time()

            train_m = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
            val_m, _, _, _ = evaluate(model, val_loader, criterion)

            ep_time = time.time() - ep_start
            row = {"epoch": epoch, "epoch_time_s": round(ep_time, 2)}
            for k, v in train_m.items(): row[f"train_{k}"] = round(v, 6)
            for k, v in val_m.items():   row[f"val_{k}"]   = round(v, 6)
            fold_epoch_history.append(row)

            print(f"   Epoch {epoch}/{CFG['epochs']} | train_loss={train_m['loss']:.4f} "
                  f"val_loss={val_m['loss']:.4f} val_f1={val_m['f1']:.4f} [{ep_time:.0f}s]")

            # Early stopping
            if val_m["f1"] > best_f1:
                best_f1 = val_m["f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_c = 0
            else:
                patience_c += 1
                if patience_c >= CFG["patience"]:
                    print(f"   ⏹️ Early stopping at epoch {epoch}")
                    break

        # Save best model for this fold
        model_path = f"{CFG['output_dir']}/mbert_fold{fold}_best.pth"
        torch.save({
            "model_state_dict": best_state,
            "config": CFG,
            "best_f1": best_f1,
            "fold": fold,
            "id2label": CFG["id2label"],
            "label2id": {v: k for k, v in CFG["id2label"].items()},
        }, model_path)
        print(f"   💾 Model saved: {model_path}")

        fold_results.append({
            "fold": fold,
            "best_val_f1": best_f1,
            "model_path": model_path
        })
        fold_history.append(fold_epoch_history)

    CV_TIME = time.time() - CV_START
    print(f"\n⏱️ Cross-validation time : {CV_TIME/60:.2f} min ({CV_TIME:.1f} s)")

    # ── 4. Train final model on full combined data ────────────────
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL ON FULL COMBINED DATA")
    print(f"{'='*60}")

    full_train_ds = DarijaDataset(combined_df, tokenizer, CFG["max_len"])
    full_train_loader = DataLoader(full_train_ds, batch_size=CFG["batch_size"],
                                    shuffle=True, num_workers=2, pin_memory=True)

    final_model = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels=CFG["num_labels"],
        id2label=CFG["id2label"],
        label2id={v: k for k, v in CFG["id2label"].items()},
    ).to(DEVICE)

    # Weighted loss for full training
    counts = combined_df[CFG["label_col"]].value_counts().sort_index()
    weights = torch.tensor(
        [len(combined_df) / (CFG["num_labels"] * c) for c in counts],
        dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(final_model.parameters(),
                      lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    total_steps = len(full_train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    final_history = []
    best_final_f1 = -1.0
    best_final_state = None

    FINAL_START = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        ep_start = time.time()
        train_m = train_one_epoch(final_model, full_train_loader, optimizer, scheduler, criterion)
        ep_time = time.time() - ep_start
        row = {"epoch": epoch, "epoch_time_s": round(ep_time, 2)}
        for k, v in train_m.items(): row[f"train_{k}"] = round(v, 6)
        final_history.append(row)

        print(f"   Epoch {epoch}/{CFG['epochs']} | train_loss={train_m['loss']:.4f} "
              f"train_f1={train_m['f1']:.4f} [{ep_time:.0f}s]")

        if train_m["f1"] > best_final_f1:
            best_final_f1 = train_m["f1"]
            best_final_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}

    FINAL_TIME = time.time() - FINAL_START

    # Load best final model
    final_model.load_state_dict({k: v.to(DEVICE) for k, v in best_final_state.items()})

    # ── 5. Evaluation on test set ─────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*60}")

    test_ds = DarijaDataset(test_df, tokenizer, CFG["max_len"])
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"] * 2,
                             shuffle=False, num_workers=2, pin_memory=True)

    EVAL_START = time.time()
    test_m, y_true, y_pred, y_prob = evaluate(final_model, test_loader, criterion)
    EVAL_TIME = time.time() - EVAL_START
    TOTAL_TIME = time.time() - TOTAL_START

    # Classification report
    class_report = classification_report(y_true, y_pred,
                                        target_names=["Non-Hate", "Hate"],
                                        output_dict=True, zero_division=0)

    print("\n" + "═"*60)
    print("RAPPORT FINAL — TEST SET")
    print("═"*60)
    print(classification_report(y_true, y_pred,
          target_names=["Non-Hate", "Hate"], zero_division=0))
    print(f"  ROC-AUC     : {test_m['roc_auc']:.4f}")

    # ── 6. Save all outputs ───────────────────────────────────────
    OUT = CFG["output_dir"]
    print(f"\n💾 Sauvegarde des fichiers dans ./{OUT}/")

    # a) Final model
    final_model_path = f"{OUT}/mbert_final_best.pth"
    torch.save({
        "model_state_dict": best_final_state,
        "config": CFG,
        "best_f1": best_final_f1,
        "id2label": CFG["id2label"],
        "label2id": {v: k for k, v in CFG["id2label"].items()},
    }, final_model_path)
    print(f"  🧠 Final model    → {final_model_path}")

    # b) All fold models summary
    fold_summary_df = pd.DataFrame(fold_results)
    fold_summary_path = f"{OUT}/fold_summary.csv"
    fold_summary_df.to_csv(fold_summary_path, index=False)
    print(f"  📊 Fold summary   → {fold_summary_path}")

    # c) Tokenizer
    tok_dir = f"{OUT}/tokenizer"
    tokenizer.save_pretrained(tok_dir)
    print(f"  🔡 Tokenizer      → {tok_dir}/")

    # d) Config JSON
    cfg_path = f"{OUT}/config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({**CFG, "id2label": {str(k): v for k, v in CFG["id2label"].items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"  ⚙️ Config         → {cfg_path}")

    # e) CV History JSON (all folds)
    cv_history_path = f"{OUT}/cv_training_history.json"
    with open(cv_history_path, "w", encoding="utf-8") as f:
        json.dump(fold_history, f, ensure_ascii=False, indent=2)
    print(f"  📋 CV history     → {cv_history_path}")

    # f) Final training history CSV
    final_history_df = pd.DataFrame(final_history)
    final_hist_path = f"{OUT}/final_training_history.csv"
    final_history_df.to_csv(final_hist_path, index=False)
    print(f"  📋 Final history  → {final_hist_path}")

    # g) Test predictions CSV
    pred_df = pd.DataFrame({
        "text":               test_df[CFG["text_col"]].tolist(),
        "true_label":         y_true,
        "predicted_label":    y_pred,
        "is_correct":         (y_true == y_pred).astype(int),
        "probability_non_hate": y_prob[:, 0].round(6),
        "probability_hate":     y_prob[:, 1].round(6),
    })
    pred_path = f"{OUT}/test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  🎯 Predictions    → {pred_path}")

    # h) Classification report CSV
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_path = f"{OUT}/classification_report.csv"
    class_report_df.to_csv(class_report_path, index=True)
    print(f"  📋 Class report   → {class_report_path}")

    # i) Final metrics CSV
    timing = {
        "cv_time_s":         round(CV_TIME, 2),
        "cv_time_min":       round(CV_TIME / 60, 2),
        "final_train_time_s": round(FINAL_TIME, 2),
        "final_train_time_min": round(FINAL_TIME / 60, 2),
        "eval_time_s":       round(EVAL_TIME, 2),
        "total_time_s":      round(TOTAL_TIME, 2),
        "total_time_min":    round(TOTAL_TIME / 60, 2),
        "n_folds":           CFG["n_folds"],
        "best_fold_f1_mean": round(fold_summary_df["best_val_f1"].mean(), 6),
        "best_fold_f1_std":  round(fold_summary_df["best_val_f1"].std(), 6),
    }
    metrics_dict = {**{f"test_{k}": round(v, 6) for k, v in test_m.items()}, **timing}
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = f"{OUT}/final_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  📈 Metrics        → {metrics_path}")

    # ── 7. Plots ──────────────────────────────────────────────────
    print("\n🎨 Génération des graphiques...")

    # Plot CV metrics (train vs cross-val)
    plot_cv_metrics(fold_history, f"{OUT}/cv_training_curves.png")

    # Plot green confusion matrix
    plot_confusion_green(y_true, y_pred, f"{OUT}/confusion_matrix.png")

    # Plot ROC-AUC
    plot_roc(y_true, y_prob, test_m["roc_auc"], f"{OUT}/roc_auc.png")

    # Additional: Plot final training loss curve
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    epochs = final_history_df["epoch"].tolist()
    train_loss = final_history_df["train_loss"].tolist()
    train_f1 = final_history_df["train_f1"].tolist()

    ax.plot(epochs, train_loss, color="#2ecc71", lw=2.5, marker="o", label="Training Loss")
    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Loss", color="#94a3b8", fontsize=11)
    ax.set_title("Final Model Training Loss", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", color="#334155", lw=0.7, linestyle="--")
    ax.legend(fontsize=10, facecolor="#1e293b", edgecolor="#334155", labelcolor="white")

    plt.tight_layout()
    plt.savefig(f"{OUT}/final_training_loss.png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📉 Final loss curve → {OUT}/final_training_loss.png")

    # ── 8. Summary ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("✅  STRATIFIED 10-FOLD CV COMPLETED")
    print("═"*60)
    print(f"\n📊 CROSS-VALIDATION SUMMARY:")
    print(f"   Mean best F1 across folds: {fold_summary_df['best_val_f1'].mean():.4f} ± {fold_summary_df['best_val_f1'].std():.4f}")
    print(f"\n📊 FINAL TEST RESULTS:")
    print(f"   Accuracy   : {test_m['accuracy']:.4f}")
    print(f"   Precision  : {test_m['precision']:.4f}")
    print(f"   Recall     : {test_m['recall']:.4f}")
    print(f"   F1 Score   : {test_m['f1']:.4f}")
    print(f"   ROC-AUC    : {test_m['roc_auc']:.4f}")
    print(f"\n⏱️ TIMING:")
    print(f"   CV time       : {CV_TIME/60:.2f} min")
    print(f"   Final training: {FINAL_TIME/60:.2f} min")
    print(f"   Total exec    : {TOTAL_TIME/60:.2f} min")
    print(f"\n📁 Fichiers générés dans → ./{OUT}/")
    print("═"*60)