"""
╔══════════════════════════════════════════════════════════════════╗
║   FULL FINE-TUNING  —  bert-base-multilingual-cased (mBERT)    ║
║   Tâche  : Classification binaire (Hate Speech — Darija)        ║
║   Outputs: modèle .pth, CSV history/metrics/predictions,        ║
║             PNG courbes (5 métriques), confusion matrix, ROC-AUC ║
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
    roc_curve,
)
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# Set your paths
code_location = '/content/drive/MyDrive/MultilingualBert-mBert/'
results_folder = os.path.join(code_location, 'mbert-Results')

# IMPORTANT: Create the results folder FIRST
os.makedirs(results_folder, exist_ok=True)

# THEN change to that directory
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
    epochs        = 2,
    lr            = 2e-5,
    weight_decay  = 0.01,
    warmup_ratio  = 0.1,
    patience      = 3,          # early stopping
    seed          = 42,
    output_dir    = "outputs",
    num_labels    = 2,
    id2label      = {0: "Non-Hate", 1: "Hate"},
)

os.makedirs(CFG["output_dir"], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device : {DEVICE}")
print(f"📁  Output : {CFG['output_dir']}/\n")

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

# ─── PLOT — 5 métriques (Train vs Val) ───────────────────────────
def plot_metrics(history_df, path):
    metrics = ["loss", "accuracy", "f1", "precision", "recall"]
    titles  = ["Loss", "Accuracy", "F1 Score", "Precision", "Recall"]
    colors  = {"train": "#2563eb", "val": "#dc2626"}

    fig = plt.figure(figsize=(20, 12), facecolor="#0f172a")
    fig.suptitle(
        "mBERT Fine-Tuning — Training Curves\nDarija Hate Speech Detection",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.90, bottom=0.08)

    axes = [fig.add_subplot(gs[r, c]) for r, c in
            [(0,0),(0,1),(0,2),(1,0),(1,1)]]

    epochs = history_df["epoch"].tolist()

    for ax, metric, title in zip(axes, metrics, titles):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        train_vals = history_df[f"train_{metric}"].tolist()
        val_vals   = history_df[f"val_{metric}"].tolist()

        ax.plot(epochs, train_vals, color=colors["train"], lw=2.2,
                marker="o", markersize=4, label="Train")
        ax.plot(epochs, val_vals,   color=colors["val"],   lw=2.2,
                marker="s", markersize=4, label="Validation", linestyle="--")

        # best val marker
        best_idx = int(np.argmin(val_vals) if metric == "loss" else np.argmax(val_vals))
        ax.scatter([epochs[best_idx]], [val_vals[best_idx]],
                   color="#facc15", s=80, zorder=5)

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
    print(f"  📈 Courbes sauvegardées → {path}")

# ─── PLOT — Confusion Matrix ──────────────────────────────────────
def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Hate", "Hate"],
                yticklabels=["Non-Hate", "Hate"],
                linewidths=0.5, linecolor="#334155",
                annot_kws={"size": 18, "weight": "bold", "color": "white"},
                ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_title("Confusion Matrix — Test Set", color="white",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color="#94a3b8", fontsize=11)
    ax.set_ylabel("True Label",      color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")

    # per-class accuracy
    for i in range(2):
        total = cm[i].sum()
        acc   = cm[i, i] / total if total > 0 else 0
        ax.text(2.15, i + 0.5, f"Recall\n{acc:.2%}",
                va="center", ha="center", color="#facc15",
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  📊 Confusion matrix → {path}")

# ─── PLOT — ROC-AUC ──────────────────────────────────────────────
def plot_roc(y_true, y_prob, auc_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")

    ax.plot(fpr, tpr, color="#2563eb", lw=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="#64748b", lw=1.5,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")

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

    # ── 1. Data ──────────────────────────────────────────────────
    print("📂 Chargement des données...")
    train_df = pd.read_csv(CFG["train_csv"], sep=CFG["sep"])
    val_df   = pd.read_csv(CFG["val_csv"],   sep=CFG["sep"])
    test_df  = pd.read_csv(CFG["test_csv"],  sep=CFG["sep"])
    print(f"   Train : {len(train_df):,}  |  Val : {len(val_df):,}  |  Test : {len(test_df):,}")

    # ── 2. Tokenizer & Datasets ───────────────────────────────────
    print(f"\n🔡 Tokenizer : {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    train_ds = DarijaDataset(train_df, tokenizer, CFG["max_len"])
    val_ds   = DarijaDataset(val_df,   tokenizer, CFG["max_len"])
    test_ds  = DarijaDataset(test_df,  tokenizer, CFG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"] * 2,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"] * 2,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── 3. Model ──────────────────────────────────────────────────
    print(f"\n🤖 Chargement du modèle...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels=CFG["num_labels"],
        id2label=CFG["id2label"],
        label2id={v: k for k, v in CFG["id2label"].items()},
    ).to(DEVICE)

    # Weighted loss (déséquilibre 56/44)
    counts = train_df[CFG["label_col"]].value_counts().sort_index()
    weights = torch.tensor(
        [len(train_df) / (CFG["num_labels"] * c) for c in counts],
        dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"   Class weights : {weights.cpu().tolist()}")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Params total     : {total_params:,}")
    print(f"   Params trainable : {trainable_params:,}")

    # ── 4. Optimizer & Scheduler ──────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    total_steps   = len(train_loader) * CFG["epochs"]
    warmup_steps  = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── 5. Training loop ──────────────────────────────────────────
    print(f"\n🚀 Début du fine-tuning ({CFG['epochs']} epochs max)...\n")
    history = []
    best_f1    = -1.0
    patience_c = 0
    best_state = None
    TRAIN_START = time.time()

    for epoch in range(1, CFG["epochs"] + 1):
        ep_start = time.time()
        print(f"  Epoch {epoch}/{CFG['epochs']}  ", end="")

        train_m = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
        val_m, _, _, _ = evaluate(model, val_loader, criterion)

        ep_time = time.time() - ep_start
        row = {"epoch": epoch, "epoch_time_s": round(ep_time, 2)}
        for k, v in train_m.items(): row[f"train_{k}"] = round(v, 6)
        for k, v in val_m.items():   row[f"val_{k}"]   = round(v, 6)
        history.append(row)

        print(
            f"| train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}"
            f"  val_f1={val_m['f1']:.4f}  val_acc={val_m['accuracy']:.4f}"
            f"  [{ep_time:.0f}s]"
        )

        # Early stopping on val f1 (binary)
        if val_m["f1"] > best_f1:
            best_f1    = val_m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
            print(f"    ✅ Best model saved (f1={best_f1:.4f})")
        else:
            patience_c += 1
            if patience_c >= CFG["patience"]:
                print(f"\n⏹️  Early stopping at epoch {epoch} (patience={CFG['patience']})")
                break

    TRAIN_TIME = time.time() - TRAIN_START
    print(f"\n⏱️  Training time : {TRAIN_TIME/60:.2f} min ({TRAIN_TIME:.1f} s)")

    # Reload best model
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # ── 6. Test evaluation ────────────────────────────────────────
    print("\n📊 Évaluation sur le test set...")
    EVAL_START = time.time()
    test_m, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion)
    EVAL_TIME = time.time() - EVAL_START
    TOTAL_TIME = time.time() - TOTAL_START

    print("\n" + "═"*60)
    print("RAPPORT FINAL — TEST SET")
    print("═"*60)
    print(classification_report(y_true, y_pred,
          target_names=["Non-Hate", "Hate"], zero_division=0))
    print(f"  ROC-AUC     : {test_m['roc_auc']:.4f}")
    print(f"  Eval time   : {EVAL_TIME:.2f} s")
    print(f"  Total time  : {TOTAL_TIME/60:.2f} min")

    # ── 7. Save all outputs ───────────────────────────────────────
    OUT = CFG["output_dir"]
    print(f"\n💾 Sauvegarde des fichiers dans ./{OUT}/")

    # a) Modèle .pth
    model_path = f"{OUT}/mbert_darija_best.pth"
    torch.save({
        "model_state_dict": best_state,
        "config":           CFG,
        "best_f1":          best_f1,
        "id2label":         CFG["id2label"],
        "label2id":         {v: k for k, v in CFG["id2label"].items()},
    }, model_path)
    print(f"  🧠 Modèle         → {model_path}")

    # b) Tokenizer
    tok_dir = f"{OUT}/tokenizer"
    tokenizer.save_pretrained(tok_dir)
    print(f"  🔡 Tokenizer      → {tok_dir}/")

    # c) Config JSON
    cfg_path = f"{OUT}/config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({**CFG, "id2label": {str(k): v for k, v in CFG["id2label"].items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"  ⚙️  Config         → {cfg_path}")

    # d) History CSV
    history_df = pd.DataFrame(history)
    hist_path  = f"{OUT}/training_history.csv"
    history_df.to_csv(hist_path, index=False)
    print(f"  📋 History        → {hist_path}")

    # e) Test predictions CSV
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

    # f) Final metrics CSV
    timing = {
        "training_time_s":   round(TRAIN_TIME, 2),
        "training_time_min": round(TRAIN_TIME / 60, 2),
        "eval_time_s":       round(EVAL_TIME, 2),
        "total_time_s":      round(TOTAL_TIME, 2),
        "total_time_min":    round(TOTAL_TIME / 60, 2),
        "epochs_trained":    len(history),
        "best_epoch":        int(history_df["val_f1"].idxmax()) + 1,
        "total_params":      total_params,
        "trainable_params":  trainable_params,
    }
    metrics_dict = {**{f"test_{k}": round(v, 6) for k, v in test_m.items()}, **timing}
    metrics_df   = pd.DataFrame([metrics_dict])
    metrics_path = f"{OUT}/final_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  📈 Metrics        → {metrics_path}")

    # ── 8. Plots ──────────────────────────────────────────────────
    print("\n🎨 Génération des graphiques...")
    plot_metrics(history_df, f"{OUT}/training_curves.png")
    plot_confusion(y_true, y_pred, f"{OUT}/confusion_matrix.png")
    plot_roc(y_true, y_prob, test_m["roc_auc"], f"{OUT}/roc_auc.png")

    # ── 9. Summary ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("✅  FINE-TUNING COMPLET")
    print("═"*60)
    print(f"  Accuracy   : {test_m['accuracy']:.4f}")
    print(f"  Precision  : {test_m['precision']:.4f}")
    print(f"  Recall     : {test_m['recall']:.4f}")
    print(f"  F1 Score   : {test_m['f1']:.4f}")
    print(f"  ROC-AUC    : {test_m['roc_auc']:.4f}")
    print(f"\n  Training   : {TRAIN_TIME/60:.2f} min")
    print(f"  Total exec : {TOTAL_TIME/60:.2f} min")
    print(f"\n  Fichiers générés dans → ./{OUT}/")
    print("═"*60)