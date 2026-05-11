"""
╔══════════════════════════════════════════════════════════════════╗
║   ZERO-SHOT EVALUATION — bert-base-multilingual-cased (mBERT)  ║
║   Tâche  : Classification binaire (Hate Speech — Darija)        ║
║   BUT    : Évaluer le modèle sans fine-tuning (couche aléatoire)║
║   Outputs: Métriques, matrice de confusion, classification report║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ─────────────────────────────────────────────────────
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)
warnings.filterwarnings("ignore")

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Set your paths
code_location = '/content/drive/MyDrive/MultilingualBert-mBert/'
results_folder = os.path.join(code_location, 'mbert-ZeroShot-Evaluation')

# Create results folder
os.makedirs(results_folder, exist_ok=True)
os.chdir(results_folder)
print(f"Current working directory: {os.getcwd()}")

# ─── CONFIGURATION ───────────────────────────────────────────────
CFG = dict(
    model_name    = "bert-base-multilingual-cased",
    test_csv      = "/content/drive/MyDrive/data/testAvant.csv",
    sep           = ";",
    text_col      = "text",
    label_col     = "label",
    max_len       = 128,
    batch_size    = 32,  # Plus grand car pas d'entraînement
    num_labels    = 2,
    id2label      = {0: "Non-Hate", 1: "Hate"},
    output_dir    = "zero_shot_results",
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

# ─── EVALUATION ──────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    """Évalue le modèle sans fine-tuning"""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels    = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs))

def compute_metrics(y_true, y_pred, y_prob=None):
    """Calcule les métriques de classification"""
    m = dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, pos_label=1,
                                   average="binary", zero_division=0),
        recall    = recall_score(y_true, y_pred, pos_label=1,
                                average="binary", zero_division=0),
        f1        = f1_score(y_true, y_pred, pos_label=1,
                            average="binary", zero_division=0),
    )
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            m["roc_auc"] = float("nan")
    return m

# ─── PLOT — Confusion Matrix ─────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, path):
    """Generate confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Hate", "Hate"],
                yticklabels=["Non-Hate", "Hate"],
                linewidths=2, linecolor="white",
                annot_kws={"size": 16, "weight": "bold"},
                ax=ax, cbar_kws={"shrink": 0.8, "label": "Count"})

    ax.set_title("mBERT Zero-Shot - Confusion Matrix\n(No Fine-Tuning)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # Add per-class accuracy
    for i in range(2):
        total = cm[i].sum()
        acc   = cm[i, i] / total if total > 0 else 0
        ax.text(2.15, i + 0.5, f"Class Acc\n{acc:.2%}",
                va="center", ha="center", color="red",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix → {path}")

# ─── PLOT — ROC-AUC ──────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob, auc_score, path):
    """Generate ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

    ax.plot(fpr, tpr, color="#2563eb", lw=2.5,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")

    ax.set_title("mBERT Zero-Shot - ROC-AUC Curve\n(No Fine-Tuning)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📉 ROC-AUC curve → {path}")

# ─── PLOT — Prediction Distribution ──────────────────────────────
def plot_prediction_distribution(y_true, y_pred, y_prob, path):
    """Plot prediction confidence distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    # Plot 1: Confidence distribution by true class
    for class_idx, class_name in enumerate(["Non-Hate", "Hate"]):
        mask = y_true == class_idx
        axes[0].hist(y_prob[mask, 1], bins=20, alpha=0.5,
                     label=f"True {class_name}", density=True)

    axes[0].set_xlabel("Probability of Hate Class", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("Prediction Confidence by True Class", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Plot 2: Confusion comparison
    correct = (y_true == y_pred)
    incorrect = ~correct

    axes[1].hist(y_prob[correct, 1], bins=20, alpha=0.7,
                 label=f"Correct Predictions ({correct.sum()})",
                 color="green", density=True)
    axes[1].hist(y_prob[incorrect, 1], bins=20, alpha=0.7,
                 label=f"Incorrect Predictions ({incorrect.sum()})",
                 color="red", density=True)
    axes[1].set_xlabel("Probability of Hate Class", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.suptitle("mBERT Zero-Shot - Prediction Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Prediction distribution → {path}")

# ═══════════════════════════════════════════════════════════════════
#                         MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    TOTAL_START = time.time()

    # ── 1. Load data ─────────────────────────────────────────────
    print("📂 Chargement des données...")
    test_df = pd.read_csv(CFG["test_csv"], sep=CFG["sep"])
    print(f"   Test samples: {len(test_df):,}")

    # Check class distribution
    print(f"\n📊 Class distribution in test set:")
    class_dist = test_df[CFG["label_col"]].value_counts()
    for label, count in class_dist.items():
        print(f"   {CFG['id2label'][label]}: {count} ({count/len(test_df)*100:.1f}%)")

    # ── 2. Load tokenizer and model (NO FINE-TUNING) ─────────────
    print(f"\n🤖 Loading mBERT model WITHOUT fine-tuning...")
    print(f"   ⚠️  The classification head is randomly initialized!")

    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    # Load model with random classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels=CFG["num_labels"],
        id2label=CFG["id2label"],
        label2id={v: k for k, v in CFG["id2label"].items()},
    ).to(DEVICE)

    # Freeze all BERT layers to simulate zero-shot (keep only classification head trainable)
    # Note: The classification head is already random, so we don't need to freeze,
    # but we do this to show that we're NOT fine-tuning
    for param in model.bert.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params (random head only): {trainable_params:,}")

    # ── 3. Create dataloader ─────────────────────────────────────
    print(f"\n📦 Creating dataloader...")
    test_ds = DarijaDataset(test_df, tokenizer, CFG["max_len"])
    test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── 4. Evaluate model ────────────────────────────────────────
    print(f"\n🧪 Evaluating zero-shot performance...")
    EVAL_START = time.time()
    y_true, y_pred, y_prob = evaluate(model, test_loader)
    EVAL_TIME = time.time() - EVAL_START

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    TOTAL_TIME = time.time() - TOTAL_START

    # Classification report
    class_report = classification_report(y_true, y_pred,
                                        target_names=["Non-Hate", "Hate"],
                                        output_dict=True, zero_division=0)

    # ── 5. Display results ───────────────────────────────────────
    print("\n" + "═"*70)
    print("📊 ZERO-SHOT EVALUATION RESULTS - mBERT (No Fine-Tuning)")
    print("═"*70)
    print("\n" + classification_report(y_true, y_pred,
          target_names=["Non-Hate", "Hate"], zero_division=0))
    print(f"\n{'─'*50}")
    print("SUMMARY METRICS:")
    print(f"{'─'*50}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision  : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"  Recall     : {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score   : {metrics['f1']:.4f}  ({metrics['f1']*100:.2f}%)")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
    print(f"{'─'*50}")

    # Baseline comparison (random classifier)
    # Baseline comparison (random classifier)
    if hasattr(class_dist, 'max'):
        random_acc = class_dist.max() / len(test_df)
    else:
        random_acc = max(class_dist) / len(test_df)
    print(f"\n📌 BASELINE COMPARISON:")
    print(f"  Random classifier accuracy (majority class): {random_acc:.4f} ({random_acc*100:.2f}%)")
    print(f"  mBERT Zero-Shot improvement: +{(metrics['accuracy'] - random_acc)*100:.2f}%")

    # ── 6. Save outputs ──────────────────────────────────────────
    OUT = CFG["output_dir"]
    print(f"\n💾 Saving results to ./{OUT}/")

    # Save metrics
    metrics_dict = {
        **{f"test_{k}": round(v, 6) for k, v in metrics.items()},
        "eval_time_s": round(EVAL_TIME, 2),
        "total_time_s": round(TOTAL_TIME, 2),
        "random_baseline_acc": round(random_acc, 6),
        "improvement_over_random": round(metrics['accuracy'] - random_acc, 6),
    }
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(f"{OUT}/zero_shot_metrics.csv", index=False)
    print(f"  📈 Metrics → {OUT}/zero_shot_metrics.csv")

    # Save classification report
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(f"{OUT}/classification_report.csv", index=True)
    print(f"  📋 Classification report → {OUT}/classification_report.csv")

    # Save predictions
    pred_df = pd.DataFrame({
        "text": test_df[CFG["text_col"]].tolist(),
        "true_label": y_true,
        "predicted_label": y_pred,
        "is_correct": (y_true == y_pred).astype(int),
        "prob_non_hate": y_prob[:, 0].round(6),
        "prob_hate": y_prob[:, 1].round(6),
    })
    pred_df.to_csv(f"{OUT}/predictions.csv", index=False)
    print(f"  🎯 Predictions → {OUT}/predictions.csv")

    # Save misclassified examples
    misclassified = pred_df[pred_df["is_correct"] == 0]
    misclassified.to_csv(f"{OUT}/misclassified_examples.csv", index=False)
    print(f"  ⚠️  Misclassified ({len(misclassified)}) → {OUT}/misclassified_examples.csv")

    # Save config
    with open(f"{OUT}/config.json", "w", encoding="utf-8") as f:
        json.dump({**CFG, "id2label": {str(k): v for k, v in CFG["id2label"].items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"  ⚙️  Config → {OUT}/config.json")

    # ── 7. Generate plots ────────────────────────────────────────
    print("\n🎨 Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, f"{OUT}/confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, metrics["roc_auc"], f"{OUT}/roc_auc_curve.png")
    plot_prediction_distribution(y_true, y_pred, y_prob, f"{OUT}/prediction_distribution.png")

    # Additional: Bar chart comparison with fine-tuned version
    if os.path.exists("../mbert-Results-3ep-CV10/outputs_cv10/final_metrics.csv"):
        try:
            ft_metrics = pd.read_csv("../mbert-Results-3ep-CV10/outputs_cv10/final_metrics.csv")
            fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")

            metrics_compare = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            zero_shot_vals = [metrics[m] for m in metrics_compare]
            ft_vals = [ft_metrics[f"test_{m}"].iloc[0] for m in metrics_compare]

            x = np.arange(len(metrics_compare))
            width = 0.35

            bars1 = ax.bar(x - width/2, zero_shot_vals, width, label="Zero-Shot (No Fine-Tuning)",
                          color="#2563eb", alpha=0.8)
            bars2 = ax.bar(x + width/2, ft_vals, width, label="Fine-Tuned mBERT",
                          color="#16a34a", alpha=0.8)

            ax.set_ylabel("Score", fontsize=11)
            ax.set_title("mBERT Performance: Zero-Shot vs Fine-Tuned",
                        fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([m.capitalize() for m in metrics_compare])
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(f"{OUT}/comparison_with_finetuned.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  📊 Comparison chart → {OUT}/comparison_with_finetuned.png")
        except:
            print(f"  ⚠️  Fine-tuned metrics not found for comparison")

    # ── 8. Summary ────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("✅ ZERO-SHOT EVALUATION COMPLETED")
    print("═"*70)
    print(f"\n📊 KEY FINDINGS:")
    print(f"   • mBERT without fine-tuning achieves {metrics['accuracy']*100:.2f}% accuracy")
    print(f"   • Random baseline (majority class): {random_acc*100:.2f}%")
    print(f"   • Improvement over random: +{(metrics['accuracy']-random_acc)*100:.2f}%")
    print(f"   • F1 Score: {metrics['f1']*100:.2f}%")
    print(f"   • ROC-AUC: {metrics['roc_auc']*100:.2f}%")

    print(f"\n💡 INTERPRETATION:")
    if metrics['accuracy'] > 0.7:
        print(f"   ✅ mBERT already shows good zero-shot capabilities for Darija!")
    elif metrics['accuracy'] > 0.5:
        print(f"   📈 mBERT outperforms random baseline but needs fine-tuning")
    else:
        print(f"   ⚠️  Zero-shot performance is poor - fine-tuning is essential")

    print(f"\n⏱️  Evaluation time: {EVAL_TIME:.2f} seconds")
    print(f"\n📁 All files saved in: ./{OUT}/")
    print("═"*70)