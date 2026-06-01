import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (BertTokenizer, BertForSequenceClassification,
                           get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score,
                              roc_curve, roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
import warnings
import os
import pickle
import json
import random
import time
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Style des graphiques
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")

# Fixer les graines pour reproductibilité
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé: {device}")

# ============================================================
# PARAMÈTRES — bert-base-multilingual-cased
# Standard hyperparameters recommended in the original BERT paper
# and widely validated for mBERT fine-tuning:
#   - LR    : 2e-5 or 3e-5  (2e-5 is safest for most tasks)
#   - Epochs: 3 or 4        (mBERT converges faster than monolingual)
#   - Batch : 16 or 32
#   - Max_len: 128           (sufficient for short social-media texts)
#   - Warmup: 10% of total steps
#   - Weight decay: 0.01
# ============================================================
FIXED_EPOCHS   = 4          # 4 époques (mBERT converges well in 3-4)
K_FOLDS        = 5          # 5 folds validation croisée
MODEL_NAME     = "bert-base-multilingual-cased"
MAX_LEN        = 128
BATCH_SIZE     = 16
LEARNING_RATE  = 2e-5       # Standard pour mBERT (Devlin et al., 2019)
WEIGHT_DECAY   = 0.01       # L2 regularisation standard BERT
WARMUP_RATIO   = 0.1        # 10% des steps en warmup
DROPOUT        = 0.1        # Dropout BERT par défaut (ne pas modifier)
CLIP_NORM      = 1.0        # Gradient clipping standard
THRESHOLD      = 0.5        # Seuil décision binaire

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def set_seed(seed):
    """Fixer toutes les graines aléatoires"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ============================================================
# DATASET PYTORCH
# ============================================================

class DarijaDataset(Dataset):
    """Dataset PyTorch pour bert-base-multilingual-cased"""

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':         torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

def load_data(kfold_path='/content/drive/MyDrive/BigData_11_2026/data/kfoldsdata.csv',
              test_path='/content/drive/MyDrive/BigData_11_2026/data/test.csv',
              text_col='text', label_col='label'):
    """Charger les données K-Fold et test"""

    print(f"\nChargement des données K-Folds depuis: {kfold_path}")
    if not os.path.exists(kfold_path):
        raise FileNotFoundError(f"Fichier non trouvé: {kfold_path}")

    df_kfold = pd.read_csv(kfold_path, sep=';')
    print(f"Chargé: {len(df_kfold)} échantillons")

    for col in [text_col, label_col]:
        if col not in df_kfold.columns:
            raise ValueError(f"Colonne '{col}' non trouvée")

    X_kfold = df_kfold[text_col].astype(str).tolist()
    y_kfold = df_kfold[label_col].tolist()

    print("\nDistribution des classes:")
    for label, count in pd.Series(y_kfold).value_counts().items():
        print(f"  {label}: {count} ({count/len(y_kfold)*100:.1f}%)")

    X_test, y_test = [], []
    if os.path.exists(test_path):
        print(f"\nChargement test depuis: {test_path}")
        df_test = pd.read_csv(test_path, sep=';')
        X_test  = df_test[text_col].astype(str).tolist()
        y_test  = df_test[label_col].tolist()
    else:
        print(f"\nAttention: test.csv non trouvé")

    label_encoder = LabelEncoder()
    all_labels_for_fit = y_kfold + (y_test if y_test else [])
    label_encoder.fit(all_labels_for_fit)

    y_kfold_enc = label_encoder.transform(y_kfold)
    y_test_enc  = label_encoder.transform(y_test) if y_test else np.array([])

    print(f"\nDonnées chargées: {len(X_kfold)} samples, {len(label_encoder.classes_)} classes")

    return X_kfold, y_kfold_enc, X_test, y_test_enc, label_encoder

# ============================================================
# ENTRAINEMENT ET EVALUATION
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, criterion):
    """Entraîner une époque"""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, num_classes, threshold=THRESHOLD):
    """Évaluer le modèle"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    is_binary = (num_classes == 2)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            if is_binary:
                pos_probs = probs[:, 1]
                preds     = (pos_probs >= threshold).astype(int)
                all_probs.extend(pos_probs.tolist())
            else:
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs.tolist())

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg       = 'binary' if is_binary else 'weighted'
    avg_loss  = total_loss / len(loader)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=avg, zero_division=0)
    recall    = recall_score(all_labels, all_preds, average=avg, zero_division=0)
    f1        = f1_score(all_labels, all_preds, average=avg, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels, all_probs

def compute_auc(true_labels, probs, num_classes):
    """Calculer l'AUC"""
    true = np.array(true_labels)
    if num_classes == 2:
        return roc_auc_score(true, np.array(probs))
    else:
        prob_matrix = np.array(probs)
        return roc_auc_score(true, prob_matrix, multi_class='ovr', average='weighted')

def build_model_and_optimizer(num_classes, epochs, steps_per_epoch):
    """
    Construire mBERT avec optimizer et scheduler.

    Notes pour bert-base-multilingual-cased:
    - On NE modifie PAS hidden_dropout_prob ni attention_probs_dropout_prob :
      le défaut de mBERT est 0.1 et c'est optimal. Changer ces valeurs
      post-chargement peut déstabiliser les poids pré-entraînés.
    - Warmup à 10 % des steps est la recommandation BERT officielle.
    - AdamW avec weight_decay=0.01, eps=1e-8 (défaut PyTorch/HuggingFace).
    """
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        hidden_dropout_prob=DROPOUT,               # 0.1 = valeur par défaut mBERT
        attention_probs_dropout_prob=DROPOUT,      # idem
        ignore_mismatched_sizes=True
    ).to(device)

    print("Toutes les couches sont entraînables (full fine-tuning)")

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8                   # eps recommandé pour stabilité mBERT
    )

    total_steps  = steps_per_epoch * epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps} ({100*warmup_steps/total_steps:.0f}%)")

    return model, optimizer, scheduler

# ============================================================
# ENTRAINEMENT AVEC HISTORIQUE
# ============================================================

def train_with_history(model, optimizer, scheduler, train_loader, val_loader,
                       criterion, epochs, num_classes):
    """Entraîner et retourner l'historique complet"""

    history            = []
    val_probs_per_epoch = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)

        _, train_acc, train_prec, train_rec, train_f1, _, _, _ = evaluate(
            model, train_loader, criterion, num_classes
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, _, val_labels, val_probs = evaluate(
            model, val_loader, criterion, num_classes
        )

        val_auc = compute_auc(val_labels, val_probs, num_classes)

        epoch_data = {
            'epoch':          epoch + 1,
            'train_loss':     train_loss,
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall':   train_rec,
            'train_f1':       train_f1,
            'val_loss':       val_loss,
            'val_accuracy':   val_acc,
            'val_precision':  val_prec,
            'val_recall':     val_rec,
            'val_f1':         val_f1,
            'val_auc':        val_auc
        }
        history.append(epoch_data)
        val_probs_per_epoch.append(val_probs)

        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} auc: {val_auc:.4f}")

    return history, val_probs_per_epoch

# ============================================================
# K-FOLD CROSS VALIDATION
# ============================================================

def run_kfold(all_texts, all_labels, label_encoder, results_dir, timestamp):
    """Validation croisée K-Fold stratifiée"""

    print("\n" + "="*70)
    print(f"VALIDATION CROISEE K-FOLD (K={K_FOLDS}), {FIXED_EPOCHS} EPOQUES")
    print(f"Modèle: {MODEL_NAME}")
    print("="*70)

    # mBERT utilise BertTokenizer (cased)
    tokenizer   = BertTokenizer.from_pretrained(MODEL_NAME)
    num_classes = len(label_encoder.classes_)

    skf    = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(all_texts, all_labels))

    fold_histories = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_idx+1}/{K_FOLDS} ---")

        train_texts_fold  = [all_texts[i] for i in train_idx]
        train_labels_fold = [all_labels[i] for i in train_idx]
        val_texts_fold    = [all_texts[i] for i in val_idx]
        val_labels_fold   = [all_labels[i] for i in val_idx]

        train_dataset = DarijaDataset(train_texts_fold, train_labels_fold, tokenizer)
        val_dataset   = DarijaDataset(val_texts_fold,   val_labels_fold,   tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

        model, optimizer, scheduler = build_model_and_optimizer(
            num_classes, FIXED_EPOCHS, len(train_loader)
        )
        criterion = nn.CrossEntropyLoss()

        history, _ = train_with_history(
            model, optimizer, scheduler, train_loader, val_loader,
            criterion, FIXED_EPOCHS, num_classes
        )

        fold_histories.append(history)

    # Agrégation des métriques sur tous les folds
    metrics_names = [
        'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
        'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc'
    ]

    agg = {'epoch': list(range(1, FIXED_EPOCHS + 1))}

    for metric in metrics_names:
        means = []
        stds  = []
        for e in range(FIXED_EPOCHS):
            values = [fold_histories[p][e].get(metric, float('nan'))
                      for p in range(len(fold_histories))]
            means.append(np.nanmean(values))
            stds.append(np.nanstd(values))
        agg[f'{metric}_mean'] = means
        agg[f'{metric}_std']  = stds

    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(f"{results_dir}/kfold_metriques_{timestamp}.csv", index=False)

    optimal_epoch = int(np.argmin(df_agg['val_loss_mean'])) + 1
    print(f"\nEpoque optimale: {optimal_epoch}")

    auc_at_optimal = [fold_histories[p][optimal_epoch-1].get('val_auc', float('nan'))
                      for p in range(len(fold_histories))]
    mean_auc = np.nanmean(auc_at_optimal)
    print(f"AUC moyenne (K-Fold): {mean_auc:.4f}")

    # Graphiques
    _plot_kfold_metrics(df_agg, len(fold_histories), "",
                        f"{results_dir}/kfold_metriques_complet_{timestamp}.png")
    _plot_kfold_metrics(df_agg.iloc[:optimal_epoch], len(fold_histories),
                        f" (jusqu'à époque {optimal_epoch})",
                        f"{results_dir}/kfold_metriques_optimal_{timestamp}.png")

    return df_agg, mean_auc, optimal_epoch

def _plot_kfold_metrics(df_plot, k_folds, title_suffix, save_path):
    """Tracer les métriques K-Fold (train vs validation)"""

    epochs = df_plot['epoch']

    metrics_config = [
        ('train_loss_mean',      'val_loss_mean',      'Perte',      'Loss'),
        ('train_accuracy_mean',  'val_accuracy_mean',  'Exactitude', 'Accuracy'),
        ('train_precision_mean', 'val_precision_mean', 'Précision',  'Precision'),
        ('train_recall_mean',    'val_recall_mean',    'Rappel',     'Recall'),
        ('train_f1_mean',        'val_f1_mean',        'F1-score',   'F1-score'),

    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (train_col, val_col, ylabel, title_label) in enumerate(metrics_config):
        ax = axes[i]

        if train_col and train_col in df_plot.columns:
            ax.plot(epochs, df_plot[train_col], 'b-o', label='Entraînement', linewidth=2)
        if val_col and val_col in df_plot.columns:
            ax.plot(epochs, df_plot[val_col], 'r-s', label='Validation', linewidth=2)
        elif val_col is None and train_col and train_col in df_plot.columns:
            # AUC only has val
            pass


        ax.set_xlabel('Époque')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_label} (moy sur {k_folds} folds)')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f'Métriques de la validation croisée{title_suffix} — Bert-Base-Multilingual-Cased avec affinage complet (Full Fine-tuning)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Graphique sauvegardé: {save_path}")

# ============================================================
# ENTRAINEMENT FINAL
# ============================================================

def train_final_model(train_texts, train_labels, label_encoder, n_epochs):
    """Entraîner le modèle final sur toutes les données K-Fold"""

    print("\n" + "="*70)
    print(f"ENTRAINEMENT FINAL — {n_epochs} EPOQUES")
    print(f"Modèle: {MODEL_NAME}")
    print(f"Sur {len(train_texts)} échantillons")
    print("="*70)

    set_seed(SEED)

    tokenizer   = BertTokenizer.from_pretrained(MODEL_NAME)
    num_classes = len(label_encoder.classes_)

    dataset      = DarijaDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model, optimizer, scheduler = build_model_and_optimizer(
        num_classes, n_epochs, len(train_loader)
    )

    criterion  = nn.CrossEntropyLoss()
    history    = []
    start_time = time.time()

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        _, train_acc, train_prec, train_rec, train_f1, _, _, _ = evaluate(
            model, train_loader, criterion, num_classes
        )

        history.append({
            'epoch':           epoch + 1,
            'train_loss':      train_loss,
            'train_accuracy':  train_acc,
            'train_precision': train_prec,
            'train_recall':    train_rec,
            'train_f1':        train_f1,
        })

        print(f"Epoch {epoch+1}/{n_epochs} — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

    training_time = time.time() - start_time
    print(f"\nEntraînement terminé en {training_time:.2f}s ({training_time/60:.2f} min)")

    return model, tokenizer, history, training_time

# ============================================================
# GRAPHIQUES
# ============================================================

def plot_final_metrics(history, save_path):
    """Tracer les métriques d'entraînement final"""

    if not history:
        print("Pas de données à tracer")
        return None

    epochs     = [e['epoch']           for e in history]
    train_loss = [e['train_loss']      for e in history]
    train_acc  = [e['train_accuracy']  for e in history]
    train_f1   = [e['train_f1']        for e in history]
    train_prec = [e['train_precision'] for e in history]
    train_rec  = [e['train_recall']    for e in history]

    series = [
        (train_loss, 'Perte',      'Perte'),
        (train_acc,  'Exactitude', 'Exactitude'),
        (train_f1,   'F1-score',   'F1-score'),
        (train_prec, 'Précision',  'Précision'),
        (train_rec,  'Rappel',     'Rappel'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (data, ylabel, title) in enumerate(series):
        axes[i].plot(epochs, data, 'b-o', linewidth=2, label='Entraînement')
        axes[i].set_xlabel('Époque')
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f'{title} (entraînement final)')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        axes[i].set_xticks(epochs)

    axes[5].set_visible(False)
    plt.suptitle("Métriques d'entraînement final — Bert-Base-Multilingual-Cased avec affinage complet (Full Fine-tuning)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    pd.DataFrame(history).to_csv(save_path.replace('.png', '_donnees.csv'), index=False)
    print(f"Graphique sauvegardé: {save_path}")

def plot_time_chart(training_time, total_time, optimal_epochs, save_path, save_dir, timestamp):
    """Graphique des temps d'exécution"""

    labels = [f"Entraînement final ({optimal_epochs} époques)", "Temps total"]
    times  = [training_time, total_time]
    colors = ['#2E7D32', '#1565C0']

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, times, width=0.4, color=colors)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{t:.1f}s\n({t/60:.2f} min)', ha='center', fontweight='bold')

    ax.set_ylabel('Secondes')
    ax.set_title("Temps d'exécution — Bert-Base-Multilingual-Cased avec affinage complet (Full Fine-tuning)")
    ax.set_ylim(0, max(times) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    pd.DataFrame({
        'categorie':        labels,
        'temps_secondes':   times,
        'temps_minutes':    [t/60 for t in times],
        'epoques_optimales': [optimal_epochs, optimal_epochs]
    }).to_csv(f"{save_dir}/temps_execution_{timestamp}.csv", index=False)

# ============================================================
# AUC SUR JEU DE TEST
# ============================================================

def compute_and_plot_test_auc(true_labels, probabilities, label_encoder, results_dir, timestamp):
    """Calculer et tracer l'AUC sur le jeu de test"""

    num_classes = len(label_encoder.classes_)
    true        = np.array(true_labels).astype(int)

    print("\n" + "="*70)
    print("AUC SUR LE JEU DE TEST")
    print("="*70)

    if num_classes == 2:
        probs            = np.array(probabilities)
        fpr, tpr, thresholds = roc_curve(true, probs)
        auc_val          = roc_auc_score(true, probs)
        youden_idx       = np.argmax(tpr - fpr)
        best_threshold   = thresholds[youden_idx]

        print(f"AUC: {auc_val:.4f}")
        print(f"Seuil optimal (Youden): {best_threshold:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, 'b-', linewidth=2,
                label=f'Courbe ROC (AUC = {auc_val:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Classifieur aléatoire')
        ax.scatter(fpr[youden_idx], tpr[youden_idx], color='red', s=100,
                   label=f'Seuil optimal = {best_threshold:.3f}')
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbe ROC — Bert-Base-Multilingual-Cased avec affinage complet (Full Fine-tuning)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/roc_test_{timestamp}.png", dpi=300)
        plt.close()

    else:
        prob_matrix = np.array(probabilities)
        auc_val     = roc_auc_score(true, prob_matrix,
                                    multi_class='ovr', average='weighted')
        print(f"AUC pondéré (OvR): {auc_val:.4f}")

        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        fig, ax = plt.subplots(figsize=(9, 8))

        for i, (cls_name, color) in enumerate(zip(label_encoder.classes_, colors)):
            binary_true    = (true == i).astype(int)
            fpr_i, tpr_i, _ = roc_curve(binary_true, prob_matrix[:, i])
            auc_i          = roc_auc_score(binary_true, prob_matrix[:, i])
            ax.plot(fpr_i, tpr_i, color=color,
                    label=f'Classe {cls_name} (AUC={auc_i:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', label='Classifieur aléatoire')
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title(f'Courbes ROC One-vs-Rest — Bert-Base-Multilingual-Cased fine-tuné\nAUC pondéré = {auc_val:.4f}')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/roc_test_ovr_{timestamp}.png", dpi=300)
        plt.close()

    pd.DataFrame([{'auc_test': auc_val, 'num_classes': num_classes}]).to_csv(
        f"{results_dir}/auc_test_{timestamp}.csv", index=False
    )

    return auc_val

# ============================================================
# EVALUATION SUR TEST
# ============================================================

def save_classification_report(true_labels, predictions, label_encoder, output_dir, timestamp):
    """Sauvegarder le rapport de classification"""

    target_names = [str(cls) for cls in label_encoder.classes_]
    class_report    = classification_report(true_labels, predictions,
                                            target_names=target_names, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT:")
    print("="*70)
    print(classification_report(true_labels, predictions, target_names=target_names))

    class_report_df.to_csv(f"{output_dir}/classification_report_{timestamp}.csv", index=True)
    print(f"Rapport sauvegardé")

    return class_report_df

def compute_metrics(true_labels, predictions, is_binary):
    """Calculer les métriques principales"""
    avg = 'binary' if is_binary else 'weighted'
    return {
        'exactitude': accuracy_score(true_labels, predictions),
        'precision':  precision_score(true_labels, predictions, average=avg, zero_division=0),
        'rappel':     recall_score(true_labels, predictions, average=avg, zero_division=0),
        'f1_score':   f1_score(true_labels, predictions, average=avg, zero_division=0)
    }

def plot_confusion_matrix(true_labels, predictions, label_encoder, save_path):
    """Tracer la matrice de confusion"""

    cm          = confusion_matrix(true_labels, predictions)
    class_names = label_encoder.classes_

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 14})
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Réel')
    ax.set_title('Matrice de confusion — Bert-Base-Multilingual-Cased avec affinage complet (Full Fine-tuning)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Matrice de confusion sauvegardée")

def save_misclassifications(test_texts, true_labels, predictions, probabilities,
                             label_encoder, output_dir, timestamp, num_classes):
    """Sauvegarder les erreurs de classification"""

    true = np.array(true_labels).astype(int)
    pred = np.array(predictions).astype(int)

    df_full = pd.DataFrame({
        'texte':        test_texts,
        'vrai_label':   label_encoder.inverse_transform(true),
        'label_predit': label_encoder.inverse_transform(pred),
        'est_correct':  (true == pred)
    })

    if num_classes == 2:
        probs = np.array(probabilities)
        df_full['probabilite_positive'] = probs
        df_full['probabilite_negative'] = 1 - probs

        fp = df_full[(df_full['vrai_label'] == label_encoder.classes_[0]) &
                     (df_full['label_predit'] == label_encoder.classes_[1])]
        fn = df_full[(df_full['vrai_label'] == label_encoder.classes_[1]) &
                     (df_full['label_predit'] == label_encoder.classes_[0])]

        if len(fp) > 0:
            fp.to_csv(f"{output_dir}/faux_positifs_{timestamp}.csv", index=False)
            print(f"Faux positifs: {len(fp)}")
        if len(fn) > 0:
            fn.to_csv(f"{output_dir}/faux_negatifs_{timestamp}.csv", index=False)
            print(f"Faux négatifs: {len(fn)}")
    else:
        errors = df_full[~df_full['est_correct']]
        if len(errors) > 0:
            errors.to_csv(f"{output_dir}/erreurs_classification_{timestamp}.csv", index=False)
            print(f"Erreurs: {len(errors)}")

def save_all_predictions(test_texts, true_labels, predictions, probabilities,
                          label_encoder, output_dir, timestamp, num_classes):
    """Sauvegarder toutes les prédictions avec probabilités"""

    true = np.array(true_labels).astype(int)
    pred = np.array(predictions).astype(int)

    df = pd.DataFrame({
        'texte':        test_texts,
        'vrai_label':   label_encoder.inverse_transform(true),
        'label_predit': label_encoder.inverse_transform(pred),
        'est_correct':  (true == pred)
    })

    if num_classes == 2:
        probs = np.array(probabilities)           # prob classe positive
        df['probabilite_positive'] = probs
        df['probabilite_negative'] = 1 - probs
    else:
        prob_matrix = np.array(probabilities)     # shape (n, num_classes)
        for i, cls_name in enumerate(label_encoder.classes_):
            df[f'probabilite_{cls_name}'] = prob_matrix[:, i]

    df.to_csv(f"{output_dir}/toutes_predictions_{timestamp}.csv", index=False)
    print(f"Toutes les prédictions sauvegardées: {len(df)} lignes")
    return df

# ============================================================
# TEST DE PERMUTATION
# ============================================================

def permutation_test(model, tokenizer, test_texts, test_labels, label_encoder,
                     n_permutations=999, threshold=THRESHOLD):
    """Test statistique par permutation"""

    print("\n" + "="*70)
    print("TEST DE PERMUTATION")
    print("="*70)

    num_classes = len(label_encoder.classes_)
    is_binary   = (num_classes == 2)

    test_dataset = DarijaDataset(test_texts, test_labels, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            outputs        = model(input_ids, attention_mask=attention_mask)
            probs          = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            if is_binary:
                preds = (probs[:, 1] >= threshold).astype(int)
            else:
                preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    obs_acc    = accuracy_score(all_labels, all_preds)
    print(f"Exactitude observée: {obs_acc:.4f}")

    null_dist = []
    for i in range(n_permutations):
        shuffled = np.random.permutation(all_labels)
        null_dist.append(accuracy_score(shuffled, all_preds))
        if (i+1) % 100 == 0:
            print(f"  Permutations: {i+1}/{n_permutations}")

    null_dist = np.array(null_dist)
    p_value   = np.mean(null_dist >= obs_acc)

    if   p_value < 0.001: significance = "*** hautement significatif"
    elif p_value < 0.01:  significance = "** significatif"
    elif p_value < 0.05:  significance = "* significatif"
    else:                 significance = "n.s. non significatif"

    print(f"\np-valeur: {p_value:.4f} — {significance}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_dist, bins=30, alpha=0.7, color='gray',
            edgecolor='black', label='Distribution nulle')
    ax.axvline(obs_acc, color='red', linewidth=2,
               label=f'Observée (p={p_value:.4f})')
    ax.axvline(null_dist.mean(), color='blue', linestyle='--',
               label=f'Moyenne nulle = {null_dist.mean():.3f}')
    ax.set_xlabel('Exactitude')
    ax.set_ylabel('Fréquence')
    ax.set_title('Test de permutation — Bert-Base-Multilingual-Cased avec affinage complet (Full Fine-tuning)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return {
        'exactitude_observee': obs_acc,
        'p_valeur':            p_value,
        'signification':       significance,
        'figure':              fig
    }

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    start_time = time.time()

    print("="*70)
    print("Bert-Base-Multilingual-Cased avec affinage complet (Full Fine-tuning)")
    print("Classification de texte (Darija / Hate Speech)")
    print(f"K-Folds K={K_FOLDS}, {FIXED_EPOCHS} époques")
    print(f"LR={LEARNING_RATE}, Dropout={DROPOUT}, WD={WEIGHT_DECAY}, "
          f"Warmup={int(WARMUP_RATIO*100)}%")
    print("="*70)

    data_dir   = '/content/drive/MyDrive/BigData_11_2026/data'
    output_dir = '/content/drive/MyDrive/BigData_11_2026/Bert-Base-Multilingual-Cased'

    kfold_path = f'{data_dir}/kfoldsdata.csv'
    test_path  = f'{data_dir}/test.csv'

    if not os.path.exists(kfold_path):
        print(f"Erreur: {kfold_path} non trouvé")
        return None, None, None, None, None, None

    os.makedirs(output_dir, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, 'Resultats')
    os.makedirs(results_dir, exist_ok=True)

    X_kfold, y_kfold, X_test, y_test, label_encoder = load_data(
        kfold_path=kfold_path, test_path=test_path
    )
    num_classes = len(label_encoder.classes_)
    is_binary   = (num_classes == 2)

    # Sauvegarder l'encodeur de labels
    vec_dir = os.path.join(output_dir, 'vectorisation')
    os.makedirs(vec_dir, exist_ok=True)
    with open(os.path.join(vec_dir, 'encodeur_labels.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # ── K-Fold ───────────────────────────────────────────────────
    df_agg, mean_auc_cv, optimal_epochs = run_kfold(
        X_kfold, y_kfold, label_encoder, results_dir, timestamp
    )
    print(f"\nAUC moyen K-Fold: {mean_auc_cv:.4f}")
    print(f"Epoques optimales: {optimal_epochs}")

    # ── Entraînement final ────────────────────────────────────────
    final_model, tokenizer, history, training_time = train_final_model(
        X_kfold, y_kfold, label_encoder, optimal_epochs
    )

    # Sauvegarder le modèle final (format HuggingFace)
    model_save_dir = os.path.join(output_dir, f'modele_final_{timestamp}')
    os.makedirs(model_save_dir, exist_ok=True)
    final_model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Modèle sauvegardé: {model_save_dir}")

    if history:
        plot_final_metrics(history, f"{results_dir}/metriques_finales_{timestamp}.png")

    total_time = time.time() - start_time
    plot_time_chart(training_time, total_time, optimal_epochs,
                    f"{results_dir}/temps_{timestamp}.png", results_dir, timestamp)

    # ── Évaluation sur test ───────────────────────────────────────
    if X_test and len(y_test) > 0:
        print("\n" + "="*70)
        print("EVALUATION SUR TEST")
        print("="*70)

        test_dataset = DarijaDataset(X_test, y_test, tokenizer)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        criterion    = nn.CrossEntropyLoss()

        (test_loss, test_acc, test_prec, test_rec, test_f1,
         predictions, true_labels, probabilities) = evaluate(
            final_model, test_loader, criterion, num_classes, threshold=THRESHOLD
        )

        print(f"\nPerformance sur test:")
        print(f"  Exactitude: {test_acc:.4f}")
        print(f"  Précision:  {test_prec:.4f}")
        print(f"  Rappel:     {test_rec:.4f}")
        print(f"  F1-score:   {test_f1:.4f}")

        save_classification_report(true_labels, predictions,
                                   label_encoder, results_dir, timestamp)

        auc_test = compute_and_plot_test_auc(
            true_labels, probabilities, label_encoder, results_dir, timestamp
        )
        print(f"AUC test: {auc_test:.4f}")

        save_misclassifications(X_test, true_labels, predictions, probabilities,
                                label_encoder, results_dir, timestamp, num_classes)

        save_all_predictions(X_test, true_labels, predictions, probabilities,
                             label_encoder, results_dir, timestamp, num_classes)

        metrics = compute_metrics(true_labels, predictions, is_binary)
        pd.DataFrame([metrics]).to_csv(
            f"{results_dir}/metriques_{timestamp}.csv", index=False
        )

        plot_confusion_matrix(true_labels, predictions, label_encoder,
                              f"{results_dir}/confusion_matrix_{timestamp}.png")

        perm_results = permutation_test(
            final_model, tokenizer, X_test, y_test, label_encoder
        )
        perm_results['figure'].savefig(
            f"{results_dir}/permutation_{timestamp}.png", dpi=300
        )
        plt.close()

        runtime_info = {
            'modele':                  MODEL_NAME,
            'temps_entrainement_sec':  training_time,
            'temps_total_sec':         total_time,
            'epoques_optimales':       optimal_epochs,
            'auc_kfold':               float(mean_auc_cv),
            'auc_test':                float(auc_test),
            'lr':                      LEARNING_RATE,
            'dropout':                 DROPOUT,
            'weight_decay':            WEIGHT_DECAY,
            'warmup_ratio':            WARMUP_RATIO,
            'batch_size':              BATCH_SIZE,
            'max_len':                 MAX_LEN,
        }
        pd.DataFrame([runtime_info]).to_csv(
            f"{results_dir}/execution_{timestamp}.csv", index=False
        )

        print("\n" + "="*70)
        print("RESUME FINAL")
        print("="*70)
        print(f"Modèle:          {MODEL_NAME}")
        print(f"Epoques finales: {len(history)}")
        print(f"Exactitude test: {metrics['exactitude']:.4f}")
        print(f"F1-score test:   {metrics['f1_score']:.4f}")
        print(f"AUC test:        {auc_test:.4f}")
        print(f"Temps total:     {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"p-valeur perm.:  {perm_results['p_valeur']:.4f}")
        print("\n" + "="*70)
        print("TERMINE!")
        print("="*70)

        return (final_model, tokenizer, label_encoder, metrics, total_time,
                {'auc_cv': mean_auc_cv, 'auc_test': auc_test})
    else:
        print("\nPas de test. Fin du pipeline.")
        return final_model, tokenizer, label_encoder, None, None, None

# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    try:
        result = main()
        if result[3] is not None:
            _, _, _, metrics, total_time, auc_info = result
            print(f"\nTemps total: {total_time:.2f}s")
            if auc_info:
                print(f"AUC K-Fold: {auc_info['auc_cv']:.4f}")
                print(f"AUC Test:   {auc_info['auc_test']:.4f}")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()