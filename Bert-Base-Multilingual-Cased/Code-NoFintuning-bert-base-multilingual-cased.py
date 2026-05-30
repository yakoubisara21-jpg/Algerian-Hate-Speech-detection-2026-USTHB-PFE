import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score,
                              roc_curve, roc_auc_score, classification_report)
from torch.utils.data import DataLoader, Dataset
import warnings
import os
import pickle
import random
import time
from datetime import datetime
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
# PARAMÈTRES — EVALUATION ZERO-SHOT (sans fine-tuning)
# ============================================================
MODEL_NAME  = "bert-base-multilingual-cased"   # mBERT brut, poids pré-entraînés
MAX_LEN     = 128
BATCH_SIZE  = 16
THRESHOLD   = 0.5   # Seuil de décision binaire

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

def load_test_data(test_path, text_col='text', label_col='label'):
    """
    Charger uniquement le jeu de test pour l'évaluation zero-shot.
    Retourne les textes, les labels encodés et le LabelEncoder.
    """
    print(f"\nChargement du jeu de test depuis: {test_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Fichier non trouvé: {test_path}")

    df = pd.read_csv(test_path, sep=';')
    print(f"Chargé: {len(df)} échantillons")

    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' non trouvée dans {test_path}")

    X = df[text_col].astype(str).tolist()
    y_raw = df[label_col].tolist()

    print("\nDistribution des classes:")
    for label, count in pd.Series(y_raw).value_counts().items():
        print(f"  {label}: {count} ({count/len(y_raw)*100:.1f}%)")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_raw)
    y_enc = label_encoder.transform(y_raw)

    print(f"\nClasses détectées: {list(label_encoder.classes_)}")
    return X, y_enc, label_encoder

# ============================================================
# CHARGEMENT DU MODELE ZERO-SHOT
# ============================================================

def load_zero_shot_model(num_classes):
    """
    Charger mBERT SANS fine-tuning avec une tête de classification
    initialisée aléatoirement.

    IMPORTANT:
    - Les couches BERT (encoder, embeddings) ont leurs poids pré-entraînés.
    - La couche de classification (classifier) est initialisée aléatoirement
      car num_labels est spécifié mais aucun fine-tuning n'est fait.
    - C'est une évaluation zero-shot : le modèle n'a jamais vu la tâche.
    - ignore_mismatched_sizes=True est nécessaire car la tête de classification
      du checkpoint pré-entraîné (si elle existe) peut avoir une dimension
      différente du nombre de classes cible.
    """
    print(f"\nChargement de {MODEL_NAME} (ZERO-SHOT — SANS fine-tuning)")
    print("Les poids pré-entraînés sont chargés, la tête de classification est aléatoire.")

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # Tête de classification réinitialisée
    ).to(device)

    # On ne modifie RIEN — évaluation directe sans aucun entraînement
    model.eval()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux:     {total_params:,}")
    print(f"Paramètres entraîn.:   {trainable_params:,}")
    print("Mode: eval() — aucun entraînement, aucune mise à jour des poids")

    return model

# ============================================================
# EVALUATION ZERO-SHOT
# ============================================================

def evaluate_zero_shot(model, loader, num_classes, threshold=THRESHOLD):
    """
    Évaluer le modèle sans calcul de perte (pas de critère nécessaire
    puisqu'on ne s'entraîne pas).
    Retourne les prédictions, labels réels et probabilités.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    is_binary = (num_classes == 2)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            if is_binary:
                pos_probs = probs[:, 1]
                preds     = (pos_probs >= threshold).astype(int)
                all_probs.extend(pos_probs.tolist())
            else:
                preds = np.argmax(probs, axis=1)
                all_probs.extend(probs.tolist())

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return all_preds, all_labels, all_probs

# ============================================================
# CALCUL DES METRIQUES
# ============================================================

def compute_metrics(true_labels, predictions, is_binary):
    """Calculer les métriques principales"""
    avg = 'binary' if is_binary else 'weighted'
    return {
        'exactitude': accuracy_score(true_labels, predictions),
        'precision':  precision_score(true_labels, predictions, average=avg, zero_division=0),
        'rappel':     recall_score(true_labels, predictions, average=avg, zero_division=0),
        'f1_score':   f1_score(true_labels, predictions, average=avg, zero_division=0)
    }

def compute_auc(true_labels, probs, num_classes):
    """Calculer l'AUC (ROC)"""
    true = np.array(true_labels)
    try:
        if num_classes == 2:
            return roc_auc_score(true, np.array(probs))
        else:
            prob_matrix = np.array(probs)
            return roc_auc_score(true, prob_matrix, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"  [Avertissement AUC]: {e}")
        return float('nan')

# ============================================================
# GRAPHIQUES
# ============================================================

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
    ax.set_title('Matrice de confusion — Bert-Base-Multilingual-Cased Zero-Shot (sans fine-tuning)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Matrice de confusion sauvegardée: {save_path}")


def plot_roc_curve(true_labels, probs, label_encoder, results_dir, timestamp):
    """Tracer la courbe ROC"""
    num_classes = len(label_encoder.classes_)
    true        = np.array(true_labels).astype(int)

    if num_classes == 2:
        probs_arr        = np.array(probs)
        fpr, tpr, thrs   = roc_curve(true, probs_arr)
        auc_val          = roc_auc_score(true, probs_arr)
        youden_idx       = np.argmax(tpr - fpr)
        best_thr         = thrs[youden_idx]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, 'b-', linewidth=2,
                label=f'Courbe ROC (AUC = {auc_val:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Classifieur aléatoire')
        ax.scatter(fpr[youden_idx], tpr[youden_idx], color='red', s=100,
                   label=f'Seuil optimal = {best_thr:.3f}')
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title('Courbe ROC — Bert-Base-Multilingual-Cased Zero-Shot (sans fine-tuning)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/roc_zero_shot_{timestamp}.png", dpi=300)
        plt.close()
        print(f"Courbe ROC sauvegardée")
        return auc_val

    else:
        prob_matrix = np.array(probs)
        auc_val     = roc_auc_score(true, prob_matrix,
                                    multi_class='ovr', average='weighted')
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        fig, ax = plt.subplots(figsize=(9, 8))

        for i, (cls_name, color) in enumerate(zip(label_encoder.classes_, colors)):
            binary_true     = (true == i).astype(int)
            fpr_i, tpr_i, _ = roc_curve(binary_true, prob_matrix[:, i])
            auc_i           = roc_auc_score(binary_true, prob_matrix[:, i])
            ax.plot(fpr_i, tpr_i, color=color,
                    label=f'Classe {cls_name} (AUC={auc_i:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', label='Classifieur aléatoire')
        ax.set_xlabel('Taux de faux positifs')
        ax.set_ylabel('Taux de vrais positifs')
        ax.set_title(f'Courbes ROC OvR — Bert-Base-Multilingual-Cased Zero-Shot (sans fine-tuning)\nAUC pondéré = {auc_val:.4f}')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/roc_zero_shot_ovr_{timestamp}.png", dpi=300)
        plt.close()
        print(f"Courbes ROC OvR sauvegardées")
        return auc_val


def plot_class_distribution(true_labels, predictions, label_encoder, results_dir, timestamp):
    """Comparer la distribution réelle vs prédite"""
    classes      = label_encoder.classes_
    true_counts  = [np.sum(np.array(true_labels) == i) for i in range(len(classes))]
    pred_counts  = [np.sum(np.array(predictions) == i) for i in range(len(classes))]

    x      = np.arange(len(classes))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, true_counts, width, label='Réel',    color='steelblue')
    ax.bar(x + width/2, pred_counts, width, label='Prédit',  color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel('Classe')
    ax.set_ylabel('Nombre d\'échantillons')
    ax.set_title('Distribution des classes: Réel vs Prédit — Bert-Base-Multilingual-Cased Zero-Shot (sans fine-tuning)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = f"{results_dir}/distribution_classes_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Distribution sauvegardée: {save_path}")


def plot_metrics_bar(metrics, auc_val, results_dir, timestamp):
    """Graphique résumé des métriques"""
    metric_names  = ['Exactitude', 'Précision', 'Rappel', 'F1-score', 'AUC']
    metric_values = [
        metrics['exactitude'],
        metrics['precision'],
        metrics['rappel'],
        metrics['f1_score'],
        auc_val if not np.isnan(auc_val) else 0.0
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(metric_names, metric_values, color=colors, width=0.5)

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Métriques d\'évaluation — Bert-Base-Multilingual-Cased Zero-Shot (sans fine-tuning)')
    ax.axhline(y=1/len(metrics), color='gray', linestyle='--', alpha=0.5,
               label='Baseline aléatoire')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    save_path = f"{results_dir}/metriques_resume_{timestamp}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Résumé métriques sauvegardé: {save_path}")


# ============================================================
# RAPPORT DE CLASSIFICATION
# ============================================================

def save_classification_report(true_labels, predictions, label_encoder,
                                output_dir, timestamp):
    """Sauvegarder et afficher le rapport de classification"""
    target_names    = [str(cls) for cls in label_encoder.classes_]
    report_dict     = classification_report(true_labels, predictions,
                                            target_names=target_names,
                                            output_dict=True)
    report_df       = pd.DataFrame(report_dict).transpose()

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT (Zero-Shot):")
    print("="*70)
    print(classification_report(true_labels, predictions, target_names=target_names))

    report_df.to_csv(f"{output_dir}/classification_report_zero_shot_{timestamp}.csv",
                     index=True)
    print("Rapport sauvegardé")
    return report_df


# ============================================================
# ERREURS DE CLASSIFICATION
# ============================================================

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
        df_full['probabilite_positive'] = np.array(probabilities)
        fp = df_full[(df_full['vrai_label'] == label_encoder.classes_[0]) &
                     (df_full['label_predit'] == label_encoder.classes_[1])]
        fn = df_full[(df_full['vrai_label'] == label_encoder.classes_[1]) &
                     (df_full['label_predit'] == label_encoder.classes_[0])]
        if len(fp) > 0:
            fp.to_csv(f"{output_dir}/faux_positifs_zero_shot_{timestamp}.csv", index=False)
        if len(fn) > 0:
            fn.to_csv(f"{output_dir}/faux_negatifs_zero_shot_{timestamp}.csv", index=False)
        print(f"Faux positifs: {len(fp)} | Faux négatifs: {len(fn)}")
    else:
        errors = df_full[~df_full['est_correct']]
        if len(errors) > 0:
            errors.to_csv(f"{output_dir}/erreurs_zero_shot_{timestamp}.csv", index=False)
        print(f"Erreurs: {len(errors)} / {len(df_full)}")


# ============================================================
# TEST DE PERMUTATION
# ============================================================

def permutation_test(predictions, true_labels, n_permutations=999):
    """
    Test statistique par permutation.
    Pas besoin du modèle — on réutilise les prédictions déjà calculées.
    """
    print("\n" + "="*70)
    print("TEST DE PERMUTATION (Zero-Shot)")
    print("="*70)

    all_labels = np.array(true_labels)
    all_preds  = np.array(predictions)
    obs_acc    = accuracy_score(all_labels, all_preds)
    print(f"Exactitude observée: {obs_acc:.4f}")

    null_dist = []
    for i in range(n_permutations):
        shuffled = np.random.permutation(all_labels)
        null_dist.append(accuracy_score(shuffled, all_preds))
        if (i+1) % 200 == 0:
            print(f"  Permutations: {i+1}/{n_permutations}")

    null_dist = np.array(null_dist)
    p_value   = np.mean(null_dist >= obs_acc)

    if   p_value < 0.001: significance = "*** hautement significatif"
    elif p_value < 0.01:  significance = "** significatif"
    elif p_value < 0.05:  significance = "* significatif"
    else:                 significance = "n.s. non significatif"

    print(f"p-valeur: {p_value:.4f} — {significance}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_dist, bins=30, alpha=0.7, color='gray',
            edgecolor='black', label='Distribution nulle (permutations)')
    ax.axvline(obs_acc, color='red', linewidth=2,
               label=f'Observée = {obs_acc:.4f} (p={p_value:.4f})')
    ax.axvline(null_dist.mean(), color='blue', linestyle='--',
               label=f'Moyenne nulle = {null_dist.mean():.3f}')
    ax.set_xlabel('Exactitude')
    ax.set_ylabel('Fréquence')
    ax.set_title('Test de permutation — mBERT Zero-Shot')
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
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70)
    print("Bert-Base-Multilingual-Cased Zero-Shot — Évaluation SANS fine-tuning")
    print("Classification de texte (Darija / Hate Speech)")
    print(f"Modèle: {MODEL_NAME}  |  MAX_LEN={MAX_LEN}  |  BATCH={BATCH_SIZE}")
    print("="*70)

    # ── Chemins ──────────────────────────────────────────────────
    data_dir   = '/content/drive/MyDrive/BigData_11_2026/data'
    output_dir = '/content/drive/MyDrive/BigData_11_2026/Bert-Base-Multilingual-Cased'
    test_path  = f'{data_dir}/test.csv'

    if not os.path.exists(test_path):
        print(f"Erreur: {test_path} non trouvé.")
        return

    results_dir = os.path.join(output_dir, 'Resultats_ZeroShot')
    os.makedirs(results_dir, exist_ok=True)

    # ── Chargement des données ────────────────────────────────────
    X_test, y_test, label_encoder = load_test_data(test_path)
    num_classes = len(label_encoder.classes_)
    is_binary   = (num_classes == 2)

    # Sauvegarder l'encodeur de labels
    vec_dir = os.path.join(output_dir, 'vectorisation')
    os.makedirs(vec_dir, exist_ok=True)
    with open(os.path.join(vec_dir, 'encodeur_labels_zero_shot.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    # ── Chargement tokenizer et modèle zero-shot ─────────────────
    print(f"\nChargement du tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = load_zero_shot_model(num_classes)

    # ── Création du DataLoader ────────────────────────────────────
    test_dataset = DarijaDataset(X_test, y_test, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"\nDataLoader créé: {len(test_dataset)} échantillons, "
          f"{len(test_loader)} batches")

    # ── Évaluation zero-shot ──────────────────────────────────────
    print("\n" + "="*70)
    print("EVALUATION ZERO-SHOT EN COURS...")
    print("="*70)

    eval_start  = time.time()
    predictions, true_labels, probabilities = evaluate_zero_shot(
        model, test_loader, num_classes, threshold=THRESHOLD
    )
    eval_time   = time.time() - eval_start
    print(f"Évaluation terminée en {eval_time:.2f}s")

    # ── Métriques ─────────────────────────────────────────────────
    metrics = compute_metrics(true_labels, predictions, is_binary)
    auc_val = compute_auc(true_labels, probabilities, num_classes)

    print("\n" + "="*70)
    print("RESULTATS — mBERT Zero-Shot (sans fine-tuning)")
    print("="*70)
    print(f"  Exactitude: {metrics['exactitude']:.4f}")
    print(f"  Précision:  {metrics['precision']:.4f}")
    print(f"  Rappel:     {metrics['rappel']:.4f}")
    print(f"  F1-score:   {metrics['f1_score']:.4f}")
    print(f"  AUC:        {auc_val:.4f}")

    # Baseline aléatoire théorique
    random_baseline = 1.0 / num_classes
    print(f"\n  Baseline aléatoire théorique: {random_baseline:.4f}")
    print(f"  Gain vs baseline: {metrics['exactitude'] - random_baseline:+.4f}")

    # ── Rapport de classification ─────────────────────────────────
    save_classification_report(true_labels, predictions, label_encoder,
                                results_dir, timestamp)

    # ── Graphiques ────────────────────────────────────────────────
    plot_confusion_matrix(true_labels, predictions, label_encoder,
                          f"{results_dir}/confusion_matrix_zero_shot_{timestamp}.png")

    plot_roc_curve(true_labels, probabilities, label_encoder, results_dir, timestamp)

    plot_class_distribution(true_labels, predictions, label_encoder,
                             results_dir, timestamp)

    plot_metrics_bar(metrics, auc_val, results_dir, timestamp)

    # ── Erreurs ───────────────────────────────────────────────────
    save_misclassifications(X_test, true_labels, predictions, probabilities,
                             label_encoder, results_dir, timestamp, num_classes)

    # ── Test de permutation ───────────────────────────────────────
    perm_results = permutation_test(predictions, true_labels, n_permutations=999)
    perm_results['figure'].savefig(
        f"{results_dir}/permutation_zero_shot_{timestamp}.png", dpi=300
    )
    plt.close()

    # ── Sauvegarde résumé ─────────────────────────────────────────
    total_time = time.time() - start_time

    summary = {
        'modele':            MODEL_NAME,
        'mode':              'zero_shot_sans_finetuning',
        'nb_echantillons':   len(X_test),
        'nb_classes':        num_classes,
        'classes':           str(list(label_encoder.classes_)),
        'exactitude':        metrics['exactitude'],
        'precision':         metrics['precision'],
        'rappel':            metrics['rappel'],
        'f1_score':          metrics['f1_score'],
        'auc':               float(auc_val),
        'p_valeur_perm':     perm_results['p_valeur'],
        'signification':     perm_results['signification'],
        'baseline_aleatoire': random_baseline,
        'gain_vs_baseline':  metrics['exactitude'] - random_baseline,
        'temps_eval_sec':    eval_time,
        'temps_total_sec':   total_time,
        'batch_size':        BATCH_SIZE,
        'max_len':           MAX_LEN,
        'threshold':         THRESHOLD,
    }
    pd.DataFrame([summary]).to_csv(
        f"{results_dir}/resume_zero_shot_{timestamp}.csv", index=False
    )

    print("\n" + "="*70)
    print("RESUME FINAL — Zero-Shot")
    print("="*70)
    print(f"Modèle:          {MODEL_NAME} (SANS fine-tuning)")
    print(f"Échantillons:    {len(X_test)}")
    print(f"Exactitude:      {metrics['exactitude']:.4f}")
    print(f"F1-score:        {metrics['f1_score']:.4f}")
    print(f"AUC:             {auc_val:.4f}")
    print(f"p-valeur perm.:  {perm_results['p_valeur']:.4f} ({perm_results['signification']})")
    print(f"Temps total:     {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Résultats dans:  {results_dir}")
    print("\n" + "="*70)
    print("TERMINE!")
    print("="*70)

    return model, tokenizer, label_encoder, metrics, auc_val, perm_results


# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    try:
        result = main()
        if result is not None:
            _, _, _, metrics, auc_val, perm = result
            print(f"\nExactitude finale: {metrics['exactitude']:.4f}")
            print(f"AUC finale:        {auc_val:.4f}")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()