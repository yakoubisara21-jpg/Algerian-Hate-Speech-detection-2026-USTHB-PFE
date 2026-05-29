"""
DziriBERT Évaluation du Modèle PRÉ-ENTRAÎNÉ (sans fine‑tuning)
Pour comparer les performances avant / après fine‑tuning.

- Charge le modèle DziriBERT original (alger-ia/dziribert)
- Aucun entraînement, uniquement une évaluation sur test.csv
- Métriques, matrice de confusion, courbe ROC, prédictions, etc.
- Seuil = 0.5 (identique au script fine‑tuné)
"""

import numpy as np
import pandas as pd
import torch
import time
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
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ── Graine pour reproductibilité ──
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé: {device}")

# ── Constantes (identiques au script fine‑tuné) ──
MODEL_NAME         = "alger-ia/dziribert"
MAX_LEN            = 128
BATCH_SIZE         = 16
THRESHOLD          = 0.5

# ══════════════════════════════════════════════════════
# DATASET PYTORCH (identique)
# ══════════════════════════════════════════════════════

class DziriDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ══════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES DE TEST
# ══════════════════════════════════════════════════════

def load_test_data(test_path='/content/drive/MyDrive/BigData_11_2026/data/test.csv',
                   text_col='text', label_col='label'):
    """Charge uniquement test.csv et encode les labels."""
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Fichier test non trouvé: {test_path}")

    df_test = pd.read_csv(test_path, sep=';')
    print(f"Test chargé: {len(df_test)} échantillons")

    for col in [text_col, label_col]:
        if col not in df_test.columns:
            raise ValueError(f"Colonne '{col}' non trouvée. "
                             f"Colonnes disponibles: {df_test.columns.tolist()}")

    X_test = df_test[text_col].astype(str).tolist()
    y_test = df_test[label_col].tolist()

    # Encodage des labels (fit uniquement sur test)
    label_encoder = LabelEncoder()
    y_test_enc = label_encoder.fit_transform(y_test)

    print("\nDistribution des classes (test):")
    for label, count in pd.Series(y_test).value_counts().items():
        print(f"  {label}: {count} ({count/len(y_test)*100:.1f}%)")

    return X_test, y_test_enc, label_encoder

# ══════════════════════════════════════════════════════
# ÉVALUATION (identique à la fonction evaluate du script original)
# ══════════════════════════════════════════════════════

def evaluate(model, loader, criterion, num_classes, threshold=THRESHOLD):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    is_binary = (num_classes == 2)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)

            if is_binary:
                pos_probs = probs[:, 1].cpu().numpy()
                preds = (pos_probs >= threshold).astype(int)
                all_probs.extend(pos_probs.tolist())
            else:
                prob_np = probs.cpu().numpy()
                preds = np.argmax(prob_np, axis=1)
                all_probs.extend(prob_np.max(axis=1).tolist())

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg = 'binary' if is_binary else 'weighted'
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=avg, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=avg, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=avg, zero_division=0)

    return (avg_loss, accuracy, precision, recall, f1,
            all_preds, all_labels, all_probs)

# ══════════════════════════════════════════════════════
# FONCTIONS DE SAUVEGARDE (inspirées du script original)
# ══════════════════════════════════════════════════════

def save_evaluation_results(texts, true_labels, predictions, probabilities,
                             label_encoder, output_path):
    df = pd.DataFrame({
        'texte': texts,
        'vrai_label': label_encoder.inverse_transform(true_labels),
        'label_predit': label_encoder.inverse_transform(predictions),
        'est_correct': (np.array(true_labels) == np.array(predictions)),
        'probabilite_positive': probabilities,
        'probabilite_negative': 1 - np.array(probabilities)
    })
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Résultats sauvegardés: {output_path}")

def save_confusion_matrix(true_labels, predictions, label_encoder, output_csv, output_png):
    cm = confusion_matrix(true_labels, predictions)
    class_names = label_encoder.classes_
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(output_csv)
    print(f"Matrice de confusion (CSV): {output_csv}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.xlabel('Prédiction', fontsize=12, fontweight='bold')
    plt.ylabel('Réel', fontsize=12, fontweight='bold')
    plt.title('Matrice de confusion - DziriBERT (PRÉ-ENTRAÎNÉ, sans fine-tuning)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matrice de confusion (image): {output_png}")

def save_metrics_report(metrics, output_path):
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    print(f"Rapport de métriques sauvegardé: {output_path}")
    print("\n" + "=" * 70)
    print("MÉTRIQUES D'ÉVALUATION (modèle PRÉ-ENTRAÎNÉ)")
    print("=" * 70)
    print(f"Exactitude:  {metrics['exactitude']:.4f}")
    print(f"Précision:   {metrics['precision']:.4f}")
    print(f"Rappel:      {metrics['rappel']:.4f}")
    print(f"F1-score:    {metrics['f1_score']:.4f}")

def save_misclassifications(texts, true_labels, predictions, probabilities,
                             label_encoder, output_dir, timestamp):
    true = np.array(true_labels).astype(int)
    pred = np.array(predictions).astype(int)
    probs = np.array(probabilities)

    df_full = pd.DataFrame({
        'texte': texts,
        'vrai_label': label_encoder.inverse_transform(true),
        'vrai_label_numerique': true,
        'label_predit': label_encoder.inverse_transform(pred),
        'label_predit_numerique': pred,
        'probabilite_positive': probs,
        'probabilite_negative': 1 - probs,
        'est_correct': (true == pred)
    })

    fp = df_full[(df_full['vrai_label_numerique'] == 0) & (df_full['label_predit_numerique'] == 1)]
    fn = df_full[(df_full['vrai_label_numerique'] == 1) & (df_full['label_predit_numerique'] == 0)]

    if len(fp) > 0:
        fp_path = os.path.join(output_dir, f'faux_positifs_{timestamp}.csv')
        fp.to_csv(fp_path, index=False, encoding='utf-8-sig')
        print(f"Faux positifs: {len(fp)} → {fp_path}")
    if len(fn) > 0:
        fn_path = os.path.join(output_dir, f'faux_negatifs_{timestamp}.csv')
        fn.to_csv(fn_path, index=False, encoding='utf-8-sig')
        print(f"Faux négatifs: {len(fn)} → {fn_path}")

def plot_roc_curve(true_labels, probabilities, save_path):
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc = roc_auc_score(true_labels, probabilities)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Courbe ROC (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7,
             label='Classifieur aléatoire (AUC = 0.5)')
    plt.xlabel('Taux de faux positifs', fontsize=12)
    plt.ylabel('Taux de vrais positifs', fontsize=12)
    plt.title('Courbe ROC - DziriBERT (PRÉ-ENTRAÎNÉ, sans fine-tuning)',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Courbe ROC sauvegardée: {save_path}")
    return auc




def plot_time_chart(eval_time, save_path, save_dir, timestamp):
    labels = ["Évaluation (sans fine-tuning)"]
    times = [eval_time]
    colors = ['#1565C0']

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, times, width=0.4, color=colors)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{t:.1f}s\n({t/60:.2f} min)', ha='center', fontweight='bold')

    ax.set_ylabel('Secondes')
    ax.set_title("Temps d'exécution - DziriBERT pré-entraîné")
    ax.set_ylim(0, max(times) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Graphique de temps sauvegardé: {save_path}")

    pd.DataFrame({
        'categorie': labels,
        'temps_secondes': times,
        'temps_minutes': [t/60 for t in times]
    }).to_csv(f"{save_dir}/temps_execution_{timestamp}.csv", index=False)

# ══════════════════════════════════════════════════════
# MAIN : ÉVALUATION SANS FINE-TUNING
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("DziriBERT - ÉVALUATION SANS FINE-TUNING")
    print("Modèle chargé directement depuis Hugging Face")
    print(f"Seuil de classification: {THRESHOLD}")
    print("=" * 70)

    # Chemins (à adapter si nécessaire)
    data_dir = '/content/drive/MyDrive/BigData_11_2026/data'
    test_path = f'{data_dir}/test.csv'
    output_dir = '/content/drive/MyDrive/BigData_11_2026/DziriBert/Eval_Base'

    if not os.path.exists(test_path):
        print(f"Erreur: fichier test non trouvé -> {test_path}")
        return
    
    

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    start_time = time.time()

    # 1. Chargement des données de test
    X_test, y_test, label_encoder = load_test_data(test_path)
    num_classes = len(label_encoder.classes_)
    is_binary = (num_classes == 2)

    # 2. Chargement du tokenizer et du modèle PRÉ-ENTRAÎNÉ (sans fine-tuning)
    print(f"\nChargement du modèle pré-entraîné: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # Configuration de dropout (identique à celle utilisée en fine-tuning)
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1

    print(f"Modèle chargé avec succès. Classes: {label_encoder.classes_}")

    # 3. Préparation du DataLoader de test
    test_dataset = DziriDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    # 4. Évaluation
    print("\nÉvaluation en cours...")
    (test_loss, test_acc, test_prec, test_rec, test_f1,
     predictions, true_labels, probabilities) = evaluate(
        model, test_loader, criterion, num_classes, threshold=THRESHOLD
    )

    print(f"\nPerte:           {test_loss:.6f}")
    print(f"Exactitude:      {test_acc:.4f}")
    print(f"Précision:       {test_prec:.4f}")
    print(f"Rappel:          {test_rec:.4f}")
    print(f"F1-score:        {test_f1:.4f}")

    # Rapport de classification détaillé
    print("\nRapport de classification détaillé:")
    print(classification_report(
        true_labels, predictions,
        target_names=[str(c) for c in label_encoder.classes_]
    ))

    # 5. Sauvegarde des résultats
    metrics = {
        'exactitude': test_acc,
        'precision': test_prec,
        'rappel': test_rec,
        'f1_score': test_f1
    }
    save_metrics_report(metrics, os.path.join(output_dir, f'rapport_metriques_{timestamp}.csv'))

    save_evaluation_results(
        X_test, true_labels, predictions, probabilities, label_encoder,
        os.path.join(output_dir, f'resultats_evaluation_{timestamp}.csv')
    )

    save_confusion_matrix(
        true_labels, predictions, label_encoder,
        output_csv=os.path.join(output_dir, f'matrice_confusion_{timestamp}.csv'),
        output_png=os.path.join(output_dir, f'matrice_confusion_{timestamp}.png')
    )

    save_misclassifications(
        X_test, true_labels, predictions, probabilities,
        label_encoder, output_dir, timestamp
    )

    # 6. Courbe ROC (si binaire)
    if is_binary:
        auc = plot_roc_curve(
            true_labels, probabilities,
            os.path.join(output_dir, f'roc_curve_{timestamp}.png')
        )
        print(f"AUC: {auc:.4f}")
        pd.DataFrame({'auc': [auc]}).to_csv(
            os.path.join(output_dir, f'auc_{timestamp}.csv'), index=False
        )

    # 7. Sauvegarde des prédictions brutes
    pd.DataFrame({
        'vrai_label': true_labels,
        'probabilite_predite': probabilities,
        'classe_predite': predictions
    }).to_csv(os.path.join(output_dir, f'predictions_brutes_{timestamp}.csv'), index=False)

    total_time = time.time() - start_time
    print(f"\nÉvaluation terminée en {total_time:.2f}s ({total_time/60:.2f} min)")


    plot_time_chart(total_time,
                os.path.join(output_dir, f'temps_{timestamp}.png'),
                output_dir, timestamp)

    # 8. Informations d'exécution
    runtime_info = {
        'modele': MODEL_NAME,
        'seuil': THRESHOLD,
        'max_len': MAX_LEN,
        'batch_size': BATCH_SIZE,
        'nb_echantillons_test': len(X_test),
        'exactitude': test_acc,
        'precision': test_prec,
        'rappel': test_rec,
        ' f1_score': test_f1,
        'timestamp': timestamp,
        'temps_evaluation_sec': total_time,      # ← new
        'temps_evaluation_min': total_time / 60  # ← new
    }
    
    if is_binary:
        runtime_info['auc'] = auc

    pd.DataFrame([runtime_info]).to_csv(
        os.path.join(output_dir, f'info_evaluation_{timestamp}.csv'), index=False
    )

    print("\n" + "=" * 70)
    print("ÉVALUATION DU MODÈLE PRÉ-ENTRAÎNÉ TERMINÉE")
    print(f"Tous les résultats sont dans : {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    # Optionnel : changer le répertoire de travail
    # os.chdir('/content/drive/MyDrive/BigData_11_2026')
    main()