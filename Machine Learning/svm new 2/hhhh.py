import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ============================================
# CHARGEMENT DES DONNEES
# ============================================
train_df = pd.read_csv('data/train.csv', sep=';')        # 20,000 lignes
val_df = pd.read_csv('data/validation.csv', sep=';')    # 10,000 lignes
test_df = pd.read_csv('data/test.csv', sep=';')         # 10,000 lignes

print("="*70)
print("STRATÉGIE DE VALIDATION AVEC VOS 3 FICHIERS CSV")
print("="*70)
print(f"Train set: {len(train_df)} lignes (fichier original)")
print(f"Validation set: {len(val_df)} lignes (fichier original)")
print(f"Test set: {len(test_df)} lignes (fichier original)")
print("\n📊 Distribution des labels:")
print(f"Train - 0: {sum(train_df['label']==0)}, 1: {sum(train_df['label']==1)}")
print(f"Val   - 0: {sum(val_df['label']==0)}, 1: {sum(val_df['label']==1)}")
print(f"Test  - 0: {sum(test_df['label']==0)}, 1: {sum(test_df['label']==1)}")

# ============================================
# PARAMETRES FIXES (pour éviter overfitting)
# ============================================
print("\n" + "="*70)
print("PARAMÈTRES POUR ÉVITER L'OVERFITTING")
print("="*70)
print("✓ TF-IDF: max_features=5000 (réduit dimensionnalité)")
print("✓ TF-IDF: min_df=5 (ignore mots rares = bruit)")
print("✓ TF-IDF: ngram_range=(1,2) (contexte des mots)")
print("✓ SVM: kernel='linear' (frontière simple)")
print("✓ SVM: C=1.0 (régularisation équilibrée)")

tfidf = TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2))

# ============================================
# TRIAL 1: Entraînement sur TRAIN original, Validation sur VAL original
# ============================================
print("\n" + "="*70)
print("TRIAL 1: Entraînement sur TRAIN original (20,000) → Validation sur VAL original (10,000)")
print("="*70)

# Transformation des données
X_train_trial1 = tfidf.fit_transform(train_df['text'])
X_val_trial1 = tfidf.transform(val_df['text'])
X_test_final = tfidf.transform(test_df['text'])

y_train_trial1 = train_df['label']
y_val_trial1 = val_df['label']
y_test = test_df['label']

# Entraînement
svm_trial1 = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_trial1.fit(X_train_trial1, y_train_trial1)

# Prédictions
y_train_pred_trial1 = svm_trial1.predict(X_train_trial1)
y_val_pred_trial1 = svm_trial1.predict(X_val_trial1)

# Métriques Trial 1
tr_acc1 = accuracy_score(y_train_trial1, y_train_pred_trial1)
tr_prec1 = precision_score(y_train_trial1, y_train_pred_trial1)
tr_rec1 = recall_score(y_train_trial1, y_train_pred_trial1)
tr_f11 = f1_score(y_train_trial1, y_train_pred_trial1)
tr_loss1 = np.mean(np.maximum(0, 1 - y_train_trial1 * svm_trial1.decision_function(X_train_trial1)))

val_acc1 = accuracy_score(y_val_trial1, y_val_pred_trial1)
val_prec1 = precision_score(y_val_trial1, y_val_pred_trial1)
val_rec1 = recall_score(y_val_trial1, y_val_pred_trial1)
val_f11 = f1_score(y_val_trial1, y_val_pred_trial1)
val_loss1 = np.mean(np.maximum(0, 1 - y_val_trial1 * svm_trial1.decision_function(X_val_trial1)))

print(f"Train - Acc: {tr_acc1:.4f}, F1: {tr_f11:.4f}, Loss: {tr_loss1:.4f}")
print(f"Val   - Acc: {val_acc1:.4f}, F1: {val_f11:.4f}, Loss: {val_loss1:.4f}")
print(f"Écart Train-Val: {abs(tr_acc1 - val_acc1):.4f}")

# ============================================
# FUSION TRAIN + VALIDATION pour les trials suivants
# ============================================
print("\n" + "="*70)
print("CRÉATION DU DATASET FUSIONNÉ (TRAIN + VALIDATION)")
print("="*70)

# Fusionner train et validation pour avoir plus de données
merged_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Dataset fusionné: {len(merged_df)} lignes (Train {len(train_df)} + Val {len(val_df)})")

# Transformer TOUTES les données fusionnées avec le même TF-IDF (appris sur trial1)
X_merged = tfidf.transform(merged_df['text'])
y_merged = merged_df['label'].values

print(f"Distribution fusionnée - 0: {sum(y_merged==0)}, 1: {sum(y_merged==1)}")

# ============================================
# TRIALS 2 à 5: Différents pourcentages du dataset fusionné
# ============================================
print("\n" + "="*70)
print("TRIALS 2 à 5: Différents pourcentages du dataset fusionné")
print("="*70)

# Pourcentages à tester (du dataset fusionné)
pourcentages = [0.9, 0.8, 0.7, 0.6]  # 90%, 80%, 70%, 60% du merged dataset
trials_metrics = []

# Ajouter Trial 1 aux métriques
trials_metrics.append({
    'trial': 1,
    'train_size': len(train_df),
    'train_pct': 100,
    'tr_acc': tr_acc1, 'val_acc': val_acc1,
    'tr_f1': tr_f11, 'val_f1': val_f11,
    'tr_prec': tr_prec1, 'val_prec': val_prec1,
    'tr_rec': tr_rec1, 'val_rec': val_rec1,
    'tr_loss': tr_loss1, 'val_loss': val_loss1
})

np.random.seed(42)
total_merged = X_merged.shape[0]  # CORRECTION: utiliser shape[0] au lieu de len()

for i, pct in enumerate(pourcentages, start=2):
    print(f"\n--- TRIAL {i}: {int(pct*100)}% du dataset fusionné ({int(total_merged*pct)} lignes) ---")
    
    # Échantillonnage aléatoire du dataset fusionné
    sample_size = int(total_merged * pct)
    sampled_indices = np.random.choice(total_merged, sample_size, replace=False)
    
    X_train_subset = X_merged[sampled_indices]
    y_train_subset = y_merged[sampled_indices]
    
    # Le reste du merged dataset sert de validation pour ce trial
    val_indices = [idx for idx in range(total_merged) if idx not in sampled_indices]
    X_val_subset = X_merged[val_indices]
    y_val_subset = y_merged[val_indices]
    
    print(f"  Train: {X_train_subset.shape[0]} lignes")  # CORRECTION: utiliser shape[0]
    print(f"  Val:   {X_val_subset.shape[0]} lignes")    # CORRECTION: utiliser shape[0]
    
    # Entraînement
    svm_subset = SVC(kernel='linear', C=1.0, random_state=42)
    svm_subset.fit(X_train_subset, y_train_subset)
    
    # Prédictions
    y_train_pred = svm_subset.predict(X_train_subset)
    y_val_pred = svm_subset.predict(X_val_subset)
    
    # Métriques
    tr_acc = accuracy_score(y_train_subset, y_train_pred)
    tr_prec = precision_score(y_train_subset, y_train_pred)
    tr_rec = recall_score(y_train_subset, y_train_pred)
    tr_f1 = f1_score(y_train_subset, y_train_pred)
    tr_loss = np.mean(np.maximum(0, 1 - y_train_subset * svm_subset.decision_function(X_train_subset)))
    
    val_acc = accuracy_score(y_val_subset, y_val_pred)
    val_prec = precision_score(y_val_subset, y_val_pred)
    val_rec = recall_score(y_val_subset, y_val_pred)
    val_f1 = f1_score(y_val_subset, y_val_pred)
    val_loss = np.mean(np.maximum(0, 1 - y_val_subset * svm_subset.decision_function(X_val_subset)))
    
    print(f"  Train - Acc: {tr_acc:.4f}, F1: {tr_f1:.4f}, Loss: {tr_loss:.4f}")
    print(f"  Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Loss: {val_loss:.4f}")
    print(f"  Écart Train-Val: {abs(tr_acc - val_acc):.4f}")
    
    trials_metrics.append({
        'trial': i,
        'train_size': X_train_subset.shape[0],  # CORRECTION: utiliser shape[0]
        'train_pct': int(pct*100),
        'tr_acc': tr_acc, 'val_acc': val_acc,
        'tr_f1': tr_f1, 'val_f1': val_f1,
        'tr_prec': tr_prec, 'val_prec': val_prec,
        'tr_rec': tr_rec, 'val_rec': val_rec,
        'tr_loss': tr_loss, 'val_loss': val_loss
    })

# ============================================
# SAUVEGARDE DES MÉTRIQUES
# ============================================
trials_df = pd.DataFrame(trials_metrics)
trials_df.to_csv('svm_trials_metrics.csv', index=False)
print("\n✅ Trials metrics saved to 'svm_trials_metrics.csv'")
print("\n📊 RÉSUMÉ DES 5 TRIALS:")
print(trials_df[['trial', 'train_size', 'tr_acc', 'val_acc', 'tr_loss', 'val_loss']].to_string())

# ============================================
# MODÈLE FINAL: Entraînement sur TOUT (TRAIN + VALIDATION)
# ============================================
print("\n" + "="*70)
print("ENTRAÎNEMENT DU MODÈLE FINAL SUR TOUT LE DATASET DISPONIBLE")
print("="*70)

# Ré-entraîner sur tout le merged dataset
X_full = X_merged
y_full = y_merged

print(f"Entraînement final sur {X_full.shape[0]} lignes (Train + Val fusionnés)")

start_time = time.time()
svm_final = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_final.fit(X_full, y_full)
train_time = time.time() - start_time

# ============================================
# ÉVALUATION FINALE SUR TEST SET
# ============================================
print("\n" + "="*70)
print("ÉVALUATION FINALE SUR LE TEST SET (10,000 lignes non vues)")
print("="*70)

y_pred_test = svm_final.predict(X_test_final)
y_prob_test = svm_final.predict_proba(X_test_final)

# Métriques finales
test_acc = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test)
test_rec = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_prob_test[:, 1])

print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall:    {test_rec:.4f}")
print(f"Test F1-Score:  {test_f1:.4f}")
print(f"Test ROC-AUC:   {roc_auc:.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# ============================================
# SAUVEGARDE DES RÉSULTATS DÉTAILLÉS
# ============================================
test_results = pd.DataFrame({
    'text': test_df['text'],
    'true_label': y_test,
    'predicted_label': y_pred_test,
    'is_correct': y_test == y_pred_test,
    'probability_non_hate': y_prob_test[:, 0],
    'probability_hate': y_prob_test[:, 1]
})
test_results.to_csv('svm_test_evaluation.csv', index=False)
print("\n✅ Test evaluation saved to 'svm_test_evaluation.csv'")

# ============================================
# PLOTTING - 5 GRAPHIQUES
# ============================================
print("\n" + "="*70)
print("GÉNÉRATION DES GRAPHIQUES")
print("="*70)

# 1. Learning curves (5 trials)
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
metrics_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall', 'Loss']
keys = [('tr_acc', 'val_acc'), ('tr_f1', 'val_f1'), ('tr_prec', 'val_prec'), 
        ('tr_rec', 'val_rec'), ('tr_loss', 'val_loss')]

for i, (m_name, (tr_k, val_k)) in enumerate(zip(metrics_names, keys)):
    axes[i].plot(trials_df['trial'], trials_df[tr_k], marker='o', label='Train', linewidth=2, markersize=8, color='blue')
    axes[i].plot(trials_df['trial'], trials_df[val_k], marker='s', label='Validation', linewidth=2, markersize=8, color='orange')
    axes[i].set_title(f'{m_name} per Trial', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Trial Number', fontsize=10)
    axes[i].set_ylabel(m_name, fontsize=10)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xticks(trials_df['trial'])

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300)
print("✅ Learning curves saved to 'learning_curves.png'")

# 2. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Confusion Matrix (Test Set - 10,000 unseen samples)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("✅ Confusion matrix saved to 'confusion_matrix.png'")

# 3. ROC-AUC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob_test[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_auc_curve.png', dpi=300)
print("✅ ROC-AUC curve saved to 'roc_auc_curve.png'")

# 4. Évolution de l'écart Train-Val (indicateur d'overfitting)
plt.figure(figsize=(10, 6))
trials_df['overfitting_gap'] = abs(trials_df['tr_acc'] - trials_df['val_acc'])
plt.bar(trials_df['trial'], trials_df['overfitting_gap'], color='steelblue', alpha=0.7)
plt.axhline(y=trials_df['overfitting_gap'].mean(), color='red', linestyle='--', label=f'Moyenne: {trials_df["overfitting_gap"].mean():.4f}')
plt.title("Écart Train-Validation (Indicateur d'Overfitting)", fontsize=14, fontweight='bold')
plt.xlabel("Trial", fontsize=12)
plt.ylabel("Accuracy Gap |Train - Val|", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('overfitting_gap.png', dpi=300)
print("✅ Overfitting gap chart saved to 'overfitting_gap.png'")

# ============================================
# SAUVEGARDE DU MODÈLE
# ============================================
model_artifacts = {
    'svm_model': svm_final,
    'tfidf_vectorizer': tfidf,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'test_roc_auc': roc_auc,
    'trials_metrics': trials_metrics
}

with open('best_model.pth', 'wb') as f:
    pickle.dump(model_artifacts, f)
print("\n✅ Best model saved to 'best_model.pth'")

# ============================================
# RÉSUMÉ FINAL
# ============================================
print("\n" + "="*70)
print("RÉSUMÉ FINAL")
print("="*70)
print(f"⏱️  Temps d'entraînement final: {train_time:.2f} secondes")
print(f"\n📊 Performance sur le Test Set (10,000 textes non vus):")
print(f"   ┌─────────────────────────────────────┐")
print(f"   │ Accuracy:  {test_acc:.4f}  ({test_acc*100:.2f}%)         │")
print(f"   │ F1-Score:  {test_f1:.4f}                     │")
print(f"   │ Precision: {test_prec:.4f}                     │")
print(f"   │ Recall:    {test_rec:.4f}                     │")
print(f"   │ ROC-AUC:   {roc_auc:.4f}                     │")
print(f"   └─────────────────────────────────────┘")

print(f"\n📁 Fichiers générés:")
print(f"   - svm_trials_metrics.csv  (métriques des 5 trials)")
print(f"   - svm_test_evaluation.csv (prédictions détaillées)")
print(f"   - learning_curves.png     (courbes d'apprentissage)")
print(f"   - confusion_matrix.png    (matrice de confusion)")
print(f"   - roc_auc_curve.png       (courbe ROC-AUC)")
print(f"   - overfitting_gap.png     (écart Train-Val)")
print(f"   - best_model.pth          (modèle sauvegardé)")

# Petit conseil pour charger le modèle plus tard
print("\n💡 Pour recharger le modèle plus tard:")
print("   import pickle")
print("   with open('best_model.pth', 'rb') as f:")
print("       model = pickle.load(f)")
print("   svm = model['svm_model']")
print("   tfidf = model['tfidf_vectorizer']")