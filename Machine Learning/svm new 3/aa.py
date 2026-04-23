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
print("STRATÉGIE: TRAIN+VAL FUSIONNÉS → TEST FIXE")
print("="*70)

# Fusionner TRAIN + VALIDATION
merged_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Dataset fusionné (Train+Val): {len(merged_df)} lignes")
print(f"  - Train original: {len(train_df)} lignes")
print(f"  - Validation original: {len(val_df)} lignes")
print(f"Test set (fixe): {len(test_df)} lignes (jamais utilisé pendant l'entraînement)")

print("\n📊 Distribution des labels:")
print(f"Fusionné - 0: {sum(merged_df['label']==0)}, 1: {sum(merged_df['label']==1)}")
print(f"Test     - 0: {sum(test_df['label']==0)}, 1: {sum(test_df['label']==1)}")

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

# Créer et entraîner TF-IDF sur TOUT le dataset fusionné (pour que la transformation soit cohérente)
tfidf = TfidfVectorizer(max_features=5000, min_df=5, ngram_range=(1, 2))
X_full = tfidf.fit_transform(merged_df['text'])
y_full = merged_df['label'].values

# Transformer le test set (fixe) une fois pour toutes
X_test_fixed = tfidf.transform(test_df['text'])
y_test_fixed = test_df['label'].values

print(f"\n✅ TF-IDF entraîné sur {X_full.shape[0]} documents")
print(f"   Dimension des features: {X_full.shape[1]}")
print(f"   Test set transformé: {X_test_fixed.shape[0]} documents")

# ============================================
# TRIALS: Différents pourcentages du dataset fusionné
# ============================================
print("\n" + "="*70)
print("TRIALS: Entraînement sur différents % du dataset fusionné")
print("Test set reste FIXE (10,000 textes jamais vus)")
print("="*70)

# Pourcentages à tester du dataset fusionné
pourcentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
trials_metrics = []

np.random.seed(42)
total_samples = X_full.shape[0]

for pct in pourcentages:
    print(f"\n--- TRIAL: {int(pct*100)}% du dataset fusionné ---")
    
    # Échantillonnage aléatoire
    sample_size = int(total_samples * pct)
    sampled_indices = np.random.choice(total_samples, sample_size, replace=False)
    
    X_train_subset = X_full[sampled_indices]
    y_train_subset = y_full[sampled_indices]
    
    print(f"  Entraînement: {X_train_subset.shape[0]} lignes")
    print(f"  Test (fixe):   {X_test_fixed.shape[0]} lignes")
    
    # Entraînement
    svm_subset = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    svm_subset.fit(X_train_subset, y_train_subset)
    
    # Prédictions sur TRAIN
    y_train_pred = svm_subset.predict(X_train_subset)
    y_train_prob = svm_subset.predict_proba(X_train_subset)
    # Pour binaire: y_train_prob[:, 1] est la probabilité de la classe 1 (hate)
    
    # Prédictions sur TEST (fixe)
    y_test_pred = svm_subset.predict(X_test_fixed)
    y_test_prob = svm_subset.predict_proba(X_test_fixed)
    
    # Métriques TRAIN
    tr_acc = accuracy_score(y_train_subset, y_train_pred)
    tr_prec = precision_score(y_train_subset, y_train_pred)
    tr_rec = recall_score(y_train_subset, y_train_pred)
    tr_f1 = f1_score(y_train_subset, y_train_pred)
    tr_auc = roc_auc_score(y_train_subset, y_train_prob[:, 1])
    
    # Métriques TEST (fixe)
    test_acc = accuracy_score(y_test_fixed, y_test_pred)
    test_prec = precision_score(y_test_fixed, y_test_pred)
    test_rec = recall_score(y_test_fixed, y_test_pred)
    test_f1 = f1_score(y_test_fixed, y_test_pred)
    test_auc = roc_auc_score(y_test_fixed, y_test_prob[:, 1])
    
    # Hinge loss
    tr_loss = np.mean(np.maximum(0, 1 - y_train_subset * svm_subset.decision_function(X_train_subset)))
    test_loss = np.mean(np.maximum(0, 1 - y_test_fixed * svm_subset.decision_function(X_test_fixed)))
    
    print(f"  Train - Acc: {tr_acc:.4f}, F1: {tr_f1:.4f}, Loss: {tr_loss:.4f}")
    print(f"  Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Loss: {test_loss:.4f}")
    print(f"  Écart Train-Test: {abs(tr_acc - test_acc):.4f}")
    
    trials_metrics.append({
        'pct_train': int(pct*100),
        'train_size': X_train_subset.shape[0],
        'tr_acc': tr_acc, 'test_acc': test_acc,
        'tr_f1': tr_f1, 'test_f1': test_f1,
        'tr_prec': tr_prec, 'test_prec': test_prec,
        'tr_rec': tr_rec, 'test_rec': test_rec,
        'tr_loss': tr_loss, 'test_loss': test_loss,
        'tr_auc': tr_auc, 'test_auc': test_auc,
        'gap_acc': abs(tr_acc - test_acc),
        'gap_f1': abs(tr_f1 - test_f1)
    })

# ============================================
# DATAFRAME DES RÉSULTATS
# ============================================
trials_df = pd.DataFrame(trials_metrics)
trials_df.to_csv('svm_trials_metrics.csv', index=False)
print("\n" + "="*70)
print("RÉSULTATS DES 10 TRIALS")
print("="*70)
print(trials_df[['pct_train', 'train_size', 'tr_acc', 'test_acc', 'gap_acc']].to_string())

# ============================================
# MODÈLE FINAL: Entraînement sur TOUT le dataset fusionné
# ============================================
print("\n" + "="*70)
print("ENTRAÎNEMENT DU MODÈLE FINAL SUR 100% DU DATASET FUSIONNÉ")
print("="*70)

start_time = time.time()
svm_final = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_final.fit(X_full, y_full)
train_time = time.time() - start_time

print(f"✅ Modèle final entraîné sur {X_full.shape[0]} lignes")
print(f"⏱️  Temps d'entraînement: {train_time:.2f} secondes")

# ============================================
# ÉVALUATION FINALE SUR TEST SET
# ============================================
print("\n" + "="*70)
print("ÉVALUATION FINALE SUR LE TEST SET (10,000 lignes non vues)")
print("="*70)

y_test_pred = svm_final.predict(X_test_fixed)
y_test_prob = svm_final.predict_proba(X_test_fixed)

# Métriques finales
test_acc_final = accuracy_score(y_test_fixed, y_test_pred)
test_prec_final = precision_score(y_test_fixed, y_test_pred)
test_rec_final = recall_score(y_test_fixed, y_test_pred)
test_f1_final = f1_score(y_test_fixed, y_test_pred)
test_auc_final = roc_auc_score(y_test_fixed, y_test_prob[:, 1])

print(f"Test Accuracy:  {test_acc_final:.4f} ({test_acc_final*100:.2f}%)")
print(f"Test Precision: {test_prec_final:.4f}")
print(f"Test Recall:    {test_rec_final:.4f}")
print(f"Test F1-Score:  {test_f1_final:.4f}")
print(f"Test ROC-AUC:   {test_auc_final:.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test_fixed, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_fixed, y_test_pred))

# ============================================
# SAUVEGARDE DES RÉSULTATS DÉTAILLÉS
# ============================================
test_results = pd.DataFrame({
    'text': test_df['text'],
    'true_label': y_test_fixed,
    'predicted_label': y_test_pred,
    'is_correct': y_test_fixed == y_test_pred,
    'probability_non_hate': y_test_prob[:, 0],
    'probability_hate': y_test_prob[:, 1]
})
test_results.to_csv('svm_test_evaluation.csv', index=False) 
print("\n✅ Test evaluation saved to 'svm_test_evaluation.csv'")

# ============================================
# PLOTTING - 5 GRAPHIQUES (Train vs Test)
# ============================================
print("\n" + "="*70)
print("GÉNÉRATION DES GRAPHIQUES")
print("="*70)

# Style professionnel
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'train': '#2E86AB', 'test': '#A23B72'}

# Figure 1: 5 subplots (Accuracy, F1, Precision, Recall, Loss)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# 1. Accuracy
axes[0].plot(trials_df['pct_train'], trials_df['tr_acc'], marker='o', linewidth=2, 
             markersize=8, color=colors['train'], label='Train')
axes[0].plot(trials_df['pct_train'], trials_df['test_acc'], marker='s', linewidth=2, 
             markersize=8, color=colors['test'], label='Test')
axes[0].fill_between(trials_df['pct_train'], trials_df['tr_acc'], trials_df['test_acc'], 
                      alpha=0.2, color='gray')
axes[0].set_title('Accuracy vs Training Size', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Percentage of Training Data (%)', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 105)
axes[0].set_ylim(0.5, 1.0)

# 2. F1-Score
axes[1].plot(trials_df['pct_train'], trials_df['tr_f1'], marker='o', linewidth=2, 
             markersize=8, color=colors['train'], label='Train')
axes[1].plot(trials_df['pct_train'], trials_df['test_f1'], marker='s', linewidth=2, 
             markersize=8, color=colors['test'], label='Test')
axes[1].fill_between(trials_df['pct_train'], trials_df['tr_f1'], trials_df['test_f1'], 
                      alpha=0.2, color='gray')
axes[1].set_title('F1-Score vs Training Size', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Percentage of Training Data (%)', fontsize=12)
axes[1].set_ylabel('F1-Score', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 105)
axes[1].set_ylim(0.5, 1.0)

# 3. Precision
axes[2].plot(trials_df['pct_train'], trials_df['tr_prec'], marker='o', linewidth=2, 
             markersize=8, color=colors['train'], label='Train')
axes[2].plot(trials_df['pct_train'], trials_df['test_prec'], marker='s', linewidth=2, 
             markersize=8, color=colors['test'], label='Test')
axes[2].fill_between(trials_df['pct_train'], trials_df['tr_prec'], trials_df['test_prec'], 
                      alpha=0.2, color='gray')
axes[2].set_title('Precision vs Training Size', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Percentage of Training Data (%)', fontsize=12)
axes[2].set_ylabel('Precision', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 105)
axes[2].set_ylim(0.5, 1.0)

# 4. Recall
axes[3].plot(trials_df['pct_train'], trials_df['tr_rec'], marker='o', linewidth=2, 
             markersize=8, color=colors['train'], label='Train')
axes[3].plot(trials_df['pct_train'], trials_df['test_rec'], marker='s', linewidth=2, 
             markersize=8, color=colors['test'], label='Test')
axes[3].fill_between(trials_df['pct_train'], trials_df['tr_rec'], trials_df['test_rec'], 
                      alpha=0.2, color='gray')
axes[3].set_title('Recall vs Training Size', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Percentage of Training Data (%)', fontsize=12)
axes[3].set_ylabel('Recall', fontsize=12)
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_xlim(0, 105)
axes[3].set_ylim(0.5, 1.0)

# 5. Loss
axes[4].plot(trials_df['pct_train'], trials_df['tr_loss'], marker='o', linewidth=2, 
             markersize=8, color=colors['train'], label='Train')
axes[4].plot(trials_df['pct_train'], trials_df['test_loss'], marker='s', linewidth=2, 
             markersize=8, color=colors['test'], label='Test')
axes[4].fill_between(trials_df['pct_train'], trials_df['tr_loss'], trials_df['test_loss'], 
                      alpha=0.2, color='gray')
axes[4].set_title('Hinge Loss vs Training Size', fontsize=14, fontweight='bold')
axes[4].set_xlabel('Percentage of Training Data (%)', fontsize=12)
axes[4].set_ylabel('Loss', fontsize=12)
axes[4].legend()
axes[4].grid(True, alpha=0.3)
axes[4].set_xlim(0, 105)

# 6. ROC-AUC (optionnel)
axes[5].plot(trials_df['pct_train'], trials_df['tr_auc'], marker='o', linewidth=2, 
             markersize=8, color=colors['train'], label='Train')
axes[5].plot(trials_df['pct_train'], trials_df['test_auc'], marker='s', linewidth=2, 
             markersize=8, color=colors['test'], label='Test')
axes[5].fill_between(trials_df['pct_train'], trials_df['tr_auc'], trials_df['test_auc'], 
                      alpha=0.2, color='gray')
axes[5].set_title('ROC-AUC vs Training Size', fontsize=14, fontweight='bold')
axes[5].set_xlabel('Percentage of Training Data (%)', fontsize=12)
axes[5].set_ylabel('ROC-AUC', fontsize=12)
axes[5].legend()
axes[5].grid(True, alpha=0.3)
axes[5].set_xlim(0, 105)
axes[5].set_ylim(0.5, 1.0)

plt.suptitle('Impact of Training Data Size on Model Performance\n(Train vs Test)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('learning_curves_train_vs_test.png', dpi=300, bbox_inches='tight')
print("✅ Learning curves (Train vs Test) saved to 'learning_curves_train_vs_test.png'")

# Figure supplémentaire: Overfitting Gap
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trials_df['pct_train'], trials_df['gap_acc'], marker='o', linewidth=2, 
        markersize=8, color='#C73E1D', label='Accuracy Gap')
ax.plot(trials_df['pct_train'], trials_df['gap_f1'], marker='s', linewidth=2, 
        markersize=8, color='#F18F01', label='F1-Score Gap')
ax.axhline(y=0.05, color='red', linestyle='--', label='Acceptable Threshold (5%)')
ax.fill_between(trials_df['pct_train'], 0, trials_df['gap_acc'], alpha=0.2, color='#C73E1D')
ax.set_title('Overfitting Indicator: Train-Test Gap vs Training Size', fontsize=14, fontweight='bold')
ax.set_xlabel('Percentage of Training Data (%)', fontsize=12)
ax.set_ylabel('Absolute Gap |Train - Test|', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig('overfitting_gap.png', dpi=300)
print("✅ Overfitting gap saved to 'overfitting_gap.png'")

# Confusion Matrix du modèle final
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
            xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'])
plt.title('Confusion Matrix - Final Model (100% Training Data)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("✅ Confusion matrix saved to 'confusion_matrix.png'")

# ROC Curve du modèle final
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test_fixed, y_test_prob[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc_final:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve - Final Model', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_auc_curve.png', dpi=300)
print("✅ ROC-AUC curve saved to 'roc_auc_curve.png'")

# ============================================
# SAUVEGARDE DU MODÈLE
# ============================================
model_artifacts = {
    'svm_model': svm_final,
    'tfidf_vectorizer': tfidf,
    'test_accuracy': test_acc_final,
    'test_precision': test_prec_final,
    'test_recall': test_rec_final,
    'test_f1': test_f1_final,
    'test_roc_auc': test_auc_final,
    'trials_metrics': trials_metrics,
    'training_sizes': trials_df['pct_train'].tolist()
}

with open('best_model.pth', 'wb') as f:
    pickle.dump(model_artifacts, f)
print("\n✅ Best model saved to 'best_model.pth'")

# ============================================
# RÉSUMÉ FINAL
# ============================================
print("\n" + "="*70)
print("RÉSUMÉ FINAL - IMPACT DE LA QUANTITÉ DE DONNÉES")
print("="*70)

print("\n📊 Évolution des performances avec l'augmentation des données:")
print("-"*70)
print(f"{'Données':<12} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Amélioration':<15}")
print("-"*70)

for _, row in trials_df.iterrows():
    pct = row['pct_train']
    train_acc = row['tr_acc']
    test_acc = row['test_acc']
    gap = row['gap_acc']
    improvement = ""
    if pct == 100:
        improvement = "✅ Final"
    elif pct == 10:
        improvement = "Baseline"
    print(f"{pct}% ({int(row['train_size']):,})   {train_acc:.4f}     {test_acc:.4f}     {gap:.4f}     {improvement}")

print("-"*70)
print(f"\n🎯 Meilleure performance (100% des données):")
print(f"   Accuracy:  {test_acc_final:.4f} ({test_acc_final*100:.2f}%)")
print(f"   F1-Score:  {test_f1_final:.4f}")
print(f"   Precision: {test_prec_final:.4f}")
print(f"   Recall:    {test_rec_final:.4f}")
print(f"   ROC-AUC:   {test_auc_final:.4f}")

print(f"\n📈 Amélioration entre 10% et 100% des données:")
baseline = trials_df[trials_df['pct_train']==10].iloc[0]
final = trials_df[trials_df['pct_train']==100].iloc[0]
print(f"   Accuracy:  {baseline['test_acc']:.4f} → {final['test_acc']:.4f} (+{final['test_acc']-baseline['test_acc']:.4f})")
print(f"   F1-Score:  {baseline['test_f1']:.4f} → {final['test_f1']:.4f} (+{final['test_f1']-baseline['test_f1']:.4f})")

print("\n📁 Fichiers générés:")
print("   - svm_trials_metrics.csv              (métriques des 10 trials)")
print("   - svm_test_evaluation.csv             (prédictions détaillées)")
print("   - learning_curves_train_vs_test.png   (5 graphiques Train vs Test)")
print("   - overfitting_gap.png                 (écart Train-Test)")
print("   - confusion_matrix.png                (matrice de confusion)")
print("   - roc_auc_curve.png                  (courbe ROC-AUC)")
print("   - best_model.pth                      (modèle sauvegardé)")

# Petit conseil pour charger le modèle plus tard
print("\n💡 Pour recharger le modèle plus tard:")
print("   import pickle")
print("   with open('best_model.pth', 'rb') as f:")
print("       model = pickle.load(f)")
print("   svm = model['svm_model']")
print("   tfidf = model['tfidf_vectorizer']")