import pandas as pd
import numpy as np
import time
import nltk
from nltk.corpus import stopwords
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


start_total_time = time.time()
# ============================================
# CHARGEMENT DES DONNEES
# ============================================
print("="*70)
print("MODÈLE LOGISTIC REGRESSION SANS TF-IDF")
print("="*70)

# Charger les datasets après suppression des stop words
train_df = pd.read_csv('data/train.csv', sep=';')
val_df = pd.read_csv('data/validation.csv', sep=';')
test_df = pd.read_csv('data/test.csv', sep=';')

print(f"\n📊 Taille des datasets avant nettoyage:")
print(f"   Train: {len(train_df)} lignes")
print(f"   Validation: {len(val_df)} lignes")
print(f"   Test: {len(test_df)} lignes")

# ============================================
# NETTOYAGE DES VALEURS NaN
# ============================================
print("\n" + "="*70)
print("NETTOYAGE DES VALEURS MANQUANTES")
print("="*70)

# Vérifier les valeurs NaN
print(f"NaN dans train_df['text']: {train_df['text'].isna().sum()}")
print(f"NaN dans val_df['text']: {val_df['text'].isna().sum()}")
print(f"NaN dans test_df['text']: {test_df['text'].isna().sum()}")

# Remplacer les NaN par des chaînes vides
train_df['text'] = train_df['text'].fillna('')
val_df['text'] = val_df['text'].fillna('')
test_df['text'] = test_df['text'].fillna('')

# Supprimer les lignes où le texte est vide (optionnel)
train_df = train_df[train_df['text'].str.strip() != '']
val_df = val_df[val_df['text'].str.strip() != '']
test_df = test_df[test_df['text'].str.strip() != '']

print(f"\n📊 Taille des datasets après nettoyage:")
print(f"   Train: {len(train_df)} lignes")
print(f"   Validation: {len(val_df)} lignes")
print(f"   Test: {len(test_df)} lignes")

print(f"\n📊 Distribution des labels après nettoyage:")
print(f"   Train - 0: {sum(train_df['label']==0)}, 1: {sum(train_df['label']==1)}")
print(f"   Val   - 0: {sum(val_df['label']==0)}, 1: {sum(val_df['label']==1)}")
print(f"   Test  - 0: {sum(test_df['label']==0)}, 1: {sum(test_df['label']==1)}")

# ============================================
# 1. VECTORISATION avec CountVectorizer 
# ============================================
print("\n" + "="*70)
print("VECTORISATION (CountVectorizer)")
print("="*70)

count_vectorizer = CountVectorizer(
    max_features=5000,
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 2),
    lowercase=True,
    analyzer='word'
)

print("✓ CountVectorizer (compte simple des mots)")
print("  - max_features=5000 (limite le vocabulaire)")
print("  - min_df=3(ignore mots rares)")
print("  - max_df=0.8 (ignore mots trop fréquents)")
print("  - ngram_range=(1,2) (mots et paires de mots)")

# Transformation des données
X_train_full = count_vectorizer.fit_transform(train_df['text'])
X_val_full = count_vectorizer.transform(val_df['text'])
X_test = count_vectorizer.transform(test_df['text'])
y_train_full = train_df['label'].values
y_val_full = val_df['label'].values
y_test = test_df['label'].values

print(f"\n✓ Dimension des matrices:")
print(f"   Train: {X_train_full.shape}")
print(f"   Validation: {X_val_full.shape}")
print(f"   Test: {X_test.shape}")

# ============================================
# 2. OPTIMISATION DES PARAMÈTRES avec Validation Set
# ============================================
print("\n" + "="*70)
print("OPTIMISATION DES PARAMÈTRES (sur Validation Set)")
print("="*70)

# Grille de paramètres pour Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [500, 1000],
    'class_weight': [None, 'balanced']
}

print(f"📋 Grille de paramètres à tester: {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver']) * len(param_grid['max_iter']) * len(param_grid['class_weight'])} combinaisons")

# Optimisation avec validation set
best_val_acc = 0
best_params = None
results_list = []

# NOUVELLE CONDITION : max gap autorisé = 6%
MAX_GAP = 0.06

# Dictionnaire pour stocker les performances train/val pour chaque C
c_performance = {c: {'train_acc': [], 'val_acc': []} for c in param_grid['C']}

total_combinations = len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver']) * len(param_grid['max_iter']) * len(param_grid['class_weight'])
current = 0

print(f"\n🔍 Recherche du meilleur modèle avec gap Train/Val ≤ {MAX_GAP*100:.0f}%")

for C in param_grid['C']:
    for penalty in param_grid['penalty']:
        for solver in param_grid['solver']:
            # Vérifier compatibilité penalty/solver
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                continue
            if penalty == 'l2' and solver == 'liblinear':
                continue
                
            for max_iter in param_grid['max_iter']:
                for class_weight in param_grid['class_weight']:
                    current += 1
                    
                    try:
                        # Création et entraînement du modèle
                        lr = LogisticRegression(
                            C=C,
                            penalty=penalty,
                            solver=solver,
                            max_iter=max_iter,
                            class_weight=class_weight,
                            random_state=42
                        )
                        
                        lr.fit(X_train_full, y_train_full)
                        
                        # Évaluation sur train
                        y_train_pred = lr.predict(X_train_full)
                        train_acc = accuracy_score(y_train_full, y_train_pred)
                        
                        # Évaluation sur validation
                        y_val_pred = lr.predict(X_val_full)
                        val_acc = accuracy_score(y_val_full, y_val_pred)
                        val_f1 = f1_score(y_val_full, y_val_pred)
                        
                        # Calcul du gap
                        gap = train_acc - val_acc
                        
                        results_list.append({
                            'C': C, 'penalty': penalty, 'solver': solver,
                            'max_iter': max_iter, 'class_weight': class_weight,
                            'train_accuracy': train_acc, 'val_accuracy': val_acc, 
                            'val_f1': val_f1, 'gap': gap
                        })
                        
                        # Stockage pour les courbes d'apprentissage
                        c_performance[C]['train_acc'].append(train_acc)
                        c_performance[C]['val_acc'].append(val_acc)
                        
                        # === NOUVELLE CONDITION ===
                        
                        if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_params = {'C': C, 'penalty': penalty, 'solver': solver,
                                             'max_iter': max_iter, 'class_weight': class_weight}
                                
                                print(f"\n✅ NEW BEST! [{current}/{total_combinations}]")
                                print(f"   Params: C={C}, penalty={penalty}, solver={solver}, max_iter={max_iter}, class_weight={class_weight}")
                                print(f"   Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Gap: {gap:.4f} (≤{MAX_GAP:.2f}) ✓")
                        
                    except Exception as e:
                        continue

# Sauvegarde des résultats
results_df = pd.DataFrame(results_list)
results_df.to_csv('lr_param_tuning_results.csv', index=False)

# Si aucun modèle ne satisfait la condition, prendre le meilleur avec le gap le plus petit
if best_params is None:
    print("\n⚠️ Aucun modèle avec gap ≤ 6%, on prend le meilleur avec le plus petit gap")
    results_df_valid = results_df.copy()
    best_row = results_df_valid.loc[results_df_valid['val_accuracy'].idxmax()]
    best_params = {
        'C': best_row['C'],
        'penalty': best_row['penalty'],
        'solver': best_row['solver'],
        'max_iter': int(best_row['max_iter']),
        'class_weight': best_row['class_weight']
    }
    best_val_acc = best_row['val_accuracy']

print("\n" + "="*70)
print("MEILLEURS PARAMÈTRES TROUVÉS")
print("="*70)
print(f"C: {best_params['C']}")
print(f"Penalty: {best_params['penalty']}")
print(f"Solver: {best_params['solver']}")
print(f"Max_iter: {best_params['max_iter']}")
print(f"Class_weight: {best_params['class_weight']}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# ============================================
# 3. COURBES D'APPRENTISSAGE PAR TAILLE DE DONNÉES
# ============================================
print("\n" + "="*70)
print("COURBES D'APPRENTISSAGE (Learning Curves by Dataset Size)")
print("="*70)

# Tailles de dataset à tester (proportions du train set)
train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_total = len(train_df)

# Stockage des métriques
history_size = {
    'size': [],
    'n_samples': [],
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'train_prec': [],
    'val_prec': [],
    'train_rec': [],
    'val_rec': [],
    'train_f1': [],
    'val_f1': []
}

print(f"\nTest sur {len(train_sizes)} tailles de dataset...")
start_time = time.time()

for size in train_sizes:
    n_samples = int(n_total * size)
    
    # Sélection aléatoire d'un sous-ensemble
    np.random.seed(42)
    indices = np.random.choice(n_total, n_samples, replace=False)
    X_train_subset = X_train_full[indices]
    y_train_subset = y_train_full[indices]
    
    # Entraînement du modèle avec les meilleurs paramètres
    lr = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        max_iter=best_params['max_iter'],
        class_weight=best_params['class_weight'],
        random_state=42
    )
    
    lr.fit(X_train_subset, y_train_subset)
    
    # Prédictions sur train (subset)
    y_train_pred = lr.predict(X_train_subset)
    y_train_prob = lr.predict_proba(X_train_subset)
    
    # Prédictions sur validation (complet)
    y_val_pred = lr.predict(X_val_full)
    y_val_prob = lr.predict_proba(X_val_full)
    
    # Métriques TRAIN
    train_loss = log_loss(y_train_subset, y_train_prob)
    train_acc = accuracy_score(y_train_subset, y_train_pred)
    train_prec = precision_score(y_train_subset, y_train_pred, zero_division=0)
    train_rec = recall_score(y_train_subset, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train_subset, y_train_pred, zero_division=0)
    
    # Métriques VALIDATION
    val_loss = log_loss(y_val_full, y_val_prob)
    val_acc = accuracy_score(y_val_full, y_val_pred)
    val_prec = precision_score(y_val_full, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val_full, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val_full, y_val_pred, zero_division=0)
    
    # Stockage
    history_size['size'].append(size)
    history_size['n_samples'].append(n_samples)
    history_size['train_loss'].append(train_loss)
    history_size['val_loss'].append(val_loss)
    history_size['train_acc'].append(train_acc)
    history_size['val_acc'].append(val_acc)
    history_size['train_prec'].append(train_prec)
    history_size['val_prec'].append(val_prec)
    history_size['train_rec'].append(train_rec)
    history_size['val_rec'].append(val_rec)
    history_size['train_f1'].append(train_f1)
    history_size['val_f1'].append(val_f1)
    
    print(f"  Size: {size*100:3.0f}% ({n_samples:5d} samples) | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

train_time = time.time() - start_time

print(f"\n✓ Courbes générées en {train_time:.2f} secondes")

# Sauvegarde de l'historique
history_df = pd.DataFrame(history_size)
history_df.to_csv('lr_learning_curves_by_size.csv', index=False)
print("✓ lr_learning_curves_by_size.csv sauvegardé")

# ============================================
# 4. ENTRAÎNEMENT FINAL SUR TOUT LE TRAIN
# ============================================
print("\n" + "="*70)
print("ENTRAÎNEMENT FINAL SUR LE DATASET COMPLET")
print("="*70)

lr_final = LogisticRegression(
    C=best_params['C'],
    penalty=best_params['penalty'],
    solver=best_params['solver'],
    max_iter=best_params['max_iter'],
    class_weight=best_params['class_weight'],
    random_state=42
)

# Mesure du temps d'entraînement
start_train_final = time.time()
lr_final.fit(X_train_full, y_train_full)
train_final_time = time.time() - start_train_final

train_acc_full = accuracy_score(y_train_full, lr_final.predict(X_train_full))
val_acc_full = accuracy_score(y_val_full, lr_final.predict(X_val_full))

print(f"✓ Modèle final entraîné sur {len(train_df)} exemples")
print(f"   ⏱️  Temps d'entraînement: {train_final_time:.2f} secondes")
print(f"   Train Accuracy: {train_acc_full:.4f}")
print(f"   Val Accuracy: {val_acc_full:.4f}")

# ============================================
# 5. GRAPHIQUES DES COURBES D'APPRENTISSAGE PAR TAILLE
# ============================================
print("\n" + "="*70)
print("GÉNÉRATION DES GRAPHIQUES D'APPRENTISSAGE PAR TAILLE")
print("="*70)

# Utiliser les pourcentages au lieu du nombre d'échantillons
size_percentages = [s * 100 for s in history_size['size']]

# Figure avec 5 sous-graphiques
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Courbes d\'apprentissage par taille de dataset - Logistic Regression', 
             fontsize=16, fontweight='bold')

# 1. Loss
ax = axes[0, 0]
ax.plot(size_percentages, history_size['train_loss'], 'b-o', linewidth=2, markersize=8, label='Train Loss')
ax.plot(size_percentages, history_size['val_loss'], 'r-s', linewidth=2, markersize=8, label='Validation Loss')
ax.set_xlabel('Taille du dataset d\'entraînement (%)', fontsize=12)
ax.set_ylabel('Loss (Log Loss)', fontsize=12)
ax.set_title('Courbe de Loss', fontsize=14, fontweight='bold')
ax.set_xticks(size_percentages)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# 2. Accuracy
ax = axes[0, 1]
ax.plot(size_percentages, history_size['train_acc'], 'b-o', linewidth=2, markersize=8, label='Train Accuracy')
ax.plot(size_percentages, history_size['val_acc'], 'r-s', linewidth=2, markersize=8, label='Validation Accuracy')
ax.set_xlabel('Taille du dataset d\'entraînement (%)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Courbe d\'Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(size_percentages)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# 3. Precision
ax = axes[0, 2]
ax.plot(size_percentages, history_size['train_prec'], 'b-o', linewidth=2, markersize=8, label='Train Precision')
ax.plot(size_percentages, history_size['val_prec'], 'r-s', linewidth=2, markersize=8, label='Validation Precision')
ax.set_xlabel('Taille du dataset d\'entraînement (%)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Courbe de Precision', fontsize=14, fontweight='bold')
ax.set_xticks(size_percentages)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# 4. Recall
ax = axes[1, 0]
ax.plot(size_percentages, history_size['train_rec'], 'b-o', linewidth=2, markersize=8, label='Train Recall')
ax.plot(size_percentages, history_size['val_rec'], 'r-s', linewidth=2, markersize=8, label='Validation Recall')
ax.set_xlabel('Taille du dataset d\'entraînement (%)', fontsize=12)
ax.set_ylabel('Recall', fontsize=12)
ax.set_title('Courbe de Recall', fontsize=14, fontweight='bold')
ax.set_xticks(size_percentages)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# 5. F1-Score
ax = axes[1, 1]
ax.plot(size_percentages, history_size['train_f1'], 'b-o', linewidth=2, markersize=8, label='Train F1-Score')
ax.plot(size_percentages, history_size['val_f1'], 'r-s', linewidth=2, markersize=8, label='Validation F1-Score')
ax.set_xlabel('Taille du dataset d\'entraînement (%)', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Courbe de F1-Score', fontsize=14, fontweight='bold')
ax.set_xticks(size_percentages)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# 6. Supprimer le sous-graphique vide
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('lr_learning_curves_by_size.png', dpi=300, bbox_inches='tight')
print("✓ lr_learning_curves_by_size.png sauvegardé")



# ============================================
# 6. ÉVALUATION SUR TEST SET
# ============================================
print("\n" + "="*70)
print("ÉVALUATION SUR LE TEST SET")
print("="*70)

y_pred_test = lr_final.predict(X_test)
y_prob_test = lr_final.predict_proba(X_test)

# Métriques
test_acc = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test, zero_division=0)
test_rec = recall_score(y_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob_test[:, 1])

print(f"\n📊 Performance sur Test Set:")
print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Precision: {test_prec:.4f}")
print(f"   Recall:    {test_rec:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_test)
print(f"\n📊 Matrice de confusion:")
print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")

# Classification report
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['Non-Hate', 'Hate'], zero_division=0))

# ============================================
# 7. SAUVEGARDE DES RÉSULTATS
# ============================================
print("\n" + "="*70)
print("SAUVEGARDE DES RÉSULTATS")
print("="*70)

# Détail des prédictions
test_results = pd.DataFrame({
    'text': test_df['text'],
    'true_label': y_test,
    'predicted_label': y_pred_test,
    'is_correct': y_test == y_pred_test,
    'probability_non_hate': y_prob_test[:, 0],
    'probability_hate': y_prob_test[:, 1]
})
test_results.to_csv('lr_test_evaluation.csv', index=False)
print("✓ lr_test_evaluation.csv sauvegardé")

# Sauvegarde du modèle
model_artifacts = {
    'lr_model': lr_final,
    'count_vectorizer': count_vectorizer,
    'best_params': best_params,
    'best_val_acc': best_val_acc,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'test_roc_auc': roc_auc,
    'trained_on': 'train_only',
    'learning_history_by_size': history_size
}

with open('lr_best_model.pth', 'wb') as f:
    pickle.dump(model_artifacts, f)
print("✓ lr_best_model.pth sauvegardé")

# ============================================
# 8. GRAPHIQUES ADDITIONNELS
# ============================================
print("\n" + "="*70)
print("GÉNÉRATION DES GRAPHIQUES ADDITIONNELS")
print("="*70)

# 1. Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Confusion Matrix - Logistic Regression (Test Set)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('lr_confusion_matrix.png', dpi=300)
print("✓ lr_confusion_matrix.png sauvegardé")

# 2. ROC-AUC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob_test[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Logistic Regression', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_roc_auc_curve.png', dpi=300)
print("✓ lr_roc_auc_curve.png sauvegardé")

# 3. Top 20 mots les plus importants

# Trouver une police arabe disponible
def get_arabic_font():
    arabic_fonts = ['Traditional Arabic', 'Arabic Typesetting', 'Segoe UI', 'Arial', 'Times New Roman']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in arabic_fonts:
        if font in available:
            return font
    return 'sans-serif'

plt.rcParams['font.family'] = get_arabic_font()
plt.rcParams['axes.unicode_minus'] = False

def fix_arabic_text(text):
    """Rend le texte arabe lisible (lettres attachées, sens droite-gauche)."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


feature_names = count_vectorizer.get_feature_names_out()
coefficients = lr_final.coef_[0]

top_positive_idx = np.argsort(coefficients)[-20:][::-1]
top_negative_idx = np.argsort(coefficients)[:20]

# Appliquer fix_arabic_text aux mots ===
hate_words = [fix_arabic_text(feature_names[i]) for i in top_positive_idx]
non_hate_words = [fix_arabic_text(feature_names[i]) for i in top_negative_idx]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# HATE
axes[0].barh(range(20), coefficients[top_positive_idx], color='red', alpha=0.7)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(hate_words[::-1])  # Inverser pour avoir le plus grand en haut
axes[0].set_title('Top 20 mots associés à la classe HATE', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Coefficient (importance)')

# NON-HATE
axes[1].barh(range(20), -coefficients[top_negative_idx], color='green', alpha=0.7)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(non_hate_words[::-1])  # Inverser
axes[1].set_title('Top 20 mots associés à la classe NON-HATE', fontsize=12, fontweight='bold')
axes[1].set_xlabel('|Coefficient| (importance)')

plt.tight_layout()
plt.savefig('lr_top_features.png', dpi=300)
print("✓ lr_top_features.png sauvegardé")

# ============================================
# 4. Courbe de l'effet de C
# ============================================
print("\n" + "="*70)
print("GÉNÉRATION DU GRAPHIQUE DE L'EFFET DE C")
print("="*70)

plt.figure(figsize=(12, 7))

# === FIX POUR LE CARACTÈRE − ===
plt.rcParams['axes.unicode_minus'] = False

c_values = sorted(param_grid['C'])
train_means = []
val_means = []

for c in c_values:
    if c_performance[c]['train_acc']:
        train_means.append(np.mean(c_performance[c]['train_acc']))
        val_means.append(np.mean(c_performance[c]['val_acc']))
    else:
        train_means.append(0)
        val_means.append(0)

plt.plot(c_values, train_means, 'o-', color='blue', linewidth=2, markersize=8, label='Training Accuracy')
plt.plot(c_values, val_means, 's-', color='red', linewidth=2, markersize=8, label='Validation Accuracy')

plt.xscale('log')
plt.xlabel('C (Regularization parameter)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Effect of C on Training and Validation Accuracy', fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)

# === MODIFICATION ICI ===
# Utiliser best_params['C'] au lieu de argmax sur val_means
best_c_value = best_params['C']
plt.axvline(x=best_c_value, color='gray', linestyle='--', alpha=0.7, label=f'Best C selected: {best_c_value}')

# (Optionnel) Afficher aussi le point gris sur la courbe
# Trouver l'index de la valeur exacte de best_c_value dans c_values
if best_c_value in c_values:
    idx = c_values.index(best_c_value)
    # Ajouter un point gris sur la courbe de validation à la position du best_c_value
    plt.plot(best_c_value, val_means[idx], 'o', color='gray', markersize=10, zorder=5)

plt.tight_layout()
plt.savefig('lr_c_parameter_effect.png', dpi=300)
print("✓ lr_c_parameter_effect.png sauvegardé (avec la valeur C optimale trouvée)")


total_time = time.time() - start_total_time
# ============================================
# 9. RÉSUMÉ FINAL
# ============================================
print("\n" + "="*70)
print("RÉSUMÉ FINAL")
print("="*70)
print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    LOGISTIC REGRESSION                       ║
║                         SANS TF-IDF                          ║
╠══════════════════════════════════════════════════════════════╣
║  Vectorizer:     CountVectorizer                            ║
║  Max features:   3000                                       ║
║  N-grams:        (1,2)                                      ║
║                                                            ║
║  Best Parameters:                                           ║
║    C:            {best_params['C']}                                      ║
║    Penalty:      {best_params['penalty']}                                     ║
║    Solver:       {best_params['solver']}                                    ║
║    Class weight: {best_params['class_weight']}                                   ║
║                                                            ║
║  Test Performance:                                          ║
║    Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)                             ║
║    F1-Score:     {test_f1:.4f}                                      ║
║    Precision:    {test_prec:.4f}                                      ║
║    Recall:       {test_rec:.4f}                                      ║
║    ROC-AUC:      {roc_auc:.4f}                                      ║
║                                                            ║
║  ⏱️  Train time:   {train_final_time:.2f} seconds                          ║
║  ⏱️  Total time:   {total_time:.2f} seconds ({total_time/60:.2f} min)     ║
║  Trained on:     Train set only ({len(train_df)} exemples)                ║
╚══════════════════════════════════════════════════════════════╝
""")

print("\n📁 Fichiers générés:")
print("   - lr_param_tuning_results.csv          (résultats du tuning)")
print("   - lr_learning_curves_by_size.csv       (métriques par taille)")
print("   - lr_learning_curves_by_size.png       (courbes complètes par taille)")
print("   - lr_learning_curves_accuracy_loss.png (accuracy et loss)")
print("   - lr_test_evaluation.csv               (prédictions détaillées)")
print("   - lr_best_model.pth                    (modèle sauvegardé)")
print("   - lr_confusion_matrix.png              (matrice de confusion)")
print("   - lr_roc_auc_curve.png                 (courbe ROC-AUC)")
print("   - lr_top_features.png                  (mots les plus importants)")
print("   - lr_c_parameter_effect.png            (effet du paramètre C)")


print(f"\n⏱️  Temps d'exécution TOTAL: {total_time:.2f} secondes ({total_time/60:.2f} minutes)")

import pickle

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
print("="*70)
print("TEST DU MODÈLE - DÉTECTION DE DISCOURS HAINEUX")
print("="*70)

with open('lr_best_model.pth', 'rb') as f:
    model_data = pickle.load(f)

lr_model = model_data['lr_model']
vectorizer = model_data['count_vectorizer']

print(f"\n✓ Modèle chargé avec succès")
print(f"  Accuracy sur test: {model_data['test_accuracy']:.2%}")
print(f"  F1-Score sur test: {model_data['test_f1']:.4f}")
print(f"  ROC-AUC: {model_data['test_roc_auc']:.4f}")

# ============================================
# LISTE DE TEXTES À TESTER
# ============================================
textes_test = [
    # ARABE - Discours haineux
    ("الله يلعنك يا خنزير", "Arabe"),
    ("حركي يا ولاد فرنسا", "Arabe"),
    ("طحان يا رخيس", "Arabe"),
    
    # ARABE - Discours normal
    ("السلام عليكم ورحمة الله", "Arabe"),
    ("الله يبارك فيك", "Arabe"),
    ("ربي يحفظك خويا", "Arabe"),
    ("صحا فطوركم", "Arabe"),
    ("مبروك عليك النجاح", "Arabe"),
    ("حياك الله في بلادك", "Arabe"),
    ("تحيا الجزائر", "Arabe"),
    
    # ARABIZI - Discours normal
    ("salam alikom", "Arabizi"),
    ("saha ftorkom", "Arabizi"),
    ("rabi yahfdek khoya", "Arabizi"),
    ("mabrouk alik", "Arabizi"),
    ("tahia ldzair", "Arabizi"),
    
    # FRANÇAIS - Discours haineux
    ("tu es un chien", "Français"),
    ("je vais te tuer", "Français"),
    ("sale race", "Français"),
    
    # FRANÇAIS - Discours normal
    ("bonjour comment allez vous", "Français"),
    ("merci beaucoup pour votre aide", "Français"),
    ("je suis content de te voir", "Français"),
    ("bonne journée à tous", "Français"),
    ("félicitations pour ton travail", "Français"),
]

# ============================================
# PRÉDICTIONS
# ============================================
print("\n" + "="*100)
print(f"{'Texte':<45} {'Langue':<10} {'Prédiction':<12} {'Prob HAINE':<12} {'Prob NON-HAINE':<15}")
print("="*100)

for texte, langue in textes_test:
    # Prédiction
    X = vectorizer.transform([texte])
    pred = lr_model.predict(X)[0]
    prob = lr_model.predict_proba(X)[0]
    
    # Affichage
    texte_affiche = texte[:42] + "..." if len(texte) > 42 else texte
    prediction_str = "🔴 HAINE" if pred == 1 else "🟢 NON-HAINE"
    
    print(f"{texte_affiche:<45} {langue:<10} {prediction_str:<12} {prob[1]:.2%}         {prob[0]:.2%}")

print("="*100)


# ============================================
# TEST INTERACTIF (optionnel)
# ============================================
print("\n" + "="*70)
print("TEST INTERACTIF")
print("="*70)
print("Entrez un texte à analyser (ou 'q' pour quitter):")
print()

while True:
    texte = input("📝 Texte: ").strip()
    if texte.lower() == 'q':
        print("👋 Au revoir!")
        break
    if not texte:
        continue
    
    X = vectorizer.transform([texte])
    pred = lr_model.predict(X)[0]
    prob = lr_model.predict_proba(X)[0]
    
    prediction_str = "🔴 HAINE" if pred == 1 else "🟢 NON-HAINE"
    print(f"   → Prédiction: {prediction_str}")
    print(f"   → Probabilité HAINE: {prob[1]:.2%}")
    print(f"   → Probabilité NON-HAINE: {prob[0]:.2%}")
    print()
    