import pandas as pd
import numpy as np
import time
import arabic_reshaper
from bidi.algorithm import get_display
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score, log_loss)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


start_total_time = time.time()

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
print("=" * 70)
print("MODÈLE LOGISTIC REGRESSION – SGD (sklearn) ")
print("=" * 70)

train_df = pd.read_csv('data/train.csv', sep=';')
val_df   = pd.read_csv('data/validation.csv', sep=';')
test_df  = pd.read_csv('data/test.csv', sep=';')

print(f"\n📊 Taille des datasets avant nettoyage:")
print(f"   Train: {len(train_df)} lignes")
print(f"   Validation: {len(val_df)} lignes")
print(f"   Test: {len(test_df)} lignes")

# ============================================================
# NETTOYAGE
# ============================================================
print("\n" + "=" * 70)
print("NETTOYAGE DES VALEURS MANQUANTES")
print("=" * 70)

for df in [train_df, val_df, test_df]:
    df['text'] = df['text'].fillna('')

train_df = train_df[train_df['text'].str.strip() != '']
val_df   = val_df[val_df['text'].str.strip() != '']
test_df  = test_df[test_df['text'].str.strip() != '']

print(f"📊 Taille après nettoyage:")
print(f"   Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
print(f"\n📊 Distribution des labels:")
print(f"   Train – 0: {sum(train_df['label']==0)}, 1: {sum(train_df['label']==1)}")
print(f"   Val   – 0: {sum(val_df['label']==0)},   1: {sum(val_df['label']==1)}")
print(f"   Test  – 0: {sum(test_df['label']==0)},  1: {sum(test_df['label']==1)}")

# ============================================================
# VECTORISATION
# ============================================================
print("\n" + "=" * 70)
print("VECTORISATION (CountVectorizer)")
print("=" * 70)

count_vectorizer = CountVectorizer(
    max_features=10000,
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 2),
    lowercase=True,
    analyzer='word'
)

X_train = count_vectorizer.fit_transform(train_df['text'])
X_val   = count_vectorizer.transform(val_df['text'])
X_test  = count_vectorizer.transform(test_df['text'])
y_train = train_df['label'].values
y_val   = val_df['label'].values
y_test  = test_df['label'].values

print(f"✓ Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ============================================================
# OPTIMISATION DES HYPER-PARAMÈTRES (Validation Set)
# ============================================================
print("\n" + "=" * 70)
print("OPTIMISATION DES HYPER-PARAMÈTRES (Validation Set)")
print("=" * 70)

# SGDClassifier avec loss='log_loss' == Logistic Regression entraînée par SGD
param_grid = {
    'alpha':        [0.0001, 0.001, 0.01, 0.1],   # régularisation (1/C)
    'penalty':      ['l1', 'l2', 'elasticnet'],
    'learning_rate':['optimal', 'constant', 'invscaling'],
    'eta0':         [0.01, 0.1],                   # LR initiale (pour constant/invscaling)
    'class_weight': [None, 'balanced'],
}

total_combinations = (len(param_grid['alpha']) * len(param_grid['penalty']) *
                      len(param_grid['learning_rate']) * len(param_grid['eta0']) *
                      len(param_grid['class_weight']))
print(f"📋 {total_combinations} combinaisons à tester (30 époques rapides)")

best_val_acc = 0
best_params  = None
results_list = []
current = 0

for alpha in param_grid['alpha']:
    for penalty in param_grid['penalty']:
        for lr_sched in param_grid['learning_rate']:
            for eta0 in param_grid['eta0']:
                # 'optimal' calcule son propre LR, eta0 n'est pas utilisé
                if lr_sched == 'optimal' and eta0 != param_grid['eta0'][0]:
                    continue
                for cw in param_grid['class_weight']:
                    current += 1
                    try:
                        model = SGDClassifier(
                            loss='log_loss',          # => Logistic Regression
                            penalty=penalty,
                            alpha=alpha,
                            learning_rate=lr_sched,
                            eta0=eta0,
                            max_iter=30,
                            class_weight=cw,
                            random_state=42,
                            l1_ratio=0.15            # pour elasticnet
                        )
                        model.fit(X_train, y_train)

                        y_val_pred  = model.predict(X_val)
                        y_train_pred = model.predict(X_train)

                        val_acc  = accuracy_score(y_val, y_val_pred)
                        val_f1   = f1_score(y_val, y_val_pred, zero_division=0)
                        train_acc = accuracy_score(y_train, y_train_pred)
                        gap = train_acc - val_acc

                        results_list.append({
                            'alpha': alpha, 'penalty': penalty,
                            'learning_rate': lr_sched, 'eta0': eta0,
                            'class_weight': str(cw),
                            'train_accuracy': train_acc,
                            'val_accuracy': val_acc,
                            'val_f1': val_f1, 'gap': gap
                        })

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = {
                                'alpha': alpha, 'penalty': penalty,
                                'learning_rate': lr_sched, 'eta0': eta0,
                                'class_weight': cw
                            }
                            print(f"\n✅ NEW BEST [{current}]")
                            print(f"   alpha={alpha}, penalty={penalty}, "
                                  f"lr_sched={lr_sched}, eta0={eta0}, class_weight={cw}")
                            print(f"   Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {gap:.4f}")

                    except Exception as e:
                        continue

results_df = pd.DataFrame(results_list)
results_df.to_csv('lr_sgd_param_tuning_results.csv', index=False)

print("\n" + "=" * 70)
print("MEILLEURS PARAMÈTRES TROUVÉS")
print("=" * 70)
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"  Best Validation Accuracy: {best_val_acc:.4f}")

# ============================================================
# ENTRAÎNEMENT FINAL AVEC COURBES PAR ÉPOQUE (partial_fit)
# ============================================================
print("\n" + "=" * 70)
print("ENTRAÎNEMENT FINAL – COURBES PAR ÉPOQUE (partial_fit)")
print("=" * 70)

EPOCHS = 50

# Initialiser le modèle final avec les meilleurs paramètres
lr_final = SGDClassifier(
    loss='log_loss',
    penalty=best_params['penalty'],
    alpha=best_params['alpha'],
    learning_rate=best_params['learning_rate'],
    eta0=best_params['eta0'],
    class_weight=best_params['class_weight'],
    random_state=42,
    l1_ratio=0.15,
    warm_start=False       # on gère nous-mêmes avec partial_fit
)

# Historique par époque
history = {
    'epoch': [],
    'train_loss': [], 'val_loss': [],
    'train_acc': [],  'val_acc': [],
    'train_prec': [], 'val_prec': [],
    'train_rec': [],  'val_rec': [],
    'train_f1': [],   'val_f1': []
}

classes = np.array([0, 1])
n_samples = X_train.shape[0]
indices = np.arange(n_samples)

print(f"\nEntraînement sur {EPOCHS} époques avec partial_fit...\n")
start_train = time.time()

for epoch in range(1, EPOCHS + 1):
    # Mélanger à chaque époque
    np.random.seed(epoch)
    shuffled = np.random.permutation(indices)
    X_shuffled = X_train[shuffled]
    y_shuffled = y_train[shuffled]

    # partial_fit = une passe sur les données (1 époque)
    lr_final.partial_fit(X_shuffled, y_shuffled, classes=classes)

    # ---- Métriques TRAIN ----
    y_train_pred = lr_final.predict(X_train)
    y_train_prob = lr_final.predict_proba(X_train)
    tl = log_loss(y_train, y_train_prob)
    ta = accuracy_score(y_train, y_train_pred)
    tp = precision_score(y_train, y_train_pred, zero_division=0)
    tr = recall_score(y_train, y_train_pred, zero_division=0)
    tf = f1_score(y_train, y_train_pred, zero_division=0)

    # ---- Métriques VALIDATION ----
    y_val_pred = lr_final.predict(X_val)
    y_val_prob = lr_final.predict_proba(X_val)
    vl = log_loss(y_val, y_val_prob)
    va = accuracy_score(y_val, y_val_pred)
    vp = precision_score(y_val, y_val_pred, zero_division=0)
    vr = recall_score(y_val, y_val_pred, zero_division=0)
    vf = f1_score(y_val, y_val_pred, zero_division=0)

    history['epoch'].append(epoch)
    history['train_loss'].append(tl);  history['val_loss'].append(vl)
    history['train_acc'].append(ta);   history['val_acc'].append(va)
    history['train_prec'].append(tp);  history['val_prec'].append(vp)
    history['train_rec'].append(tr);   history['val_rec'].append(vr)
    history['train_f1'].append(tf);    history['val_f1'].append(vf)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Époque {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {tl:.4f}, Acc: {ta:.4f} | "
              f"Val Loss: {vl:.4f}, Acc: {va:.4f}")

train_time = time.time() - start_train
print(f"\n✓ Entraîné en {train_time:.2f} secondes")

# Sauvegarde historique
history_df = pd.DataFrame(history)
history_df.to_csv('lr_sgd_learning_curves_by_epoch.csv', index=False)
print("✓ lr_sgd_learning_curves_by_epoch.csv sauvegardé")

# ============================================================
# GRAPHIQUES DES COURBES D'APPRENTISSAGE PAR ÉPOQUE
# ============================================================
print("\n" + "=" * 70)
print("GÉNÉRATION DES GRAPHIQUES PAR ÉPOQUE")
print("=" * 70)

epochs_x = history['epoch']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Courbes d\'apprentissage par époque – LR SGD (sklearn)',
             fontsize=16, fontweight='bold')

metrics = [
    ('train_loss',  'val_loss',  'Loss (Log Loss)', 'Courbe de Loss'),
    ('train_acc',   'val_acc',   'Accuracy',         'Courbe d\'Accuracy'),
    ('train_prec',  'val_prec',  'Precision',        'Courbe de Precision'),
    ('train_rec',   'val_rec',   'Recall',           'Courbe de Recall'),
    ('train_f1',    'val_f1',    'F1-Score',         'Courbe de F1-Score'),
]

ax_list = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]

for ax, (tk, vk, ylabel, title) in zip(ax_list, metrics):
    ax.plot(epochs_x, history[tk], 'b-o', linewidth=2, markersize=4,
            label=f'Train')
    ax.plot(epochs_x, history[vk], 'r-s', linewidth=2, markersize=4,
            label=f'Validation')
    ax.set_xlabel('Époque', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('lr_sgd_learning_curves_by_epoch.png', dpi=300, bbox_inches='tight')
print("✓ lr_sgd_learning_curves_by_epoch.png sauvegardé")
plt.close()

# ============================================================
# ÉVALUATION SUR TEST SET
# ============================================================
print("\n" + "=" * 70)
print("ÉVALUATION SUR LE TEST SET")
print("=" * 70)

y_pred_test = lr_final.predict(X_test)
y_prob_test = lr_final.predict_proba(X_test)

test_acc  = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test, zero_division=0)
test_rec  = recall_score(y_test, y_pred_test, zero_division=0)
test_f1   = f1_score(y_test, y_pred_test, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_prob_test[:, 1])
cm        = confusion_matrix(y_test, y_pred_test)

print(f"\n📊 Performance sur Test Set:")
print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Precision: {test_prec:.4f}")
print(f"   Recall:    {test_rec:.4f}")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")
print(f"\n📊 Matrice de confusion:")
print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred_test,
                            target_names=['Non-Hate', 'Hate'], zero_division=0))

# ============================================================
# SAUVEGARDE
# ============================================================
test_results = pd.DataFrame({
    'text': test_df['text'],
    'true_label': y_test,
    'predicted_label': y_pred_test,
    'is_correct': y_test == y_pred_test,
    'probability_non_hate': y_prob_test[:, 0],
    'probability_hate': y_prob_test[:, 1]
})
test_results.to_csv('lr_sgd_test_evaluation.csv', index=False)
print("\n✓ lr_sgd_test_evaluation.csv sauvegardé")

model_artifacts = {
    'lr_model': lr_final,
    'count_vectorizer': count_vectorizer,
    'best_params': best_params,
    'best_val_acc': best_val_acc,
    'test_accuracy': test_acc,
    'test_f1': test_f1,
    'test_roc_auc': roc_auc,
    'learning_history_by_epoch': history
}
with open('lr_sgd_best_model.pth', 'wb') as f:
    pickle.dump(model_artifacts, f)
print("✓ lr_sgd_best_model.pth sauvegardé")

# ============================================================
# GRAPHIQUES ADDITIONNELS
# ============================================================
print("\n" + "=" * 70)
print("GÉNÉRATION DES GRAPHIQUES ADDITIONNELS")
print("=" * 70)

# 1. Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.title('Confusion Matrix – LR SGD sklearn (Test Set)',
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('lr_sgd_confusion_matrix.png', dpi=300)
print("✓ lr_sgd_confusion_matrix.png sauvegardé")
plt.close()

# 2. ROC-AUC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob_test[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Classifier')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve – LR SGD sklearn', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_sgd_roc_auc_curve.png', dpi=300)
print("✓ lr_sgd_roc_auc_curve.png sauvegardé")
plt.close()

# 3. Top 20 features (coefficients)
def get_arabic_font():
    arabic_fonts = ['Traditional Arabic', 'Arabic Typesetting', 'Segoe UI', 'Arial']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in arabic_fonts:
        if font in available:
            return font
    return 'sans-serif'

plt.rcParams['font.family'] = get_arabic_font()
plt.rcParams['axes.unicode_minus'] = False

def fix_arabic_text(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

feature_names = count_vectorizer.get_feature_names_out()
coefficients  = lr_final.coef_[0]

top_pos_idx = np.argsort(coefficients)[-20:][::-1]
top_neg_idx = np.argsort(coefficients)[:20]

hate_words     = [fix_arabic_text(feature_names[i]) for i in top_pos_idx]
non_hate_words = [fix_arabic_text(feature_names[i]) for i in top_neg_idx]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(range(20), coefficients[top_pos_idx][::-1], color='red', alpha=0.7)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(hate_words[::-1])
axes[0].set_title('Top 20 mots – HATE', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Coefficient (importance)')

axes[1].barh(range(20), -coefficients[top_neg_idx][::-1], color='green', alpha=0.7)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(non_hate_words[::-1])
axes[1].set_title('Top 20 mots – NON-HATE', fontsize=12, fontweight='bold')
axes[1].set_xlabel('|Coefficient| (importance)')

plt.tight_layout()
plt.savefig('lr_sgd_top_features.png', dpi=300)
print("✓ lr_sgd_top_features.png sauvegardé")
plt.close()

# 4. Effet de alpha (régularisation)
print("\n" + "=" * 70)
print("GRAPHIQUE – EFFET DU PARAMÈTRE ALPHA")
print("=" * 70)

alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
alpha_train_accs, alpha_val_accs = [], []

for alpha_val in alpha_values:
    m = SGDClassifier(
        loss='log_loss',
        penalty=best_params['penalty'],
        alpha=alpha_val,
        learning_rate=best_params['learning_rate'],
        eta0=best_params['eta0'],
        max_iter=30,
        class_weight=best_params['class_weight'],
        random_state=42,
        l1_ratio=0.15
    )
    m.fit(X_train, y_train)
    alpha_train_accs.append(accuracy_score(y_train, m.predict(X_train)))
    alpha_val_accs.append(accuracy_score(y_val, m.predict(X_val)))

plt.figure(figsize=(10, 6))
plt.rcParams['axes.unicode_minus'] = False
plt.plot(alpha_values, alpha_train_accs, 'o-', color='blue', linewidth=2,
         markersize=8, label='Training Accuracy')
plt.plot(alpha_values, alpha_val_accs, 's-', color='red', linewidth=2,
         markersize=8, label='Validation Accuracy')
plt.xscale('log')
plt.xlabel('Alpha (Regularization strength)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Effet du paramètre Alpha – SGD Logistic Regression',
          fontsize=16, fontweight='bold')
plt.axvline(x=best_params['alpha'], color='gray', linestyle='--', alpha=0.7,
            label=f"Best alpha: {best_params['alpha']}")
plt.legend(loc='best', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_sgd_alpha_effect.png', dpi=300)
print("✓ lr_sgd_alpha_effect.png sauvegardé")
plt.close()

# ============================================================
# RÉSUMÉ FINAL
# ============================================================
total_time = time.time() - start_total_time

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"""
╔══════════════════════════════════════════════════════════════╗
║     LOGISTIC REGRESSION – SGDClassifier (sklearn)           ║
║                     SANS TF-IDF                              ║
╠══════════════════════════════════════════════════════════════╣
║  Vectorizer:      CountVectorizer                           ║
║  Max features:    5000  |  N-grams: (1,2)                   ║
║                                                            ║
║  Best Parameters:                                           ║
║    alpha:         {best_params['alpha']}                              ║
║    penalty:       {best_params['penalty']}                                  ║
║    learning_rate: {best_params['learning_rate']}                           ║
║    eta0:          {best_params['eta0']}                                 ║
║    class_weight:  {best_params['class_weight']}                           ║
║    epochs:        {EPOCHS}                                         ║
║                                                            ║
║  Test Performance:                                          ║
║    Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)                         ║
║    F1-Score:   {test_f1:.4f}                                    ║
║    Precision:  {test_prec:.4f}                                    ║
║    Recall:     {test_rec:.4f}                                    ║
║    ROC-AUC:    {roc_auc:.4f}                                    ║
║                                                            ║
║  ⏱️  Train time:  {train_time:.2f} s                                ║
║  ⏱️  Total time:  {total_time:.2f} s ({total_time/60:.2f} min)          ║
╚══════════════════════════════════════════════════════════════╝
""")

print("📁 Fichiers générés:")
print("   - lr_sgd_param_tuning_results.csv       (résultats du tuning)")
print("   - lr_sgd_learning_curves_by_epoch.csv   (métriques par époque)")
print("   - lr_sgd_learning_curves_by_epoch.png   (courbes par époque ✨)")
print("   - lr_sgd_test_evaluation.csv            (prédictions détaillées)")
print("   - lr_sgd_best_model.pth                 (modèle sauvegardé)")
print("   - lr_sgd_confusion_matrix.png           (matrice de confusion)")
print("   - lr_sgd_roc_auc_curve.png              (courbe ROC-AUC)")
print("   - lr_sgd_top_features.png               (coefficients importants)")
print("   - lr_sgd_alpha_effect.png               (effet du paramètre alpha)")

print(f"\n⏱️  Temps total: {total_time:.2f} s ({total_time/60:.2f} min)")

# ============================================================
# TEST INTERACTIF
# ============================================================
print("\n" + "=" * 70)
print("TEST INTERACTIF")
print("=" * 70)

with open('lr_sgd_best_model.pth', 'rb') as f:
    loaded = pickle.load(f)

lr_loaded  = loaded['lr_model']
vectorizer = loaded['count_vectorizer']

print(f"✓ Modèle rechargé | Test Acc: {loaded['test_accuracy']:.2%} | "
      f"F1: {loaded['test_f1']:.4f} | AUC: {loaded['test_roc_auc']:.4f}\n")

textes_test = [
    ("الله يلعنك يا خنزير",           "Arabe – Haineux"),
    ("حركي يا ولاد فرنسا",             "Arabe – Haineux"),
    ("طحان يا رخيس",                   "Arabe – Haineux"),
    ("السلام عليكم ورحمة الله",        "Arabe – Normal"),
    ("الله يبارك فيك",                "Arabe – Normal"),
    ("ربي يحفظك خويا",                "Arabe – Normal"),
    ("تحيا الجزائر",                  "Arabe – Normal"),
    ("salam alikom",                  "Arabizi – Normal"),
    ("mabrouk alik",                  "Arabizi – Normal"),
    ("tu es un chien",                "Français – Haineux"),
    ("sale race",                     "Français – Haineux"),
    ("bonjour comment allez vous",    "Français – Normal"),
    ("merci beaucoup pour votre aide","Français – Normal"),
]

print(f"{'Texte':<48} {'Catégorie':<22} {'Prédiction':<14} {'P(HAINE)':<10} {'P(NON-HAINE)'}")
print("-" * 115)

for texte, cat in textes_test:
    X = vectorizer.transform([texte])
    pred = lr_loaded.predict(X)[0]
    prob = lr_loaded.predict_proba(X)[0]
    t_str    = texte[:44] + "..." if len(texte) > 44 else texte
    pred_str = "🔴 HAINE" if pred == 1 else "🟢 NON-HAINE"
    print(f"{t_str:<48} {cat:<22} {pred_str:<14} {prob[1]:.2%}      {prob[0]:.2%}")

print("\n" + "=" * 70)
print("TEST INTERACTIF (entrez 'q' pour quitter)")
print("=" * 70)

while True:
    texte = input("\n📝 Texte: ").strip()
    if texte.lower() == 'q':
        print("👋 Au revoir!")
        break
    if not texte:
        continue
    X = vectorizer.transform([texte])
    pred = lr_loaded.predict(X)[0]
    prob = lr_loaded.predict_proba(X)[0]
    print(f"   → Prédiction:            {'🔴 HAINE' if pred == 1 else '🟢 NON-HAINE'}")
    print(f"   → Probabilité HAINE:     {prob[1]:.2%}")
    print(f"   → Probabilité NON-HAINE: {prob[0]:.2%}")