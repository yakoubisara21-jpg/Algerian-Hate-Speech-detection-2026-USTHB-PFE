import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, log_loss)
import warnings
import time
import joblib
import pickle
warnings.filterwarnings('ignore')
import os
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')

# Set your paths
code_location = '/content/drive/MyDrive/TF-IDF+LogisticRegression/'
results_folder = os.path.join(code_location, 'Results-TF-IDF+LogisticRegression')

os.makedirs(results_folder, exist_ok=True)
os.chdir(results_folder)
print(f"Current working directory: {os.getcwd()}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

COLORS = {
    'train': 'b',
    'val':   'r',
    'aux1':  '#F18F01',
    'aux2':  '#C73E1D',
    'aux3':  '#3D5A80',
}

# ==================== 1. LOAD AND PREPROCESS DATA ====================

print("Loading datasets...")
start_time_total = time.time()

df_train = pd.read_csv('/content/drive/MyDrive/data/kfoldsdata_sans_stopwords.csv', sep=';')
df_test  = pd.read_csv('/content/drive/MyDrive/data/test_sans_stopwords.csv',       sep=';')

df_train = df_train.dropna()
df_test  = df_test.dropna()

print(f"Training set size: {len(df_train)}")
print(f"Test set size: {len(df_test)}")

X_train = df_train['text'].astype(str)
y_train = df_train['label'].astype(int)
X_test  = df_test['text'].astype(str)
y_test  = df_test['label'].astype(int)

print(f"\nClass distribution in training:\n{y_train.value_counts()}")
print(f"\nClass distribution in test:\n{y_test.value_counts()}")

# ==================== 2. TF-IDF VECTORIZATION ====================

print("\nVectorizing text data...")
vectorizer_start = time.time()

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words=None,
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

vectorizer_time = time.time() - vectorizer_start
print(f"TF-IDF shape: {X_train_tfidf.shape}")
print(f"Vectorization time: {vectorizer_time:.2f} seconds")

# ==================== 3. GRID SEARCH WITH 5-FOLD CV ====================

print("\nPerforming Grid Search with 5-fold cross-validation...")
gridsearch_start = time.time()

param_grid = {
    'solver':       ['lbfgs', 'saga', 'liblinear'],
    'penalty':      ['l2'],
    'C':            [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'class_weight': [None, 'balanced'],
    'max_iter':     [1000],
}

lr  = LogisticRegression(random_state=42)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    lr, param_grid, cv=cv, scoring='f1_macro',
    n_jobs=-1, verbose=1, return_train_score=True,
)
grid_search.fit(X_train_tfidf, y_train)

pd.DataFrame([grid_search.best_params_]).to_csv(
    'meilleurs_hyperparametres.csv', index=False)
print("Best hyperparameters saved to 'meilleurs_hyperparametres.csv'")

gridsearch_time = time.time() - gridsearch_start
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
print(f"Grid search time: {gridsearch_time:.2f} seconds")

grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results.to_csv('grid_search_results.csv', index=False)
print("\nGrid search results saved to 'grid_search_results.csv'")

# ==================== 4. BEST MODEL ====================

print("\nUsing best Logistic Regression model ...")
best_lr = grid_search.best_estimator_

# ==================== 5. FINAL EVALUATION ON TEST SET ====================

print("\nEvaluating on test set...")
inference_start = time.time()

y_pred = best_lr.predict(X_test_tfidf)
y_prob = best_lr.predict_proba(X_test_tfidf)[:, 1]

inference_time = time.time() - inference_start
total_time     = time.time() - start_time_total

accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

test_metrics = {
    'Exactitude': accuracy,
    'F1-Score':   f1,
    'Rappel':     recall,
    'Précision':  precision,
    'ROC AUC':    roc_auc
}

target_names = ['0', '1']
class_report  = classification_report(y_test, y_pred,
                    target_names=target_names, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# ==================== 5b. SAVE TIME METRICS ====================

time_metrics = {
    'Métrique':    ['Vectorisation', 'Recherche hyperparamètres',
                    'Inférence (ensemble de test)', 'Exécution totale'],
    'Temps (sec)': [vectorizer_time, gridsearch_time, inference_time, total_time],
    'Temps (min)': [vectorizer_time/60, gridsearch_time/60,
                    inference_time/60, total_time/60],
}
pd.DataFrame(time_metrics).to_csv('temps_execution.csv', index=False)
print("\nTemps d'exécution sauvegardés dans 'temps_execution.csv'")

print("\n=== TEMPS D'EXÉCUTION ===")
for m, ts, tm in zip(time_metrics['Métrique'],
                     time_metrics['Temps (sec)'],
                     time_metrics['Temps (min)']):
    print(f"  {m}: {ts:.2f} sec ({tm:.2f} min)")

print("\n=== MÉTRIQUES SUR L'ENSEMBLE DE TEST ===")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

pd.DataFrame([test_metrics]).to_csv('test_metrics.csv', index=False)
print("\nMétriques de test sauvegardées dans 'test_metrics.csv'")
print(classification_report(y_test, y_pred, target_names=target_names))
class_report_df.to_csv('classification_report.csv')
print("Rapport de classification sauvegardé dans 'classification_report.csv'")

# Detailed evaluation dataframe
eval_df = pd.DataFrame({
    'texte':                 X_test.values,
    'vrai_label':            y_test.values,
    'label_predit':          y_pred,
    'est_correct':           y_test.values == y_pred,
    'probabilite_haine':     y_prob,
    'probabilite_non_haine': 1 - y_prob,
})
eval_df.to_csv('test_evaluation.csv', index=False, encoding='utf-8-sig')
print("Évaluation de test sauvegardée dans 'test_evaluation.csv'")

# Save FP / FN  (same as SVC)
fp_df = eval_df[(eval_df['vrai_label'] == 0) & (eval_df['label_predit'] == 1)]
fn_df = eval_df[(eval_df['vrai_label'] == 1) & (eval_df['label_predit'] == 0)]

if len(fp_df) > 0:
    fp_df.to_csv(f'faux_positifs_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f"Faux positifs: {len(fp_df)} → faux_positifs_{timestamp}.csv")

if len(fn_df) > 0:
    fn_df.to_csv(f'faux_negatifs_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f"Faux négatifs: {len(fn_df)} → faux_negatifs_{timestamp}.csv")

# ==================== 6. COURBES D'APPRENTISSAGE (style SVC) ====================

print("\nGénération des courbes d'apprentissage...")

def plot_learning_curves_lr(estimator, X, y, cv_folds=5,
                             train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Courbes d'apprentissage pour Régression Logistique avec TF-IDF.
    Style identique au LinearSVC :
      - seaborn-v0_8-darkgrid
      - bleu (train) / rouge (val), marqueurs o- / s-
      - bandes ±1 std, alpha=0.12
      - grille 2×3, 6e case masquée
      - labels et titres en français
    """
    metrics_config = [
        ('accuracy',  'Exactitude'),
        ('f1',        'F1-Score'),
        ('precision', 'Précision'),
        ('recall',    'Rappel'),
    ]

    df_curves = pd.DataFrame()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Métriques validation croisée — Courbes d\'Apprentissage pour Régression Logistique avec TF-IDF\n',
        fontsize=14, fontweight='bold',
    )
    axes_flat = axes.flatten()

    # ── 4 métriques classiques ──
    for i, (metric_key, metric_label) in enumerate(metrics_config):
        ax = axes_flat[i]

        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y,
            cv=cv_folds,
            scoring=metric_key,
            train_sizes=train_sizes,
            n_jobs=-1,
        )

        tr_m = np.mean(train_scores, axis=1)
        tr_s = np.std(train_scores,  axis=1)
        vl_m = np.mean(val_scores,   axis=1)
        vl_s = np.std(val_scores,    axis=1)

        if i == 0:
            df_curves['taille_entrainement'] = train_sizes_abs
        df_curves[f'{metric_key}_train_moy'] = tr_m
        df_curves[f'{metric_key}_train_std'] = tr_s
        df_curves[f'{metric_key}_val_moy']   = vl_m
        df_curves[f'{metric_key}_val_std']   = vl_s

        ax.fill_between(train_sizes_abs, tr_m - tr_s, tr_m + tr_s,
                        alpha=0.12, color=COLORS['train'])
        ax.fill_between(train_sizes_abs, vl_m - vl_s, vl_m + vl_s,
                        alpha=0.12, color=COLORS['val'])

        ax.plot(train_sizes_abs, tr_m, 'o-',
                color=COLORS['train'], linewidth=2, markersize=6,
                label='Entraînement')
        ax.plot(train_sizes_abs, vl_m, 's-',
                color=COLORS['val'], linewidth=2.5, markersize=8,
                label='Validation')

        ax.set_xlabel("Taille de l'ensemble d'entraînement", fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'{metric_label} (moy. sur {cv_folds} plis)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── 5e graphique : Log Loss ──
    ax_loss = axes_flat[4]

    def log_loss_scorer(est, X_, y_):
        y_p = est.predict_proba(X_)
        return -log_loss(y_, y_p)

    _, train_loss, val_loss = learning_curve(
        estimator, X, y,
        cv=cv_folds,
        scoring=log_loss_scorer,
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    tr_lm = -np.mean(train_loss, axis=1)
    tr_ls =  np.std(train_loss,  axis=1)
    vl_lm = -np.mean(val_loss,   axis=1)
    vl_ls =  np.std(val_loss,    axis=1)

    df_curves['logloss_train_moy'] = tr_lm
    df_curves['logloss_train_std'] = tr_ls
    df_curves['logloss_val_moy']   = vl_lm
    df_curves['logloss_val_std']   = vl_ls

    ax_loss.fill_between(train_sizes_abs, tr_lm - tr_ls, tr_lm + tr_ls,
                         alpha=0.12, color=COLORS['train'])
    ax_loss.fill_between(train_sizes_abs, vl_lm - vl_ls, vl_lm + vl_ls,
                         alpha=0.12, color=COLORS['val'])
    ax_loss.plot(train_sizes_abs, tr_lm, 'o-',
                 color=COLORS['train'], linewidth=2, markersize=6,
                 label='Entraînement')
    ax_loss.plot(train_sizes_abs, vl_lm, 's-',
                 color=COLORS['val'], linewidth=2.5, markersize=8,
                 label='Validation')

    ax_loss.set_xlabel("Taille de l'ensemble d'entraînement", fontsize=12)
    ax_loss.set_ylabel('Perte (Log Loss)', fontsize=12)
    ax_loss.set_title(f'Perte Log Loss (moy. sur {cv_folds} plis)',
                      fontsize=11, fontweight='bold')
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # ── 6e case masquée (identique SVC) ──
    axes_flat[5].set_visible(False)

    plt.tight_layout()
    save_path = f'courbes_apprentissage_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : {save_path}")

    csv_path = f'courbes_apprentissage_donnees_{timestamp}.csv'
    df_curves.to_csv(csv_path, index=False)
    print(f"Données CSV sauvegardées : {csv_path}")

    return df_curves


print("Entraînement du modèle pour les courbes d'apprentissage...")
from sklearn.base import clone
lr_for_learning = clone(best_lr)
df_curves = plot_learning_curves_lr(lr_for_learning, X_train_tfidf, y_train)

# ==================== 7. CONFUSION MATRIX ====================

print("\nGenerating confusion matrix...")

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names, yticklabels=target_names,
            ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
ax.set_ylabel('Réel',       fontsize=12, fontweight='bold')
ax.set_title('Matrice de Confusion — Régression Logistique avec TF-IDF\n',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'matrice_confusion_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Matrice de confusion → matrice_confusion_{timestamp}.png")

# ==================== 8. ROC AUC CURVE ====================

print("\nGenerating ROC AUC curve...")

fig, ax = plt.subplots(figsize=(8, 8))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color=COLORS['train'], linewidth=2,
        label=f'Courbe ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7,
        label='Classifieur aléatoire (AUC = 0.5)')
ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
ax.set_title('Courbe ROC AUC — Régression Logistique avec TF-IDF\n',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f'roc_auc_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Courbe ROC AUC → roc_auc_{timestamp}.png")

# ==================== 9. SAVE MODEL ====================

print("\nSaving model...")
joblib.dump(best_lr, 'LogisticRegression_model.pkl')
joblib.dump(tfidf,   'vectorizer.pkl')

feature_names  = tfidf.get_feature_names_out()
coefficients   = best_lr.coef_[0]

model_artifacts = {
    'classifier':    best_lr,
    'vectorizer':    tfidf,
    'best_params':   grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'test_metrics':  test_metrics,
    'feature_names': feature_names,
    'coefficients':  coefficients,
}
with open('LogisticRegression_model.pth', 'wb') as f:
    pickle.dump(model_artifacts, f)

metadata = {
    'model_type':        'RegressionLogistique',
    'best_params':       grid_search.best_params_,
    'best_cv_score':     grid_search.best_score_,
    'test_metrics':      test_metrics,
    'training_size':     len(X_train),
    'vectorizer_params': tfidf.get_params(),
    'execution_times':   time_metrics,
    'feature_count':     len(feature_names),
}
with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Save feature coefficients CSV
features_df = pd.DataFrame({
    'feature':         feature_names,
    'coefficient':     coefficients,
    'abs_coefficient': np.abs(coefficients),
}).sort_values('abs_coefficient', ascending=False)
features_df.to_csv('feature_coefficients.csv', index=False)
print("Coefficients des caractéristiques sauvegardés dans 'feature_coefficients.csv'")

print("Modèle sauvegardé :")
print("  - LogisticRegression_model.pkl")
print("  - LogisticRegression_model.pth")
print("  - vectorizer.pkl")
print("  - model_metadata.pkl")

# ==================== 10. SUMMARY ====================

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)

print(f"\nMeilleurs paramètres (GridSearch):")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")

print(f"\nMeilleur F1 macro (validation croisée) : {grid_search.best_score_:.4f}")

print("\nPerformances sur l'ensemble de test :")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nTemps d'exécution :")
for m, ts, tm in zip(time_metrics['Métrique'],
                     time_metrics['Temps (sec)'],
                     time_metrics['Temps (min)']):
    print(f"  {m}: {ts:.2f} sec ({tm:.2f} min)")

print(f"\nFichiers générés dans {results_folder} :")
print("  📊 Métriques :")
print("    - grid_search_results.csv")
print("    - meilleurs_hyperparametres.csv")
print("    - test_metrics.csv")
print("    - test_evaluation.csv")
print("    - classification_report.csv")
print("    - temps_execution.csv")
print("    - feature_coefficients.csv")
print(f"    - courbes_apprentissage_donnees_{timestamp}.csv")
print(f"    - faux_positifs_{timestamp}.csv")
print(f"    - faux_negatifs_{timestamp}.csv")
print("  📈 Visualisations :")
print(f"    - courbes_apprentissage_{timestamp}.png")
print(f"    - matrice_confusion_{timestamp}.png")
print(f"    - roc_auc_{timestamp}.png")
print("  🤖 Modèle :")
print("    - LogisticRegression_model.pkl")
print("    - LogisticRegression_model.pth")
print("    - vectorizer.pkl")
print("    - model_metadata.pkl")

# ==================== 11. INFERENCE FUNCTION ====================

def predict_comment(text, model_path='LogisticRegression_model.pth'):
    with open(model_path, 'rb') as f:
        arts = pickle.load(f)
    vec       = arts['vectorizer'].transform([text])
    pred      = arts['classifier'].predict(vec)[0]
    prob      = arts['classifier'].predict_proba(vec)[0]
    prob_hate = float(prob[1])
    return {
        'texte':                 text,
        'est_discours_haineux':  bool(pred == 1),
        'probabilite_haine':     prob_hate,
        'probabilite_non_haine': 1 - prob_hate,
        'prediction':            'Discours Haineux' if pred == 1 else 'Non-Haineux',
    }

print("\n" + "=" * 70)
print("PIPELINE TERMINÉ AVEC SUCCÈS!")
print("=" * 70)

print("\nTest de predict_comment sur le premier échantillon de test :")
sample_res = predict_comment(X_test.iloc[0])
print(f"  Texte      : {X_test.iloc[0][:100]}...")
print(f"  Prédiction : {sample_res['prediction']}")
print(f"  P(haineux) : {sample_res['probabilite_haine']:.4f}")