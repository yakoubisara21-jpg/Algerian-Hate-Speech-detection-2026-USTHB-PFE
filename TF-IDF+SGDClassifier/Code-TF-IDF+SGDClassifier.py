"""
SGDClassifier avec TF-IDF pour la classification de texte - Version finale
Support: Arabizi, Arabe algérien, Anglais, Français
Entrée: train.csv + validation.csv (combinés) pour la validation croisée, et test.csv
Sortie: Métriques, courbes d'apprentissage, courbe ROC (test uniquement)
RECHERCHE D'HYPERPARAMÈTRES CUSTOM: validation croisée K-Fold avec entraînement par époques
    (même mécanisme que l'entraînement final)
Hyperparamètres recherchés: alpha, penalty, l1_ratio, class_weight, learning_rate, eta0
Objectif: MAXIMISER LE F1 MACRO (valeur à la dernière époque, moyenne sur 5 plis)
ENTRAÎNEMENT ÉPOQUE PAR ÉPOQUE: partial_fit (SGD en ligne) AVEC MINI-LOTS ET MÉLANGE
NOMBRE D'ÉPOQUES = N_EPOCHS (complet, sans arrêt anticipé)
VALIDATION CROISÉE STRATIFIÉE 5-FOLD
ÉPOQUE OPTIMALE = MIN VAL_LOSS MOYEN SUR TOUS LES PLIS (après avoir fixé les hyperparams)
AUC = VALEUR À L'ÉPOQUE OPTIMALE (UNIQUEMENT)
DEUX GRAPHIQUES: N_EPOCHS COMPLET + GRAPHIQUE JUSQU'À L'ÉPOQUE OPTIMALE
ROC SUR LA VALIDATION CROISÉE À L'ÉPOQUE OPTIMALE
MODÈLE FINAL ENTRAÎNÉ SUR TRAIN+VAL COMPLET (OPTIMAL_EPOCHS ÉPOQUES)
TEST ÉVALUÉ UNE SEULE FOIS À LA FIN
VOCABULAIRE CONSTRUIT UNIQUEMENT SUR TRAIN+VAL (PAS DE FUITE)
GRAINE PAR PLI POUR REPRODUCTIBILITÉ
MINI-LOTS + MÉLANGE À CHAQUE ÉPOQUE
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, log_loss)
from sklearn.utils.class_weight import compute_class_weight
import warnings
import time
import joblib
import pickle
import os
import random
from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# ══════════════════════════════════════════════════════
# CONSTANTES ET CONFIGURATION
# ══════════════════════════════════════════════════════

SEED     = 42
N_EPOCHS = 10       # Nombre maximum d'époques par pli (pour la recherche custom et la CV finale)
K_FOLDS  = 5        # Nombre de plis pour la validation croisée
BATCH_SIZE = 64     # Taille des mini-lots (mini-batch)

np.random.seed(SEED)
random.seed(SEED)

COLORS = {
    'train': 'b',
    'val':   'r',
    'aux1':  '#F18F01',
    'aux2':  '#C73E1D',
    'aux3':  '#3D5A80',
}

# ── Chemins ──
DATA_DIR    = '/content/drive/MyDrive/data'
OUTPUT_DIR  = '/content/drive/MyDrive/TF-IDF+SGDClassifier'
TRAIN_PATH = f'{DATA_DIR}/kfoldsdata_sans_stopwords.csv'
TEST_PATH  = f'{DATA_DIR}/test_sans_stopwords.csv'
RESULTS_DIR = f'{OUTPUT_DIR}/Results-TF-IDF+SGDClassifier'

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 70)
print("SGDClassifier (log loss) + TF-IDF — Classification de texte")
print(f"VALIDATION CROISÉE STRATIFIÉE {K_FOLDS}-FOLD")
print(f"ÉPOQUES PAR PLI: {N_EPOCHS} (complet — pas d'arrêt anticipé)")
print(f"MINI-LOTS: taille {BATCH_SIZE} avec mélange à chaque époque")
print("RECHERCHE D'HYPERPARAMÈTRES CUSTOM: même mécanisme époque par époque")
print("OBJECTIF DE LA RECHERCHE: MAXIMISER LE F1 MACRO (meilleur atteint sur la validation)")
print("ÉPOQUE OPTIMALE (modèle final): MIN VAL_LOSS MOYEN SUR TOUS LES PLIS")
print("AUC À L'ÉPOQUE OPTIMALE UNIQUEMENT")
print("MODÈLE FINAL ENTRAÎNÉ SUR TRAIN+VAL COMPLET")
print("=" * 70)
print(f"\nDébut: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

start_time_total = time.time()

# ══════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════

print(f"\nChargement de train.csv depuis: {TRAIN_PATH}")
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Fichier manquant: {TRAIN_PATH}")

df_kfold = pd.read_csv(TRAIN_PATH, sep=';').dropna()
X_kfold  = df_kfold['text'].astype(str)
y_kfold  = df_kfold['label'].astype(int)

print(f"  train.csv    : {len(df_kfold)} échantillons")
print(f"\nDistribution classes — train+val combinés:")
for label, count in y_kfold.value_counts().items():
    print(f"  {label}: {count} ({count / len(y_kfold) * 100:.1f}%)")

print(f"\nChargement des données de test depuis: {TEST_PATH}")
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"Fichier manquant: {TEST_PATH}")

df_test = pd.read_csv(TEST_PATH, sep=';').dropna()
X_test  = df_test['text'].astype(str)
y_test  = df_test['label'].astype(int)
print(f"Chargé: {len(df_test)} échantillons de test")
print(f"\nDistribution classes — test:")
for label, count in y_test.value_counts().items():
    print(f"  {label}: {count} ({count / len(y_test) * 100:.1f}%)")

# ══════════════════════════════════════════════════════
# 2. VECTORISATION TF-IDF
# ══════════════════════════════════════════════════════

print("\nVectorisation TF-IDF...")
print("NOTE: Vocabulaire construit uniquement sur train+val (pas de fuite)")
vectorizer_start = time.time()

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words=None,
    sublinear_tf=True,
)

X_kfold_tfidf = tfidf.fit_transform(X_kfold)
X_test_tfidf  = tfidf.transform(X_test)

vectorizer_time = time.time() - vectorizer_start
print(f"Forme TF-IDF kfold : {X_kfold_tfidf.shape}")
print(f"Forme TF-IDF test  : {X_test_tfidf.shape}")
print(f"Temps vectorisation: {vectorizer_time:.2f} sec")

joblib.dump(tfidf, os.path.join(RESULTS_DIR, f'vectorizer.pkl'))
print(f"Vectoriseur sauvegardé: vectorizer.pkl")

# ══════════════════════════════════════════════════════
# 3. RECHERCHE D'HYPERPARAMÈTRES CUSTOM (maximisation du F1 macro)
# ══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RECHERCHE D'HYPERPARAMÈTRES CUSTOM — VALIDATION CROISÉE ÉPOQUE PAR ÉPOQUE")
print("Objectif: MAXIMISER LE F1 MACRO (meilleure valeur atteinte sur la validation)")
print("=" * 70)

# Grille des hyperparamètres (incluant learning_rate et eta0)
param_grid = {
    'alpha':         [1e-5, 1e-4, 1e-3, 1e-2],
    'penalty':       ['l2', 'elasticnet'],
    'l1_ratio':      [0.15, 0.5],
    'class_weight':  [None, 'balanced'],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0':          [0.001, 0.01, 0.1],
}

# On construit la liste de toutes les combinaisons
param_list = list(ParameterGrid(param_grid))
print(f"Nombre total de combinaisons: {len(param_list)}")

# Fonction utilitaire pour itérer sur les mini-lots (identique à celle utilisée plus tard)
def iterate_minibatches(X, y, batch_size, shuffle, sample_weights=None):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch_X = X[batch_idx]
        batch_y = y[batch_idx]
        batch_sw = sample_weights[batch_idx] if sample_weights is not None else None
        yield batch_X, batch_y, batch_sw

# Validation croisée stratifiée (les mêmes plis que pour la suite)
cv_outer = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
classes = np.array([0, 1])

grid_results = []
gridsearch_start = time.time()

for idx, params in enumerate(param_list):
    print(f"\nCombinaison {idx+1}/{len(param_list)}: {params}")
    fold_f1_at_end = []   # F1 à la dernière époque, par pli

    for fold_idx, (train_idx, val_idx) in enumerate(cv_outer.split(X_kfold_tfidf, y_kfold)):
        np.random.seed(SEED + fold_idx)
        random.seed(SEED + fold_idx)

        X_tr = X_kfold_tfidf[train_idx]
        y_tr = y_kfold.iloc[train_idx].values
        X_vl = X_kfold_tfidf[val_idx]
        y_vl = y_kfold.iloc[val_idx].values

        use_balanced = (params.get('class_weight') == 'balanced')
        if use_balanced:
            cw = compute_class_weight('balanced', classes=classes, y=y_tr)
            class_weight_map = {c: w for c, w in zip(classes, cw)}
            sample_weights_tr = np.array([class_weight_map[c] for c in y_tr])
        else:
            sample_weights_tr = None

        model = SGDClassifier(
            loss='log_loss',
            alpha=params['alpha'],
            penalty=params['penalty'],
            l1_ratio=params.get('l1_ratio', 0.15),
            learning_rate=params['learning_rate'],
            eta0=params['eta0'],
            class_weight=None,
            max_iter=1,
            tol=None,
            warm_start=False,
            random_state=SEED + fold_idx,
            n_jobs=1,
        )

        # Entraînement complet — N_EPOCHS époques sans interruption
        for epoch in range(N_EPOCHS):
            for batch_X, batch_y, batch_sw in iterate_minibatches(
                    X_tr, y_tr, BATCH_SIZE, shuffle=True,
                    sample_weights=sample_weights_tr):
                model.partial_fit(batch_X, batch_y,
                                  classes=classes, sample_weight=batch_sw)

        # Évaluation UNIQUE à la fin des N_EPOCHS époques
        y_vl_pred = model.predict(X_vl)
        f1_final = f1_score(y_vl, y_vl_pred, average='macro', zero_division=0)
        fold_f1_at_end.append(f1_final)

    # Moyenne du F1 final sur les 5 plis → score de cette combinaison
    mean_f1 = np.mean(fold_f1_at_end)
    grid_results.append({
        'params':        params,
        'mean_best_f1':  mean_f1,          # nom conservé pour compatibilité
        'fold_best_f1':  fold_f1_at_end,
    })
    print(f"  -> F1 macro moyen (fin des {N_EPOCHS} époques) = {mean_f1:.6f}")

best_grid_result = max(grid_results, key=lambda x: x['mean_best_f1'])
best_params      = best_grid_result['params']
best_cv_score    = best_grid_result['mean_best_f1']

gridsearch_time = time.time() - gridsearch_start

print("\n" + "=" * 70)
print("RÉSULTATS DE LA RECHERCHE CUSTOM")
print("=" * 70)
print(f"Meilleurs paramètres : {best_params}")
print(f"Meilleur F1 macro moyen (validation) : {best_cv_score:.6f}")
print(f"Temps de la recherche : {gridsearch_time:.2f} sec")

# Sauvegarde des résultats de la grille
grid_results_df = pd.DataFrame([
    {**res['params'], 'mean_best_f1': res['mean_best_f1']}
    for res in grid_results
])
grid_results_df.to_csv(
    os.path.join(RESULTS_DIR, f'grid_search_results_{timestamp}.csv'), index=False)
pd.DataFrame([best_params]).to_csv(
    os.path.join(RESULTS_DIR, f'meilleurs_hyperparametres.csv'), index=False)
print(f"Résultats recherche → grid_search_results_{timestamp}.csv")

# Indicateur pour l'équilibrage des classes (sera utilisé plus tard)
USE_BALANCED = (best_params.get('class_weight') == 'balanced')

# ══════════════════════════════════════════════════════
# 4. ENTRAÎNEMENT ÉPOQUE PAR ÉPOQUE — K-FOLD AVEC MINI-LOTS
#    (avec les meilleurs hyperparamètres, pour tracer les courbes et trouver l'époque optimale)
# ══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"ENTRAÎNEMENT ÉPOQUE PAR ÉPOQUE — VALIDATION CROISÉE {K_FOLDS}-FOLD")
print(f"Époques par pli: {N_EPOCHS} (complet — pas d'arrêt anticipé)")
print(f"Mini-lots: taille {BATCH_SIZE} — mélange à chaque époque")
print(f"Meilleurs params: {best_params}")
print("=" * 70)

cv_epoch = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

fold_metrics = {
    split: {m: np.full((K_FOLDS, N_EPOCHS), np.nan)
            for m in ['accuracy', 'f1', 'precision', 'recall', 'log_loss']}
    for split in ['train', 'val']
}

fold_val_probs  = [[None] * N_EPOCHS for _ in range(K_FOLDS)]
fold_val_labels = [None] * K_FOLDS

epoch_cv_start = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(
        cv_epoch.split(X_kfold_tfidf, y_kfold)):

    np.random.seed(SEED + fold_idx)
    random.seed(SEED + fold_idx)
    print(f"\n  Pli {fold_idx + 1}/{K_FOLDS} (graine: {SEED + fold_idx})")

    X_tr = X_kfold_tfidf[train_idx]
    y_tr = y_kfold.iloc[train_idx].values
    X_vl = X_kfold_tfidf[val_idx]
    y_vl = y_kfold.iloc[val_idx].values
    fold_val_labels[fold_idx] = y_vl.tolist()

    if USE_BALANCED:
        cw = compute_class_weight('balanced', classes=classes, y=y_tr)
        class_weight_map = {c: w for c, w in zip(classes, cw)}
        sample_weights_tr = np.array([class_weight_map[c] for c in y_tr])
    else:
        sample_weights_tr = None

    model = SGDClassifier(
        loss='log_loss',
        alpha=best_params['alpha'],
        penalty=best_params['penalty'],
        l1_ratio=best_params.get('l1_ratio', 0.15),
        learning_rate=best_params['learning_rate'],
        eta0=best_params['eta0'],
        class_weight=None,
        max_iter=1,
        tol=None,
        warm_start=False,
        random_state=SEED + fold_idx,
        n_jobs=1,
    )

    for epoch in range(N_EPOCHS):
        for batch_X, batch_y, batch_sw in iterate_minibatches(
                X_tr, y_tr, BATCH_SIZE, shuffle=True, sample_weights=sample_weights_tr):
            model.partial_fit(batch_X, batch_y, classes=classes, sample_weight=batch_sw)

        y_tr_pred = model.predict(X_tr)
        y_tr_prob = model.predict_proba(X_tr)
        fold_metrics['train']['accuracy'][fold_idx, epoch]   = accuracy_score(y_tr, y_tr_pred)
        fold_metrics['train']['f1'][fold_idx, epoch]         = f1_score(y_tr, y_tr_pred)
        fold_metrics['train']['precision'][fold_idx, epoch]  = precision_score(y_tr, y_tr_pred)
        fold_metrics['train']['recall'][fold_idx, epoch]     = recall_score(y_tr, y_tr_pred)
        fold_metrics['train']['log_loss'][fold_idx, epoch]   = log_loss(y_tr, y_tr_prob)

        y_vl_pred = model.predict(X_vl)
        y_vl_prob = model.predict_proba(X_vl)
        val_loss  = log_loss(y_vl, y_vl_prob)

        fold_metrics['val']['accuracy'][fold_idx, epoch]     = accuracy_score(y_vl, y_vl_pred)
        fold_metrics['val']['f1'][fold_idx, epoch]           = f1_score(y_vl, y_vl_pred)
        fold_metrics['val']['precision'][fold_idx, epoch]    = precision_score(y_vl, y_vl_pred)
        fold_metrics['val']['recall'][fold_idx, epoch]       = recall_score(y_vl, y_vl_pred)
        fold_metrics['val']['log_loss'][fold_idx, epoch]     = val_loss

        fold_val_probs[fold_idx][epoch] = y_vl_prob[:, 1].tolist()

        print(f"    Époque {epoch + 1:2d} | "
              f"Train Loss={fold_metrics['train']['log_loss'][fold_idx, epoch]:.4f} "
              f"F1={fold_metrics['train']['f1'][fold_idx, epoch]:.4f} | "
              f"Val Loss={val_loss:.4f} "
              f"F1={fold_metrics['val']['f1'][fold_idx, epoch]:.4f}")

epoch_cv_time = time.time() - epoch_cv_start
print(f"\nTemps validation croisée par époques: {epoch_cv_time:.2f} sec")

# ══════════════════════════════════════════════════════
# 5. AGRÉGATION DES MÉTRIQUES PAR ÉPOQUE
# ══════════════════════════════════════════════════════

metrics_list = ['accuracy', 'f1', 'precision', 'recall', 'log_loss']
agg = {'epoque': list(range(1, N_EPOCHS + 1))}

for split in ['train', 'val']:
    for m in metrics_list:
        arr = fold_metrics[split][m]
        agg[f'{split}_{m}_moy']    = [np.nanmean(arr[:, e]) for e in range(N_EPOCHS)]
        agg[f'{split}_{m}_std']    = [np.nanstd(arr[:, e])  for e in range(N_EPOCHS)]
        agg[f'{split}_{m}_n_plis'] = [int((~np.isnan(arr[:, e])).sum()) for e in range(N_EPOCHS)]

df_agg = pd.DataFrame(agg)
df_agg.to_csv(os.path.join(RESULTS_DIR,
              f'kfold_metriques_agregees_{timestamp}.csv'), index=False)
print(f"\nMétriques agrégées → kfold_metriques_agregees_{timestamp}.csv")

# Époque optimale = argmin(val_loss moyen) (celle qui minimise la perte)
OPTIMAL_EPOCHS = int(np.argmin(df_agg['val_log_loss_moy'])) + 1
print(f"\nÉpoque optimale (min val_loss moyen): {OPTIMAL_EPOCHS} "
      f"(val_loss = {df_agg['val_log_loss_moy'][OPTIMAL_EPOCHS - 1]:.6f})")

# ══════════════════════════════════════════════════════
# 6. GRAPHIQUES DES COURBES D'APPRENTISSAGE (inchangé)
# ══════════════════════════════════════════════════════

def plot_kfold_metrics(df_plot, k_folds, save_path, optimal_epoch=None):
    epochs = df_plot['epoque']
    pairs = [
        ('train_log_loss_moy',  'val_log_loss_moy',  'train_log_loss_std',  'val_log_loss_std',  'Perte (Log Loss)'),
        ('train_accuracy_moy',  'val_accuracy_moy',  'train_accuracy_std',  'val_accuracy_std',  'Exactitude'),
        ('train_f1_moy',        'val_f1_moy',        'train_f1_std',        'val_f1_std',        'F1-Score'),
        ('train_precision_moy', 'val_precision_moy', 'train_precision_std', 'val_precision_std', 'Précision'),
        ('train_recall_moy',    'val_recall_moy',    'train_recall_std',    'val_recall_std',    'Rappel'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Métriques validation croisée pour SGDClassifier avec TF-IDF\n', fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()
    for i, (tr_col, vl_col, tr_std, vl_std, label) in enumerate(pairs):
        tr_m = df_plot[tr_col].values
        vl_m = df_plot[vl_col].values
        tr_s = df_plot[tr_std].values
        vl_s = df_plot[vl_std].values
        axes_flat[i].fill_between(epochs, tr_m - tr_s, tr_m + tr_s, alpha=0.12, color='b')
        axes_flat[i].fill_between(epochs, vl_m - vl_s, vl_m + vl_s, alpha=0.12, color='r')
        axes_flat[i].plot(epochs, tr_m, 'o-', color='b', linewidth=2, markersize=6, label='Entraînement')
        axes_flat[i].plot(epochs, vl_m, 's-', color='r', linewidth=2.5, markersize=8, label=f'Validation ({k_folds}-Fold)')
        if optimal_epoch is not None and optimal_epoch in epochs.values:
            axes_flat[i].axvline(x=optimal_epoch, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                                 label=f'Époque optimale = {optimal_epoch}')
        axes_flat[i].set_xlabel('Époque', fontsize=12)
        axes_flat[i].set_ylabel(label,    fontsize=12)
        axes_flat[i].set_title(f'{label} (moy. sur {k_folds} plis)', fontsize=11, fontweight='bold')
        axes_flat[i].set_xticks(epochs)
        axes_flat[i].legend(fontsize=9)
        axes_flat[i].grid(True, alpha=0.3)
    axes_flat[5].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé: {save_path}")

print("\nGénération des graphiques de courbes d'apprentissage...")
plot_kfold_metrics(df_agg, K_FOLDS,
    save_path=os.path.join(RESULTS_DIR, f'kfold_metriques_complet_{timestamp}.png'),
    optimal_epoch=OPTIMAL_EPOCHS)
plot_kfold_metrics(df_agg.iloc[:OPTIMAL_EPOCHS].reset_index(drop=True), K_FOLDS,
    save_path=os.path.join(RESULTS_DIR, f'kfold_metriques_optimal_{timestamp}.png'),
    optimal_epoch=None)

# ══════════════════════════════════════════════════════
# 7. ENTRAÎNEMENT DU MODÈLE FINAL (identique, avec best_params)
# ══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ENTRAÎNEMENT DU MODÈLE FINAL SUR TRAIN+VAL COMPLET")
print(f"NOMBRE FIXE D'ÉPOQUES: {OPTIMAL_EPOCHS} (issu du K-Fold — min val_loss moyen)")
print(f"Mini-lots: taille {BATCH_SIZE} — mélange à chaque époque")
print("=" * 70)
print(f"Entraînement sur {len(X_kfold)} échantillons")

np.random.seed(SEED)
random.seed(SEED)

if USE_BALANCED:
    cw_final = compute_class_weight('balanced', classes=classes, y=y_kfold.values)
    class_weight_map_final = {c: w for c, w in zip(classes, cw_final)}
    sample_weights_final = np.array([class_weight_map_final[c] for c in y_kfold.values])
else:
    sample_weights_final = None

final_start = time.time()

final_model = SGDClassifier(
    loss='log_loss',
    alpha=best_params['alpha'],
    penalty=best_params['penalty'],
    l1_ratio=best_params.get('l1_ratio', 0.15),
    learning_rate=best_params['learning_rate'],
    eta0=best_params['eta0'],
    class_weight=None,
    max_iter=1,
    tol=None,
    warm_start=False,
    random_state=SEED,
    n_jobs=1,
)

final_history = []
print()
for epoch in range(OPTIMAL_EPOCHS):
    for batch_X, batch_y, batch_sw in iterate_minibatches(
            X_kfold_tfidf, y_kfold.values, BATCH_SIZE, shuffle=True,
            sample_weights=sample_weights_final):
        final_model.partial_fit(batch_X, batch_y, classes=classes, sample_weight=batch_sw)

    y_tr_pred = final_model.predict(X_kfold_tfidf)
    y_tr_prob = final_model.predict_proba(X_kfold_tfidf)
    tr_loss   = log_loss(y_kfold.values, y_tr_prob)
    tr_acc    = accuracy_score(y_kfold.values, y_tr_pred)
    tr_f1     = f1_score(y_kfold.values, y_tr_pred)
    tr_prec   = precision_score(y_kfold.values, y_tr_pred)
    tr_rec    = recall_score(y_kfold.values, y_tr_pred)

    final_history.append({
        'epoch':          epoch + 1,
        'train_loss':     tr_loss,
        'train_accuracy': tr_acc,
        'train_f1':       tr_f1,
        'train_precision':tr_prec,
        'train_recall':   tr_rec,
    })
    print(f"ÉPOQUE {epoch + 1}/{OPTIMAL_EPOCHS} — "
          f"Perte: {tr_loss:.4f} | Acc: {tr_acc:.4f} | "
          f"Prec: {tr_prec:.4f} | Rappel: {tr_rec:.4f} | F1: {tr_f1:.4f}")

final_time    = time.time() - final_start
total_time    = time.time() - start_time_total
print(f"\nEntraînement final terminé: {final_time:.2f}s ({final_time / 60:.2f} min)")

# Courbes d'entraînement final (inchangées)
df_final_hist = pd.DataFrame(final_history)
df_final_hist.to_csv(
    os.path.join(RESULTS_DIR, f'courbes_metriques_finales_donnees_{timestamp}.csv'), index=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Métriques d\'entraînement final pour SGDClassifier avec TF-IDF\n', fontsize=14, fontweight='bold')
axes_flat = axes.flatten()
series_final = [
    ('train_loss',      'Perte (Log Loss)', 'Perte (entraînement final)'),
    ('train_accuracy',  'Exactitude',       'Exactitude (entraînement final)'),
    ('train_f1',        'F1-Score',         'F1-Score (entraînement final)'),
    ('train_precision', 'Précision',        'Précision (entraînement final)'),
    ('train_recall',    'Rappel',           'Rappel (entraînement final)'),
]
for i, (col, ylabel, title) in enumerate(series_final):
    axes_flat[i].plot(df_final_hist['epoch'], df_final_hist[col],
                      'o-', color=COLORS['train'], linewidth=2, markersize=6, label='Entraînement')
    axes_flat[i].set_xlabel('Époque', fontsize=12)
    axes_flat[i].set_ylabel(ylabel,   fontsize=12)
    axes_flat[i].set_title(title,     fontsize=13, fontweight='bold')
    axes_flat[i].set_xticks(df_final_hist['epoch'])
    axes_flat[i].legend(fontsize=9)
    axes_flat[i].grid(True, alpha=0.3)
axes_flat[5].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'courbes_metriques_finales_{timestamp}.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Courbes entraînement final → courbes_metriques_finales_{timestamp}.png")

# ══════════════════════════════════════════════════════
# 8. ÉVALUATION SUR LE TEST (inchangée)
# ══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ÉVALUATION SUR L'ENSEMBLE DE TEST (seuil 0.5)")
print("=" * 70)

inference_start = time.time()
y_pred = final_model.predict(X_test_tfidf)
y_prob = final_model.predict_proba(X_test_tfidf)[:, 1]
inference_time = time.time() - inference_start

accuracy_test  = accuracy_score(y_test, y_pred)
f1_test        = f1_score(y_test, y_pred)
recall_test    = recall_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred)
roc_auc_test   = roc_auc_score(y_test, y_prob)

test_metrics = {
    'Exactitude': accuracy_test,
    'F1-Score':   f1_test,
    'Rappel':     recall_test,
    'Précision':  precision_test,
    'ROC AUC':    roc_auc_test
}

print(f"\nPERFORMANCE SUR L'ENSEMBLE DE TEST:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n" + classification_report(y_test, y_pred,
      target_names=['0', '1']))

pd.DataFrame([test_metrics]).to_csv(
    os.path.join(RESULTS_DIR, f'test_metrics_{timestamp}.csv'), index=False)
pd.DataFrame(classification_report(y_test, y_pred,
             target_names=['0', '1'],
             output_dict=True)).transpose().to_csv(
    os.path.join(RESULTS_DIR, f'classification_report_{timestamp}.csv'))

eval_df = pd.DataFrame({
    'texte':                 X_test.values,
    'vrai_label':            y_test.values,
    'label_predit':          y_pred,
    'est_correct':           y_test.values == y_pred,
    'probabilite_haine':     y_prob,
    'probabilite_non_haine': 1 - y_prob,
})
eval_df.to_csv(os.path.join(RESULTS_DIR, f'test_evaluation_{timestamp}.csv'),
               index=False, encoding='utf-8-sig')
print(f"Évaluation détaillée → test_evaluation_{timestamp}.csv")

fp_df = eval_df[(eval_df['vrai_label'] == 0) & (eval_df['label_predit'] == 1)]
fn_df = eval_df[(eval_df['vrai_label'] == 1) & (eval_df['label_predit'] == 0)]
if len(fp_df) > 0:
    fp_df.to_csv(os.path.join(RESULTS_DIR, f'faux_positifs_{timestamp}.csv'), index=False, encoding='utf-8-sig')
    print(f"Faux positifs: {len(fp_df)} → faux_positifs_{timestamp}.csv")
if len(fn_df) > 0:
    fn_df.to_csv(os.path.join(RESULTS_DIR, f'faux_negatifs_{timestamp}.csv'), index=False, encoding='utf-8-sig')
    print(f"Faux négatifs: {len(fn_df)} → faux_negatifs_{timestamp}.csv")

# ══════════════════════════════════════════════════════
# 9. MATRICE DE CONFUSION (inchangée)
# ══════════════════════════════════════════════════════

print("\nGénération de la matrice de confusion...")
target_names = ['0', '1']
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
    os.path.join(RESULTS_DIR, f'matrice_confusion_{timestamp}.csv'))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names, yticklabels=target_names,
            ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
ax.set_ylabel('Réel',       fontsize=12, fontweight='bold')
ax.set_title('Matrice de Confusion — SGDClassifier avec TF-IDF', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'matrice_confusion_{timestamp}.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Matrice de confusion → matrice_confusion_{timestamp}.png")
if len(target_names) == 2:
    tn, fp_n, fn_n, tp = cm.ravel()
    print(f"VN: {tn} | FP: {fp_n} | FN: {fn_n} | VP: {tp}")

# ══════════════════════════════════════════════════════
# 10. COURBE ROC (inchangée)
# ══════════════════════════════════════════════════════

print("\nGénération de la courbe ROC (test)...")
fpr_t, tpr_t, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr_t, tpr_t, color=COLORS['train'], linewidth=2,
        label=f'Courbe ROC (AUC = {roc_auc_test:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Classifieur aléatoire (AUC = 0.5)')
ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
ax.set_title('Courbe ROC AUC — SGDClassifier avec TF-IDF\n', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'roc_test_{timestamp}.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Courbe ROC (test) → roc_test_{timestamp}.png")

# ══════════════════════════════════════════════════════
# 11. COEFFICIENTS DES FEATURES (inchangé)
# ══════════════════════════════════════════════════════

print("\nCaractéristiques importantes...")
feature_names = tfidf.get_feature_names_out()
coefficients  = final_model.coef_[0]
features_df = pd.DataFrame({
    'feature':         feature_names,
    'coefficient':     coefficients,
    'abs_coefficient': np.abs(coefficients),
}).sort_values('abs_coefficient', ascending=False)
features_df.to_csv(os.path.join(RESULTS_DIR, f'feature_coefficients_{timestamp}.csv'), index=False)

# ══════════════════════════════════════════════════════
# 12. SAUVEGARDE MODÈLE ET INFOS
# ══════════════════════════════════════════════════════

print("\nSauvegarde du modèle...")
joblib.dump(final_model, os.path.join(RESULTS_DIR, f'SGDClassifier_model.pkl'))
model_artifacts = {
    'classifier':         final_model,
    'vectorizer':         tfidf,
    'best_params':        best_params,
    'best_cv_score':      best_cv_score,
    'optimal_epochs':     OPTIMAL_EPOCHS,
    'test_metrics':       test_metrics,
    'feature_names':      feature_names,
    'coefficients':       coefficients,
    'date_sauvegarde':    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(os.path.join(RESULTS_DIR, f'SGDClassifier_artifacts.pkl'), 'wb') as f:
    pickle.dump(model_artifacts, f)

runtime_info = {
    'temps_vectorisation_sec':        vectorizer_time,
    'temps_gridsearch_sec':           gridsearch_time,
    'temps_cv_epoques_sec':           epoch_cv_time,
    'temps_entrainement_final_sec':   final_time,
    'temps_inference_sec':            inference_time,
    'temps_total_sec':                total_time,
    'temps_total_min':                total_time / 60,
    'epoques_optimales':              OPTIMAL_EPOCHS,
    'taille_mini_lot':                BATCH_SIZE,
    'seuil_utilise':                  0.5,
    'k_folds':                        K_FOLDS,
    'n_epochs_max':                   N_EPOCHS,
    'horodatage':                     timestamp,
    'objectif_gridsearch': 'max_f1_macro (dernière époque, moyenne 5-Fold)',
    'critere_epoque_optimale':        'min_val_loss_moyen',
    'arret_anticipe':                 'non',
}
pd.DataFrame([runtime_info]).to_csv(
    os.path.join(RESULTS_DIR, f'info_execution_{timestamp}.csv'), index=False)

print(f"Modèle      → SGDClassifier_model_{timestamp}.pkl")
print(f"Artefacts   → SGDClassifier_artifacts_{timestamp}.pkl")
print(f"Vectoriseur → vectorizer_{timestamp}.pkl")
print(f"Exécution   → info_execution_{timestamp}.csv")

# ══════════════════════════════════════════════════════
# 13. FONCTION D'INFÉRENCE (inchangée)
# ══════════════════════════════════════════════════════

def predict_comment(text, artifacts_path):
    with open(artifacts_path, 'rb') as f:
        arts = pickle.load(f)
    vec_text    = arts['vectorizer'].transform([text])
    prediction  = arts['classifier'].predict(vec_text)[0]
    probability = arts['classifier'].predict_proba(vec_text)[0]
    return {
        'texte':                 text,
        'est_discours_haineux':  bool(prediction == 1),
        'probabilite_haine':     float(probability[1]),
        'probabilite_non_haine': float(probability[0]),
        'prediction':            'Discours Haineux' if prediction == 1 else 'Non-Haineux',
    }

# ══════════════════════════════════════════════════════
# 14. RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)

print(f"\nMeilleurs paramètres (recherche custom époque par époque, objectif max F1 macro):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

print(f"\nMeilleur F1 macro moyen (validation) : {best_cv_score:.6f}")
print(f"Époque optimale (min val_loss moyen — {K_FOLDS}-Fold): {OPTIMAL_EPOCHS}")

print(f"\nPERFORMANCE SUR LE TEST (seuil 0.5):")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

print(f"\nCONFIGURATION:")
print(f"  Objectif recherche:            maximiser le F1 macro (validation)")
print(f"  Critère époque optimale:       min val_loss moyen ({K_FOLDS}-Fold)")
print(f"  Arrêt anticipé:                désactivé (N_EPOCHS complet par pli)")
print(f"  Mini-lots:                     taille {BATCH_SIZE} avec mélange à chaque époque")
print(f"  Validation croisée:            Stratified {K_FOLDS}-Fold")
print(f"  AUC reportée:                  à l'époque optimale uniquement (CV)")
print(f"  Vocabulaire TF-IDF:            train+val uniquement (pas de fuite)")
print(f"  Graine par pli:                {SEED} + fold_idx")

print(f"\nTEMPS:")
print(f"  Vectorisation:             {vectorizer_time:.2f}s")
print(f"  Recherche custom:          {gridsearch_time:.2f}s ({gridsearch_time/60:.2f} min)")
print(f"  CV par époques:            {epoch_cv_time:.2f}s ({epoch_cv_time/60:.2f} min)")
print(f"  Entraînement final:        {final_time:.2f}s ({final_time/60:.2f} min)")
print(f"  Total:                     {total_time:.2f}s ({total_time/60:.2f} min)")

print(f"\nFICHIERS GÉNÉRÉS dans {RESULTS_DIR}:")
print(f"  📊 Métriques:")
print(f"    - grid_search_results_{timestamp}.csv")
print(f"    - meilleurs_hyperparametres_{timestamp}.csv")
print(f"    - kfold_metriques_agregees_{timestamp}.csv")
print(f"    - test_metrics_{timestamp}.csv")
print(f"    - test_evaluation_{timestamp}.csv")
print(f"    - classification_report_{timestamp}.csv")
print(f"    - feature_coefficients_{timestamp}.csv")
print(f"    - courbes_metriques_finales_donnees_{timestamp}.csv")
print(f"    - info_execution_{timestamp}.csv")
print(f"  📈 Visualisations:")
print(f"    - kfold_metriques_complet_{timestamp}.png")
print(f"    - kfold_metriques_optimal_{timestamp}.png")
print(f"    - courbes_metriques_finales_{timestamp}.png")
print(f"    - matrice_confusion_{timestamp}.png")
print(f"    - roc_test_{timestamp}.png")
print(f"  🤖 Modèle:")
print(f"    - SGDClassifier_model_{timestamp}.pkl")
print(f"    - SGDClassifier_artifacts_{timestamp}.pkl")
print(f"    - vectorizer_{timestamp}.pkl")

print("\n" + "=" * 70)
print("PIPELINE TERMINÉ AVEC SUCCÈS!")
print("=" * 70)

print("\nTest de predict_comment sur le premier échantillon de test:")
sample_res = predict_comment(
    X_test.iloc[0],
    os.path.join(RESULTS_DIR, f'SGDClassifier_artifacts.pkl')
)
print(f"  Texte      : {X_test.iloc[0][:100]}...")
print(f"  Prédiction : {sample_res['prediction']}")
print(f"  P(haineux) : {sample_res['probabilite_haine']:.4f}")