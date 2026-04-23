"""
SGD v3 — Hate Speech Detection
Improvements over baseline:
  - char(2-4) + word(1-2) TF-IDF combined features
  - Mini-batch partial_fit with per-sample balanced class weights
  - Dynamic y-axis scaling according to actual values
"""
import time
import pickle
import torch
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score, log_loss
)

# Start total execution timer
total_start_time = time.time()

# ── change these paths if needed ──────────────────────
TRAIN_PATH = 'data/train.csv'
VAL_PATH   = 'data/validation.csv'
TEST_PATH  = 'data/test.csv'
OUT_DIR    = ''          # '' = current directory
# ──────────────────────────────────────────────────────

def p(name): return OUT_DIR + name

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH, sep=';').dropna()
val_df   = pd.read_csv(VAL_PATH,   sep=';').dropna()
test_df  = pd.read_csv(TEST_PATH,  sep=';').dropna()

X_train, y_train = train_df['text'].astype(str).values, train_df['label'].astype(int).values
X_val,   y_val   = val_df['text'].astype(str).values,   val_df['label'].astype(int).values
X_test,  y_test  = test_df['text'].astype(str).values,  test_df['label'].astype(int).values
print(f"Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

# ─────────────────────────────────────────────
# 2. VECTORIZER: char(2-4) + word(1-2) n-grams
# ─────────────────────────────────────────────
print("Fitting TF-IDF (char + word)...")
vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                           max_features=50000, sublinear_tf=True, min_df=3, max_df=0.95)
vec_word = TfidfVectorizer(analyzer='word',    ngram_range=(1,2),
                           max_features=30000, sublinear_tf=True, min_df=3, max_df=0.95)
Xtr  = sp.hstack([vec_char.fit_transform(X_train), vec_word.fit_transform(X_train)], format='csr')
Xv   = sp.hstack([vec_char.transform(X_val),       vec_word.transform(X_val)],       format='csr')
Xte  = sp.hstack([vec_char.transform(X_test),      vec_word.transform(X_test)],      format='csr')
print(f"Feature shape: {Xtr.shape}")

# balanced sample weights for partial_fit
cw  = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cwd = {c: w for c, w in zip(np.unique(y_train), cw)}
sample_weights = np.array([cwd[y] for y in y_train])

# ─────────────────────────────────────────────
# 3. HYPER-PARAM SEARCH
# ─────────────────────────────────────────────
print("\nGrid search (15 passes each)...")
configs = [
    ('log_loss',       1e-4), ('log_loss',       3e-4),
    ('log_loss',       5e-4), ('log_loss',       1e-3),
    ('modified_huber', 1e-4), ('modified_huber', 3e-4),
    ('modified_huber', 5e-4), ('modified_huber', 1e-3),
]
search_rows, best_score, best_params = [], -1, {}

for loss, alpha in configs:
    clf_tmp = SGDClassifier(loss=loss, alpha=alpha, penalty='l2',
                            max_iter=15, tol=None, random_state=42, n_jobs=-1)
    clf_tmp.fit(Xtr, y_train, sample_weight=sample_weights)
    vp = clf_tmp.predict(Xv); tp = clf_tmp.predict(Xtr)
    vf1   = f1_score(y_val,   vp, average='macro', zero_division=0)
    tf1   = f1_score(y_train, tp, average='macro', zero_division=0)
    gap   = tf1 - vf1
    score = vf1 - 0.3 * gap
    row   = dict(loss=loss, alpha=alpha, train_f1=tf1, val_f1=vf1, gap=gap,
                 composite_score=score,
                 val_accuracy  = accuracy_score(y_val, vp),
                 val_precision = precision_score(y_val, vp, average='macro', zero_division=0),
                 val_recall    = recall_score(y_val,   vp, average='macro', zero_division=0),
                 selected=False)
    search_rows.append(row)
    print(f"  {loss:<16} alpha={alpha:.0e} | TrainF1={tf1:.4f}  ValF1={vf1:.4f}"
          f"  Gap={gap:.4f}  Score={score:.4f}")
    if score > best_score:
        best_score, best_params = score, {'loss': loss, 'alpha': alpha}

for r in search_rows:
    if r['loss'] == best_params['loss'] and r['alpha'] == best_params['alpha']:
        r['selected'] = True

pd.DataFrame(search_rows).to_csv(p('v3_param_search.csv'), index=False)
print(f"\nBest → {best_params}  Score={best_score:.4f}")

# ─────────────────────────────────────────────
# 4. EPOCH TRAINING — mini-batch partial_fit
# ─────────────────────────────────────────────
N_EPOCHS  = 4
BATCH     = 512
classes   = np.array([0, 1])

clf = SGDClassifier(
    loss=best_params['loss'], alpha=best_params['alpha'],
    penalty='l2', random_state=42, n_jobs=-1, learning_rate='optimal'
)

history = []

print(f"\nTraining (batch={BATCH})...")
# Start timing
training_start_time = time.time()
for epoch in range(1, N_EPOCHS+1):
    idx    = np.random.RandomState(epoch).permutation(Xtr.shape[0])
    Xtr_sh = Xtr[idx]; ytr_sh = y_train[idx]; sw_sh = sample_weights[idx]

    for start in range(0, Xtr.shape[0], BATCH):
        end = min(start+BATCH, Xtr.shape[0])
        clf.partial_fit(Xtr_sh[start:end], ytr_sh[start:end],
                        classes=classes, sample_weight=sw_sh[start:end])

    tr_pred  = clf.predict(Xtr);       vl_pred  = clf.predict(Xv)
    tr_proba = clf.predict_proba(Xtr); vl_proba = clf.predict_proba(Xv)
    vl_f1    = f1_score(y_val,   vl_pred, average='macro', zero_division=0)
    tr_f1    = f1_score(y_train, tr_pred, average='macro', zero_division=0)

    history.append(dict(
        epoch=epoch,
        train_accuracy  = accuracy_score(y_train, tr_pred),
        val_accuracy    = accuracy_score(y_val,   vl_pred),
        train_precision = precision_score(y_train, tr_pred, average='macro', zero_division=0),
        val_precision   = precision_score(y_val,   vl_pred, average='macro', zero_division=0),
        train_recall    = recall_score(y_train, tr_pred, average='macro', zero_division=0),
        val_recall      = recall_score(y_val,   vl_pred, average='macro', zero_division=0),
        train_f1        = tr_f1,
        val_f1          = vl_f1,
        train_loss      = log_loss(y_train, tr_proba),
        val_loss        = log_loss(y_val,   vl_proba),
    ))

    if epoch % 10 == 0 or epoch == 1:
        print(f"  ep{epoch:3d} | Train={tr_f1:.4f}  Val={vl_f1:.4f}")

hist_df = pd.DataFrame(history)
# Calculate training time
training_end_time = time.time()
training_time = training_end_time - training_start_time
print(f"\nTotal training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
hist_df.to_csv(p('v3_train_history.csv'), index=False)

best_epoch = hist_df['val_f1'].idxmax() + 1
best_val_f1 = hist_df['val_f1'].max()

# Save model and vectorizers in .pth format
print("\nSaving model and vectorizers...")

# Save SGD model
torch.save({
    'model_state_dict': clf.__dict__,
    'best_params': best_params,
    'best_epoch': best_epoch,
    'best_val_f1': best_val_f1,
}, p('sgd_model.pth'))

# Save vectorizers using pickle (since they're not PyTorch models)
with open(p('vec_char.pkl'), 'wb') as f:
    pickle.dump(vec_char, f)
with open(p('vec_word.pkl'), 'wb') as f:
    pickle.dump(vec_word, f)

print(f"Model saved to: {p('sgd_model.pth')}")
print(f"Vectorizers saved to: {p('vec_char.pkl')} and {p('vec_word.pkl')}")

# ─────────────────────────────────────────────
# 5. TEST EVALUATION
# ─────────────────────────────────────────────
y_pred  = clf.predict(Xte)
y_proba = clf.predict_proba(Xte)
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:,1])
roc_auc_val = auc(fpr, tpr)

test_metrics = dict(
    accuracy  = accuracy_score(y_test, y_pred),
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0),
    recall    = recall_score(y_test, y_pred, average='macro', zero_division=0),
    f1        = f1_score(y_test, y_pred, average='macro', zero_division=0),
    log_loss  = log_loss(y_test, y_proba),
    roc_auc   = roc_auc_val,
)
pd.DataFrame([test_metrics]).to_csv(p('v3_test_metrics.csv'), index=False)

eval_df = pd.DataFrame({
    'text':                 test_df['text'].tolist(),
    'true_label':           y_test.tolist(),
    'predicted_label':      y_pred.tolist(),
    'is_correct':           (y_pred == y_test).tolist(),
    'probability_non_hate': y_proba[:,0].tolist(),
    'probability_hate':     y_proba[:,1].tolist(),
})
eval_df.to_csv(p('v3_test_evaluation.csv'), index=False)

print(f"\nBest epoch: {best_epoch} (Val F1={best_val_f1:.4f})")
print(f"Test → Acc={test_metrics['accuracy']:.4f}  F1={test_metrics['f1']:.4f}"
      f"  AUC={test_metrics['roc_auc']:.4f}")

# ─────────────────────────────────────────────
# STYLE - WHITE BACKGROUND
# ─────────────────────────────────────────────
DARK_BG='white'
PANEL_BG='white'
ACCENT='#2e7d32'
TRAIN_C='#1f77b4'
VAL_C='#ff7f0e'
GRID_C='#cccccc'
TEXT_C='black'
ORANGE='#d95f02'
GREEN='#2e7d32'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':TEXT_C,
                     'axes.labelcolor':TEXT_C,'xtick.color':TEXT_C,'ytick.color':TEXT_C})
ep = hist_df['epoch'].tolist()

# ═══ PNG 1 — 5 METRIC CURVES WITH DYNAMIC Y-AXIS ═══
fig, axes = plt.subplots(2,3, figsize=(22,12), facecolor=DARK_BG)
fig.suptitle(
    'SGD - Training & Validation Curves',
    fontsize=17, fontweight='bold', color=TEXT_C, y=0.98)

curve_specs = [
    ('Accuracy',  'train_accuracy',  'val_accuracy',  'Accuracy'),
    ('Precision', 'train_precision', 'val_precision', 'Precision (macro)'),
    ('Recall',    'train_recall',    'val_recall',    'Recall (macro)'),
    ('F1 Score',  'train_f1',        'val_f1',        'F1 Score (macro)'),
    ('Loss',      'train_loss',      'val_loss',      'Log-Loss'),
]
flat_axes = [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1]]
axes[1,2].set_visible(False)

for ax, (title, tr_key, vl_key, ylabel) in zip(flat_axes, curve_specs):
    ax.set_facecolor(PANEL_BG); ax.spines[:].set_color(GRID_C)
    tr_v = hist_df[tr_key].tolist(); vl_v = hist_df[vl_key].tolist()
    
    ax.plot(ep, tr_v, color=TRAIN_C, lw=2.2, label='Train', zorder=3)
    ax.plot(ep, vl_v, color=VAL_C,   lw=2.2, label='Validation', zorder=3)
    
    # DYNAMIC Y-AXIS LIMITS based on actual values
    all_values = tr_v + vl_v
    if 'loss' in tr_key:
        # For loss, show from near minimum to slightly above maximum
        y_min = min(all_values) * 0.95
        y_max = max(all_values) * 1.1
    else:
        # For metrics (0-1 range), zoom to actual range with small padding
        y_min = max(0, min(all_values) - 0.05)
        y_max = min(1.0, max(all_values) + 0.05)
        # If range is very small, add more padding
        if max(all_values) - min(all_values) < 0.1:
            y_min = max(0, min(all_values) - 0.1)
            y_max = min(1.0, max(all_values) + 0.1)
    
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_xlim(left=0.8, right=max(ep)+0.2)
    
    ax.set_title(title, fontsize=13, fontweight='bold', color=TEXT_C, pad=8)
    ax.set_xlabel('Epoch', fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)
    ax.grid(True, color=GRID_C, lw=0.6, alpha=0.7); ax.tick_params(labelsize=9)

fig.text(0.5, 0.01,
    f"loss={best_params['loss']}, alpha={best_params['alpha']}, L2, batch={BATCH} | "
    f"char(2-4)+word(1-2) TF-IDF min_df=3 max_df=0.95 | "
    f"Best epoch {best_epoch} (Val F1={best_val_f1:.4f})",
    ha='center', fontsize=8, color='#888888')

plt.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.07, hspace=0.42, wspace=0.32)
plt.savefig(p('v3_training_curves.png'), dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close(); print("v3_training_curves.png saved.")

# ═══ PNG 2 — CONFUSION MATRIX (Styled like reference image) ═══
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp_val = cm.ravel()

# Create styled confusion matrix plot
fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
ax.set_facecolor('white')

# Create custom heatmap
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['0', '1'], fontsize=14, fontweight='bold')
ax.set_yticklabels(['0', '1'], fontsize=14, fontweight='bold')

# Add labels
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('Confusion Matrix - SGD (Test Set)', fontsize=16, fontweight='bold', pad=20)

# Add text annotations with values
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=18, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=11)

# Add metrics text box
metrics_text = (f"Accuracy  : {test_metrics['accuracy']:.4f}\n"
                f"Precision : {test_metrics['precision']:.4f}\n"
                f"Recall    : {test_metrics['recall']:.4f}\n"
                f"F1 Score  : {test_metrics['f1']:.4f}\n"
                f"Log-Loss  : {test_metrics['log_loss']:.4f}")

# Add a text box with metrics
props = dict(boxstyle='round', facecolor='white', edgecolor='#2e7d32', alpha=0.9)
ax.text(1.25, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', bbox=props, fontfamily='monospace')

plt.tight_layout()
plt.savefig(p('v3_confusion_matrix.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("v3_confusion_matrix.png saved.")

# ═══ PNG 3 — ROC-AUC ═══
j=tpr-fpr; oi=np.argmax(j)
opt_fpr,opt_tpr,opt_thr = fpr[oi],tpr[oi],thresholds[oi]
fig,ax = plt.subplots(figsize=(8,7), facecolor='white')
ax.set_facecolor('white'); ax.spines[:].set_color('#cccccc')
ax.fill_between(fpr,tpr,alpha=0.18,color=ACCENT)
ax.plot(fpr,tpr,color=TRAIN_C,lw=2.8,label=f'ROC Curve (AUC={roc_auc_val:.4f})')
ax.plot([0,1],[0,1],color='#888888',lw=1.4,ls='--',label='Random Classifier')
ax.scatter([opt_fpr],[opt_tpr],color=ORANGE,s=100,zorder=5,
           label=f'Best Threshold={opt_thr:.3f}\n(FPR={opt_fpr:.3f}, TPR={opt_tpr:.3f})')
ax.axvline(opt_fpr,color=ORANGE,lw=1,ls=':',alpha=0.5)
ax.axhline(opt_tpr,color=ORANGE,lw=1,ls=':',alpha=0.5)
ax.text(0.62,0.18,f'AUC = {roc_auc_val:.4f}',fontsize=20,fontweight='bold',color=ACCENT,
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5',facecolor='white',edgecolor=ACCENT,lw=2))
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.05])
ax.set_xlabel('False Positive Rate (FPR)',fontsize=12)
ax.set_ylabel('True Positive Rate (TPR / Recall)',fontsize=12)
ax.set_title('ROC-AUC Curve — Hate Speech Detection (Test Set)',
             fontsize=14,fontweight='bold',color='black',pad=15)
ax.legend(fontsize=10,facecolor='white',edgecolor='#cccccc',loc='lower right')
ax.grid(True,color='#cccccc',lw=0.6,alpha=0.7); ax.tick_params(colors='black',labelsize=10)
plt.tight_layout()
plt.savefig(p('v3_roc_auc_curve.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(); print("v3_roc_auc_curve.png saved.")

# Calculate total execution time
total_end_time = time.time()
total_execution_time = total_end_time - total_start_time
print(f"\nTotal execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")

# Optional: Save timing info
timing_info = pd.DataFrame([{
    'training_time_seconds': training_time,
    'training_time_minutes': training_time/60,
    'total_execution_time_seconds': total_execution_time,
    'total_execution_time_minutes': total_execution_time/60,
    'n_epochs': N_EPOCHS,
    'batch_size': BATCH,
}])
timing_info.to_csv(p('v3_timing_info.csv'), index=False)

print("\nAll outputs saved:")
print("  PNGs: v3_training_curves.png | v3_confusion_matrix.png | v3_roc_auc_curve.png")
print("  CSVs: v3_train_history.csv | v3_param_search.csv | v3_test_metrics.csv | v3_test_evaluation.csv | v3_timing_info.csv")

