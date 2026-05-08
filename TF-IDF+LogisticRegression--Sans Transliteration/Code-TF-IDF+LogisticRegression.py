import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, log_loss)
import warnings
import time
import joblib
import pickle
warnings.filterwarnings('ignore')
import os

# Set your paths
code_location = '/content/drive/MyDrive/TF-IDF+LogisticRegression--Sans Transliteration/'
results_folder = os.path.join(code_location, 'Results-TF-IDF+LogisticRegression')

# IMPORTANT: Create the results folder FIRST
os.makedirs(results_folder, exist_ok=True)

# THEN change to that directory
os.chdir(results_folder)
print(f"Current working directory: {os.getcwd()}")

# ==================== 1. LOAD AND PREPROCESS DATA ====================

print("Chargement des datasets...")
start_time_total = time.time()

# Load the three datasets
df_train = pd.read_csv('/content/drive/MyDrive/data/train.csv', sep=';')
df_validation = pd.read_csv('/content/drive/MyDrive/data/validation.csv', sep=';')
df_test = pd.read_csv('/content/drive/MyDrive/data/test.csv', sep=';')

# Combine train and validation for more training data
df_train = pd.concat([df_train, df_validation], ignore_index=True)

# Remove any rows with missing values
df_train = df_train.dropna()
df_test = df_test.dropna()

print(f"Taille de l'ensemble d'entraînement: {len(df_train)}")
print(f"Taille de l'ensemble de test: {len(df_test)}")

# Prepare features and labels
X_train = df_train['text'].astype(str)
y_train = df_train['label'].astype(int)

X_test = df_test['text'].astype(str)
y_test = df_test['label'].astype(int)

print(f"\nDistribution des classes dans l'entraînement:")
print(y_train.value_counts())
print(f"\nDistribution des classes dans le test:")
print(y_test.value_counts())

# ==================== 2. TF-IDF VECTORIZATION ====================

print("\nVectorisation des données textuelles...")
vectorizer_start = time.time()

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    lowercase=True,
    stop_words=None
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

vectorizer_time = time.time() - vectorizer_start
print(f"Forme TF-IDF: {X_train_tfidf.shape}")
print(f"Temps de vectorisation: {vectorizer_time:.2f} secondes")

# ==================== 3. GRID SEARCH WITH 10-FOLD CV ====================

print("\nRecherche par grille avec validation croisée 10-fold...")

gridsearch_start = time.time()

# Define parameter grid for Logistic Regression
param_grid = {
    'solver': ['lbfgs', 'saga', 'liblinear'], 
    'penalty': ['l2'], 
    'C': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 15.0, 20, 25, 50, 100, 1000], # Plage plus large pour mieux voir l'effet de la régularisation
    'class_weight': [None, 'balanced'],
    'max_iter': [1000] 
}

# Create Logistic Regression model
lr = LogisticRegression(random_state=42)

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    lr, param_grid, cv=cv, scoring='f1_macro',
    n_jobs=-1, verbose=1, return_train_score=True
)

grid_search.fit(X_train_tfidf, y_train)

gridsearch_time = time.time() - gridsearch_start

print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score F1 en validation croisée: {grid_search.best_score_:.4f}")
print(f"Temps de recherche par grille: {gridsearch_time:.2f} secondes")

# Save grid search results to CSV
grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results.to_csv('grid_search_results.csv', index=False)
print("\nRésultats de la recherche par grille sauvegardés dans 'grid_search_results.csv'")

# ==================== 4. FINAL LOGISTIC REGRESSION MODEL ====================

print("\nUtilisation du meilleur modèle de régression logistique...")
best_lr = grid_search.best_estimator_

# ==================== 5. FINAL EVALUATION ON TEST SET ====================

print("\nÉvaluation sur l'ensemble de test...")
inference_start = time.time()

y_pred = best_lr.predict(X_test_tfidf)
y_prob = best_lr.predict_proba(X_test_tfidf)[:, 1]

inference_time = time.time() - inference_start
total_time = time.time() - start_time_total

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
log_loss_value = log_loss(y_test, y_prob)

# Create test metrics dictionary
test_metrics = {
    'Accuracy': accuracy,
    'F1 Score': f1,
    'Recall': recall,
    'Precision': precision,
    'ROC AUC': roc_auc,
    'Log Loss': log_loss_value
}

# Generate classification report
target_names = ['0', '1']
class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# ==================== 5b. SAVE TIME METRICS TO CSV ====================

time_metrics = {
    'Metric': ['Vectorisation', 'Recherche par grille', 'Inférence (Test)', 'Exécution totale'],
    'Time (seconds)': [vectorizer_time, gridsearch_time, inference_time, total_time],
    'Time (minutes)': [vectorizer_time/60, gridsearch_time/60, inference_time/60, total_time/60]
}

time_metrics_df = pd.DataFrame(time_metrics)
time_metrics_df.to_csv('execution_times.csv', index=False)
print("\nTemps d'exécution sauvegardés dans 'execution_times.csv'")

print("\n=== TEMPS D'EXÉCUTION ===")
for metric, t_sec, t_min in zip(time_metrics['Metric'], time_metrics['Time (seconds)'], time_metrics['Time (minutes)']):
    print(f"{metric}: {t_sec:.2f} sec ({t_min:.2f} min)")

print("\n=== MÉTRIQUES SUR L'ENSEMBLE DE TEST ===")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save test metrics to CSV
test_metrics_df = pd.DataFrame([test_metrics])
test_metrics_df.to_csv('test_metrics.csv', index=False)
print("\nMétriques de test sauvegardées dans 'test_metrics.csv'")

print(classification_report(y_test, y_pred, target_names=target_names))
class_report_df.to_csv('classification_report.csv')
print("Rapport de classification sauvegardé dans 'classification_report.csv'")

# Create detailed evaluation dataframe
eval_df = pd.DataFrame({
    'text': X_test.values,
    'true_label': y_test.values,
    'predicted_label': y_pred,
    'is_correct': y_test.values == y_pred,
    'probability_hate': y_prob,
    'probability_non_hate': 1 - y_prob
})

eval_df.to_csv('test_evaluation.csv', index=False, encoding='utf-8-sig')
print("Évaluation de test sauvegardée dans 'test_evaluation.csv'")

# ==================== 6. LEARNING CURVES ====================

print("\nGénération des courbes d'apprentissage...")

def plot_learning_curves(estimator, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10)):
    """Tracer les courbes d'apprentissage pour plusieurs métriques"""

    metrics = ['accuracy', 'f1', 'precision', 'recall']
    df_curves = pd.DataFrame()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Régression Logistique avec TF-IDF - Courbes d\'Apprentissage',
             fontsize=16, fontweight='bold')
    axes = axes.flatten()

    colors = ['#1f77b4', '#d62728', '#F18F01', '#C73E1D', '#3D5A80']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=metric,
            train_sizes=train_sizes, n_jobs=-1
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        if idx == 0:
            df_curves['train_sizes'] = train_sizes_abs
        df_curves[f'{metric}_train_mean'] = train_mean
        df_curves[f'{metric}_train_std'] = train_std
        df_curves[f'{metric}_test_mean'] = test_mean
        df_curves[f'{metric}_test_std'] = test_std

        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color=colors[0])
        ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                        alpha=0.1, color=colors[1])
        ax.plot(train_sizes_abs, train_mean, 'o-', color=colors[0], label='Entraînement')
        ax.plot(train_sizes_abs, test_mean, 'o-', color=colors[1], label='Cross Validation (10-Fold)')

        # Titres des sous-graphes en français
        metric_names = {'accuracy': 'Accuracy', 'f1': 'F1-Score', 'precision': 'Précision', 'recall': 'Rappel'}
        ax.set_title(f'Courbe d\'apprentissage - {metric_names[metric]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Taille de l\'ensemble d\'entraînement', fontsize=10)
        ax.set_ylabel(metric_names[metric], fontsize=10)
        ax.set_yticks(np.arange(0.5, 1.05, 0.1))
        ax.set_ylim(0.5, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    # Courbe de Log Loss
    ax_loss = axes[4]

    def log_loss_scorer(estimator, X, y):
        y_prob = estimator.predict_proba(X)
        return -log_loss(y, y_prob)

    train_sizes_abs, train_scores_loss, test_scores_loss = learning_curve(
        estimator, X, y, cv=cv, scoring=log_loss_scorer,
        train_sizes=train_sizes, n_jobs=-1
    )

    train_mean_loss = -np.mean(train_scores_loss, axis=1)
    train_std_loss = np.std(train_scores_loss, axis=1)
    test_mean_loss = -np.mean(test_scores_loss, axis=1)
    test_std_loss = np.std(test_scores_loss, axis=1)

    df_curves['logloss_train_mean'] = train_mean_loss
    df_curves['logloss_train_std'] = train_std_loss
    df_curves['logloss_test_mean'] = test_mean_loss
    df_curves['logloss_test_std'] = test_std_loss

    ax_loss.fill_between(train_sizes_abs, train_mean_loss - train_std_loss,
                         train_mean_loss + train_std_loss, alpha=0.1, color=colors[2])
    ax_loss.fill_between(train_sizes_abs, test_mean_loss - test_std_loss,
                         test_mean_loss + test_std_loss, alpha=0.1, color=colors[3])
    ax_loss.plot(train_sizes_abs, train_mean_loss, 'o-', color=colors[2], label='Entraînement')
    ax_loss.plot(train_sizes_abs, test_mean_loss, 'o-', color=colors[3], label='Cross Validation (10-Fold)')
    ax_loss.set_title('Courbe d\'apprentissage - Perte (Log Loss)', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Taille de l\'ensemble d\'entraînement', fontsize=10)
    ax_loss.set_ylabel('Perte (Log Loss)', fontsize=10)
    ax_loss.legend(loc='best')
    ax_loss.grid(True, alpha=0.3)

    # Graphique récapitulatif des performances
    ax_summary = axes[5]
    test_metrics_values = [test_metrics['Accuracy'], test_metrics['F1 Score'],
                          test_metrics['Precision'], test_metrics['Recall'], test_metrics['ROC AUC']]
    metric_names_fr = ['Accuracy', 'F1', 'Précision', 'Rappel', 'ROC AUC']
    bars = ax_summary.bar(metric_names_fr, test_metrics_values, color=colors)
    ax_summary.set_title('Récapitulatif des Performances sur l\'Ensemble de Test\n', fontsize=12, fontweight='bold')
    ax_summary.set_ylabel('Score', fontsize=10)
    ax_summary.set_ylim([0, 1.05])

    for bar, val in zip(bars, test_metrics_values):
        ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax_summary.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    df_curves.to_csv('learning_curves_data.csv', index=False)
    print("Courbes d'apprentissage sauvegardées dans 'learning_curves.png'")
    print("Données des courbes d'apprentissage sauvegardées dans 'learning_curves_data.csv'")

    return df_curves

# Train a fresh model for learning curves
print("Entraînement du modèle pour les courbes d'apprentissage (cela peut prendre un moment)...")
from sklearn.base import clone
lr_for_learning = clone(best_lr)
df_curves = plot_learning_curves(lr_for_learning, X_train_tfidf, y_train)

# ==================== 7. CONFUSION MATRIX ====================

print("\nGénération de la matrice de confusion...")

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Haineux', 'Haineux'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Matrice de Confusion - Régression Logistique avec TF-IDF\n', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Matrice de confusion sauvegardée dans 'confusion_matrix.png'")

# ==================== 8. ROC AUC CURVE ====================

print("\nGénération de la courbe ROC AUC...")

fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Classificateur Aléatoire')
ax.set_xlabel('Taux de Faux Positifs (False Positive Rate)', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs (True Positive Rate)', fontsize=12)
ax.set_title('Courbe ROC AUC - Régression Logistique avec TF-IDF\n', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_auc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("Courbe ROC AUC sauvegardée dans 'roc_auc_curve.png'")

# ==================== 9. FEATURE IMPORTANCE ====================

print("\nGénération de la visualisation des caractéristiques importantes...")

# Get feature names and coefficients
feature_names = tfidf.get_feature_names_out()
coefficients = best_lr.coef_[0]



# Save features to CSV
features_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)
features_df.to_csv('feature_coefficients.csv', index=False)
print("Coefficients des caractéristiques sauvegardés dans 'feature_coefficients.csv'")

# ==================== 10. SAVE MODEL FOR LATER USE ====================

print("\nSauvegarde du modèle pour une utilisation ultérieure...")

# Save as pickle (joblib)
joblib.dump(best_lr, 'LogisticRegression_model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

# Save as .pth (using pickle protocol)
model_artifacts = {
    'classifier': best_lr,
    'vectorizer': tfidf,
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'test_metrics': test_metrics,
    'feature_names': feature_names,
    'coefficients': coefficients
}

with open('LogisticRegression_model.pth', 'wb') as f:
    pickle.dump(model_artifacts, f)

# Also save metadata separately
metadata = {
    'model_type': 'RegressionLogistique',
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'test_metrics': test_metrics,
    'training_size': len(X_train),
    'vectorizer_params': tfidf.get_params(),
    'execution_times': time_metrics,
    'feature_count': len(feature_names)
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Modèle sauvegardé sous:")
print("  - LogisticRegression_model.pkl (format joblib)")
print("  - LogisticRegression_model.pth (format pickle)")
print("  - vectorizer.pkl")
print("  - model_metadata.pkl")

# ==================== 11. SUMMARY REPORT ====================

print("\n" + "="*60)
print("RÉSUMÉ FINAL")
print("="*60)

print(f"\nMeilleurs paramètres de la recherche par grille:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nMeilleur score F1 en validation croisée: {grid_search.best_score_:.4f}")

print("\nPerformances sur l'ensemble de test:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nTemps d'exécution:")
for metric, t_sec, t_min in zip(time_metrics['Metric'], time_metrics['Time (seconds)'], time_metrics['Time (minutes)']):
    print(f"  {metric}: {t_sec:.2f} sec ({t_min:.2f} min)")

print("\nFichiers générés:")
print("  📊 Métriques de Performance:")
print("    1. grid_search_results.csv - Résultats du tuning par grille")
print("    2. test_metrics.csv - Métriques finales sur le test")
print("    3. test_evaluation.csv - Prédictions détaillées par texte")
print("    4. execution_times.csv - Temps d'entraînement et d'inférence")
print("    5. classification_report.csv - Rapport de classification détaillé")
print("    6. feature_coefficients.csv - Tous les coefficients des caractéristiques")
print("  📈 Visualisations:")
print("    7. learning_curves.png - Courbes d'apprentissage")
print("    8. confusion_matrix.png - Matrice de confusion")
print("    9. roc_auc_curve.png - Courbe ROC AUC")
print("    10. top_features.png - Top 20 caractéristiques par classe")
print("  🤖 Fichiers Modèle:")
print("    11. LogisticRegression_model.pkl - Modèle entraîné (joblib)")
print("    12. LogisticRegression_model.pth - Modèle entraîné (pickle)")
print("    13. vectorizer.pkl - Vectoriseur TF-IDF")
print("    14. model_metadata.pkl - Métadonnées du modèle")

# ==================== 12. INTERFACE READY FUNCTION ====================

def predict_comment(text, model_path='LogisticRegression_model.pth', vectorizer_path='vectorizer.pkl'):
    """Fonction pour prédire un commentaire unique (pour l'interface)"""
    import pickle
    import joblib

    # Load model and vectorizer
    with open(model_path, 'rb') as f:
        model_artifacts = pickle.load(f)

    vectorizer = joblib.load(vectorizer_path)
    classifier = model_artifacts['classifier']

    # Preprocess and predict
    text_vectorized = vectorizer.transform([text])
    prediction = classifier.predict(text_vectorized)[0]
    probability = classifier.predict_proba(text_vectorized)[0]

    return {
        'text': text,
        'is_hate_speech': bool(prediction == 1),
        'probability_hate': probability[1],
        'probability_non_hate': probability[0],
        'prediction': 'Discours Haineux' if prediction == 1 else 'Non-Haineux'
    }

print("\n" + "="*60)
print("INTERFACE PRÊTE")
print("="*60)
print("\nVous pouvez maintenant utiliser le modèle sauvegardé dans votre interface avec:")
print("""
from your_script import predict_comment

# Exemple d'utilisation:
result = predict_comment("Votre texte ici")
print(result)
""")

# Test the interface function
print("\nTest de la fonction d'interface sur le premier échantillon de test:")
sample_text = X_test.iloc[0]
sample_result = predict_comment(sample_text)
print(f"Texte: {sample_text[:100]}...")
print(f"Prédiction: {sample_result['prediction']}")
print(f"Probabilité Haineux: {sample_result['probability_hate']:.4f}")

# ==================== 13. SAMPLE PREDICTIONS ====================

print("\n" + "="*60)
print("PRÉDICTIONS EXEMPLES (20 premiers échantillons de test)")
print("="*60)

sample_df = eval_df.head(20)
for idx, row in sample_df.iterrows():
    text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
    status = "✓" if row['is_correct'] else "✗"
    print(f"\n{status} Échantillon {idx+1}:")
    print(f"   Texte: {text_preview}")
    print(f"   Réel: {row['true_label']} | Prédit: {row['predicted_label']} | Correct: {row['is_correct']}")
    print(f"   Probabilité Haineux: {row['probability_hate']:.4f}")

# ==================== 14. C PARAMETER ANALYSIS ====================

print("\n" + "="*60)
print("ANALYSE DU PARAMÈTRE C")
print("="*60)

# Extract C parameter effect from grid search results
c_analysis = []
for idx, row in grid_results.iterrows():
    c_analysis.append({
        'C': row['param_C'],
        'mean_train_score': row['mean_train_score'],
        'mean_test_score': row['mean_test_score'],
        'std_train_score': row['std_train_score'],
        'std_test_score': row['std_test_score']
    })

c_analysis_df = pd.DataFrame(c_analysis)
c_summary = c_analysis_df.groupby('C').agg({
    'mean_train_score': 'mean',
    'mean_test_score': 'mean',
    'std_train_score': 'mean',
    'std_test_score': 'mean'
}).reset_index()
c_summary = c_summary.sort_values('C')

# Plot C parameter effect
plt.figure(figsize=(10, 6))
plt.errorbar(c_summary['C'], c_summary['mean_train_score'],
             yerr=c_summary['std_train_score'], label='Score d\'entraînement',
             marker='o', capsize=5, color='blue')
plt.errorbar(c_summary['C'], c_summary['mean_test_score'],
             yerr=c_summary['std_test_score'], label='Score CV (10-Fold)',
             marker='s', capsize=5, color='red')
plt.xscale('log')
plt.xlabel('C (Force de Régularisation)', fontsize=12)
plt.ylabel('Score F1', fontsize=12)
plt.title('Régression Logistique avec TF-IDF - Effet du Paramètre C sur les Performances\n', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
best_c = grid_search.best_params_['C']
plt.axvline(x=best_c, color='green', linestyle='--', alpha=0.7, label=f'Meilleur C = {best_c}')
plt.legend()
plt.tight_layout()
plt.savefig('c_parameter_effect.png', dpi=300, bbox_inches='tight')
plt.show()
print("Analyse du paramètre C sauvegardée dans 'c_parameter_effect.png' ")

print("\n" + "="*60)
print("SCRIPT TERMINÉ AVEC SUCCÈS")
print("="*60)