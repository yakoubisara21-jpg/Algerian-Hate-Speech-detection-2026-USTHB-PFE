import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, 
                             roc_auc_score, roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay)
import warnings
import time
import joblib
import pickle
warnings.filterwarnings('ignore')
import os

# Set your paths
code_location = '/content/drive/MyDrive/TF-IDF+LinearSVC/'
results_folder = os.path.join(code_location, 'Results-TF-IDF+LinearSVC')

# IMPORTANT: Create the results folder FIRST
os.makedirs(results_folder, exist_ok=True)

# THEN change to that directory
os.chdir(results_folder)
print(f"Current working directory: {os.getcwd()}")


# ==================== 1. LOAD AND PREPROCESS DATA ====================

print("Loading datasets...")
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

print(f"Training set size: {len(df_train)}")
print(f"Test set size: {len(df_test)}")

# Prepare features and labels
X_train = df_train['text'].astype(str)
y_train = df_train['label'].astype(int)

X_test = df_test['text'].astype(str)
y_test = df_test['label'].astype(int)

print(f"\nClass distribution in training:")
print(y_train.value_counts())
print(f"\nClass distribution in test:")
print(y_test.value_counts())

# ==================== 2. TF-IDF VECTORIZATION ====================

print("\nVectorizing text data...")
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
print(f"TF-IDF shape: {X_train_tfidf.shape}")
print(f"Vectorization time: {vectorizer_time:.2f} seconds")

# ==================== 3. GRID SEARCH WITH 10-FOLD CV ====================

print("\nPerforming Grid Search with 10-fold cross-validation...")

gridsearch_start = time.time()

# Define parameter grid for LinearSVC
param_grid = {
    'C': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': [None, 'balanced'],
    'max_iter': [5000]
}

# Create LinearSVC model
svm = LinearSVC(random_state=42, dual='auto')

# 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    svm, param_grid, cv=cv, scoring='f1_macro', 
    n_jobs=-1, verbose=1, return_train_score=True
)

grid_search.fit(X_train_tfidf, y_train)

gridsearch_time = time.time() - gridsearch_start

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
print(f"Grid search time: {gridsearch_time:.2f} seconds")

# Save grid search results to CSV
grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results.to_csv('grid_search_results.csv', index=False)
print("\nGrid search results saved to 'grid_search_results.csv'")

# ==================== 4. FINAL LINEAR SVC (NO CALIBRATION) ====================

print("\nUsing best LinearSVC model ...")
best_svm = grid_search.best_estimator_

# ==================== 5. FINAL EVALUATION ON TEST SET ====================

print("\nEvaluating on test set...")
inference_start = time.time()

y_pred = best_svm.predict(X_test_tfidf)

# For decision function (instead of predict_proba which isn't available)
y_decision = best_svm.decision_function(X_test_tfidf)

inference_time = time.time() - inference_start
total_time = time.time() - start_time_total

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# For ROC AUC, we use decision function values (equivalent to distance from hyperplane)
roc_auc = roc_auc_score(y_test, y_decision)

# Create test metrics dictionary with classification report
test_metrics = {
    # Basic metrics
    'Accuracy': accuracy,
    'F1 Score': f1,
    'Recall': recall,
    'Precision': precision,
    'ROC AUC': roc_auc
}

# Generate classification report
target_names=['0', '1']
class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
# Convert to DataFrame and save as CSV in the exact format
class_report_df = pd.DataFrame(class_report).transpose()

# ==================== 5b. SAVE TIME METRICS TO CSV ====================

time_metrics = {
    'Metric': ['Vectorization', 'Grid Search', 'Inference (Test Set)', 'Total Execution'],
    'Time (seconds)': [vectorizer_time, gridsearch_time, inference_time, total_time],
    'Time (minutes)': [vectorizer_time/60, gridsearch_time/60, inference_time/60, total_time/60]
}

time_metrics_df = pd.DataFrame(time_metrics)
time_metrics_df.to_csv('execution_times.csv', index=False)
print("\nExecution times saved to 'execution_times.csv'")

print("\n=== EXECUTION TIMES ===")
for metric, t_sec, t_min in zip(time_metrics['Metric'], time_metrics['Time (seconds)'], time_metrics['Time (minutes)']):
    print(f"{metric}: {t_sec:.2f} sec ({t_min:.2f} min)")

print("\n=== TEST SET METRICS ===")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save test metrics to CSV 
test_metrics_df = pd.DataFrame([test_metrics])
test_metrics_df.to_csv('test_metrics.csv', index=False)
print("\nTest metrics saved to 'test_metrics.csv'")

print(classification_report(y_test, y_pred, target_names=target_names))
# Save to CSV
class_report_df.to_csv('classification_report.csv')
print("Classification report saved to 'classification_report.csv'")

# Calculate probabilities using sigmoid
prob_hate = 1 / (1 + np.exp(-y_decision))
prob_non_hate = 1 - prob_hate

# Create detailed evaluation dataframe
eval_df = pd.DataFrame({
    'text': X_test.values,
    'true_label': y_test.values,
    'predicted_label': y_pred,
    'is_correct': y_test.values == y_pred,
    'decision_score': y_decision,
    'probability_hate': prob_hate,
    'probability_non_hate': prob_non_hate
})

eval_df.to_csv('test_evaluation.csv', index=False, encoding='utf-8-sig')
print("Test evaluation saved to 'test_evaluation.csv'")

# ==================== 6. LEARNING CURVES ====================

print("\nGenerating learning curves...")

def plot_learning_curves(estimator, X, y, cv=10, train_sizes=np.linspace(0.1, 1.0, 10)):
    """Plot learning curves for multiple metrics"""
    
    # Calculate learning curves for different metrics
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    # Create a single DataFrame with train_sizes as the first column
    df_curves = pd.DataFrame()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
        ax.plot(train_sizes_abs, test_mean, 'o-', color=colors[1], label='Cross-validation (10-Fold)')
        
        ax.set_title(f'Courbe d\'apprentissage - {metric.upper()}')
        ax.set_xlabel('Taille du dataset d\'entraînement')  
        ax.set_ylabel(metric.upper())
        ax.set_yticks(np.arange(0.5, 1.05, 0.1))
        ax.set_ylim(0.5, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Plot Loss curve using the best loss function from grid search
    ax_loss = axes[4]
    
    # Extract the best loss function name
    best_loss = grid_search.best_params_.get('loss', 'squared_hinge')  # Default to squared_hinge if not found
    
    # Define custom scoring function for hinge loss (no external library)
    def hinge_loss_scorer(estimator, X, y):
        """Calculate hinge loss for LinearSVC manually"""
        y_pred_decision = estimator.decision_function(X)
        # Hinge loss: max(0, 1 - y_true * y_pred)
        # Note: y should be in {+1, -1} for hinge loss formula
        y_binary = np.where(y == 0, -1, 1)  # Convert 0/1 to -1/+1
        losses = np.maximum(0, 1 - y_binary * y_pred_decision)
        hinge_loss_value = np.mean(losses)
        return -hinge_loss_value  # Negative because learning_curve maximizes
    
    def squared_hinge_loss_scorer(estimator, X, y):
        """Calculate squared hinge loss for LinearSVC manually"""
        y_pred_decision = estimator.decision_function(X)
        # Convert 0/1 to -1/+1 for hinge loss formula
        y_binary = np.where(y == 0, -1, 1)
        # Hinge loss: max(0, 1 - y_true * y_pred)
        hinge = np.maximum(0, 1 - y_binary * y_pred_decision)
        # Square the hinge loss values
        squared_hinge = np.mean(hinge ** 2)
        return -squared_hinge  # Negative because learning_curve maximizes
    
    # Select the appropriate loss scorer based on best_loss
    if best_loss == 'hinge':
        loss_scorer = hinge_loss_scorer
        loss_name = 'Hinge Loss'
    elif best_loss == 'squared_hinge':
        loss_scorer = squared_hinge_loss_scorer
        loss_name = 'Squared Hinge Loss'
    else:
        # Fallback to squared_hinge if something else
        print(f"Warning: Unknown loss '{best_loss}', using squared_hinge")
        loss_scorer = squared_hinge_loss_scorer
        loss_name = 'Squared Hinge Loss'
    
    # Calculate learning curve for the loss
    train_sizes_abs, train_scores_loss, test_scores_loss = learning_curve(
        estimator, X, y, cv=cv, scoring=loss_scorer,
        train_sizes=train_sizes, n_jobs=-1
    )
    
    # Convert back from negative scores (since we returned negative for maximization)
    train_mean_loss = -np.mean(train_scores_loss, axis=1)
    train_std_loss = np.std(train_scores_loss, axis=1)
    test_mean_loss = -np.mean(test_scores_loss, axis=1)
    test_std_loss = np.std(test_scores_loss, axis=1)
    
    df_curves['loss_train_mean'] = train_mean_loss
    df_curves['loss_train_std'] = train_std_loss
    df_curves['loss_test_mean'] = test_mean_loss
    df_curves['loss_test_std'] = test_std_loss
    df_curves['loss_type'] = loss_name

    ax_loss.fill_between(train_sizes_abs, train_mean_loss - train_std_loss, 
                         train_mean_loss + train_std_loss, alpha=0.1, color=colors[2])
    ax_loss.fill_between(train_sizes_abs, test_mean_loss - test_std_loss, 
                         test_mean_loss + test_std_loss, alpha=0.1, color=colors[3])
    ax_loss.plot(train_sizes_abs, train_mean_loss, 'o-', color=colors[2], label='Entraînement')
    ax_loss.plot(train_sizes_abs, test_mean_loss, 'o-', color=colors[3], label='Cross-validation (10-Fold)')
    ax_loss.set_title(f'Courbe d\'apprentissage - {loss_name}')
    ax_loss.set_xlabel('Taille du dataset d\'entraînement')  
    ax_loss.set_ylabel(loss_name)
    ax_loss.legend(loc='best')
    ax_loss.grid(True, alpha=0.3)
    
    # Plot Summary of best scores
    ax_summary = axes[5]
    test_metrics_values = [test_metrics['Accuracy'], test_metrics['F1 Score'], 
                          test_metrics['Precision'], test_metrics['Recall'], test_metrics['ROC AUC']]
    metric_names = ['Accuracy', 'F1', 'Precision', 'Recall', 'ROC AUC']
    bars = ax_summary.bar(metric_names, test_metrics_values, color=colors)
    ax_summary.set_title('Résumé des performances sur le dataset de test')
    ax_summary.set_ylabel('Score')
    ax_summary.set_ylim([0, 1.05])
    
    # Add value labels on top of bars
    for bar, val in zip(bars, test_metrics_values):
        ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax_summary.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
        # Save the single DataFrame with train_sizes once
    df_curves.to_csv('learning_curves_data.csv', index=False)
    print("Learning curves saved to 'learning_curves.png'")
    print("Learning curves data saved to 'learning_curves_data.csv'")
    print(f"CSV shape: {df_curves.shape}")
    print(f"Columns: {list(df_curves.columns)}")
    print(f"Loss curve uses: {loss_name} (selected by grid search)")
    
    return df_curves

# Train a fresh model for learning curves
print("Training model for learning curves (this may take a moment)...")
from sklearn.base import clone
svm_for_learning = clone(best_svm)
df_curves = plot_learning_curves(svm_for_learning, X_train_tfidf, y_train)

# ==================== 7. CONFUSION MATRIX ====================

print("\nGenerating confusion matrix...")

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Hate', 'Hate'])
disp.plot(ax=ax, cmap='Greens', values_format='d')
ax.set_title(f'Matrice de confusion\n')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Confusion matrix saved to 'confusion_matrix.png'")

# ==================== 8. ROC AUC CURVE ====================

print("\nGenerating ROC AUC curve...")

fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_decision)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC AUC Curve - Hate Speech Detection', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_auc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("ROC AUC curve saved to 'roc_auc_curve.png'")

# ==================== 9. SAVE MODEL FOR LATER USE ====================

print("\nSaving model for later use in interface...")

# Save as pickle (standard)
joblib.dump(best_svm, 'LinearSVC_model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

# Save as .pth (using pickle protocol)
model_artifacts = {
    'classifier': best_svm,
    'vectorizer': tfidf,
    'best_params': grid_search.best_params_,
    'test_metrics': test_metrics,
    'feature_names': tfidf.get_feature_names_out()
}

with open('LinearSVC_model.pth', 'wb') as f:
    pickle.dump(model_artifacts, f)

# Also save metadata separately
metadata = {
    'model_type': 'LinearSVC',
    'best_params': grid_search.best_params_,
    'best_cv_score': grid_search.best_score_,
    'test_metrics': test_metrics,
    'training_size': len(X_train),
    'vectorizer_params': tfidf.get_params(),
    'execution_times': time_metrics
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Model saved as:")
print("  - LinearSVC_model.pkl (joblib format)")
print("  - LinearSVC_model.pth (pickle format)")
print("  - vectorizer.pkl")
print("  - model_metadata.pkl")

# ==================== 10. SUMMARY REPORT ====================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"\nBest Parameters from Grid Search:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV F1 Score: {grid_search.best_score_:.4f}")

print("\nTest Set Performance:")
for metric, value in test_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nExecution Times:")
for metric, t_sec, t_min in zip(time_metrics['Metric'], time_metrics['Time (seconds)'], time_metrics['Time (minutes)']):
    print(f"  {metric}: {t_sec:.2f} sec ({t_min:.2f} min)")

print("\nFiles generated:")
print("  📊 Performance Metrics:")
print("    1. grid_search_results.csv - Grid search tuning results")
print("    2. test_metrics.csv - Final test metrics")
print("    3. test_evaluation.csv - Detailed predictions per text")
print("    4. execution_times.csv - Training and inference times")
print("  📈 Visualizations:")
print("    5. learning_curves.png - Learning curves")
print("    6. confusion_matrix.png - Confusion matrix visualization")
print("    7. roc_auc_curve.png - ROC AUC curve with metrics")
print("  🤖 Model Files:")
print("    8. LinearSVC_model.pkl - Trained model (joblib)")
print("    9. LinearSVC_model.pth - Trained model (pickle format)")
print("    10. vectorizer.pkl - TF-IDF vectorizer")
print("    11. model_metadata.pkl - Model metadata")

# ==================== 11. INTERFACE READY FUNCTION ====================

def predict_comment(text, model_path='LinearSVC_model.pth', vectorizer_path='vectorizer.pkl'):
    """Function to predict a single comment (for interface)"""
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
    decision_score = classifier.decision_function(text_vectorized)[0]
    
    # Convert decision score to approximate probability using sigmoid
    prob_hate = 1 / (1 + np.exp(-decision_score))
    
    return {
        'text': text,
        'is_hate_speech': bool(prediction == 1),
        'probability_hate': prob_hate,
        'probability_non_hate': 1 - prob_hate,
        'decision_score': decision_score,
        'prediction': 'Hate Speech' if prediction == 1 else 'Non-Hate Speech'
    }

print("\n" + "="*60)
print("INTERFACE READY")
print("="*60)
print("\nYou can now use the saved model in your interface with:")
print("""
from your_script import predict_comment

# Example usage:
result = predict_comment("Your text here")
print(result)
""")

# Test the interface function
print("\nTesting interface function on first test sample:")
sample_text = X_test.iloc[0]
sample_result = predict_comment(sample_text)
print(f"Text: {sample_text[:100]}...")
print(f"Prediction: {sample_result['prediction']}")
print(f"Probability Hate: {sample_result['probability_hate']:.4f}")

# ==================== 12. SAMPLE PREDICTIONS ====================

print("\n" + "="*60)
print("SAMPLE PREDICTIONS (First 20 test samples)")
print("="*60)

sample_df = eval_df.head(20)
for idx, row in sample_df.iterrows():
    text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
    status = "✓" if row['is_correct'] else "✗"
    print(f"\n{status} Sample {idx+1}:")
    print(f"   Text: {text_preview}")
    print(f"   True: {row['true_label']} | Pred: {row['predicted_label']} | Correct: {row['is_correct']}")
    print(f"   Decision Score: {row['decision_score']:.4f}")