"""
Complete GRU with FastText Multilingual Embeddings for Text Classification - Full Version with ROC
Supports: Arabizi, Algerian Arabic, English, French
Input: CSV files with text and labels (train.csv, validation.csv, test.csv with semicolon separator)
Output: Evaluation CSV file with predictions and metrics + ROC Curve
WITH OPTUNA HYPERPARAMETER OPTIMIZATION - OPTIMIZING FOR VALIDATION LOSS
✅ FORCED UNIDIRECTIONAL GRU (No BiGRU)
✅ FIXED EPOCHS = 20 (No early stopping - runs all epochs)
✅ NO LABEL SMOOTHING (Standard BCE Loss)
✅ REDUCELROnPLATEAU (Adaptive learning rate reduction)
✅ ALL WORDS INCLUDED (No min_freq filtering)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
import fasttext
import warnings
import os
from collections import Counter
from datetime import datetime
import time
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.trial import TrialState

warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")

# Global variables for Optuna
global_train_loader = None
global_val_loader = None
global_vocab_size = None
global_embedding_matrix = None

# ============================================================================
# PART 1: FASTTEXT ARABIC EMBEDDINGS
# ============================================================================

class FastTextArabicEmbeddings:
    """FastText Arabic embeddings manager (cc.ar.300.bin)"""
    
    def __init__(self, embedding_dim=300, model_path='/content/drive/MyDrive/BigData_11_2026/data/cc.ar.300.bin'):
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.model = None
        
    def load_fasttext_model(self):
        """Load pre-trained FastText Arabic model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading FastText Arabic model from {self.model_path}...")
                self.model = fasttext.load_model(self.model_path)
                print("FastText Arabic model loaded successfully!")
                print("Support: Arabic, Arabizi and multilingual text")
                return True
            else:
                print(f"Model file {self.model_path} not found!")
                print("Please ensure cc.ar.300.bin is in the data directory")
                self.model = None
                return False
        except Exception as e:
            print(f"Error loading FastText model: {e}")
            print("Using random embeddings instead")
            self.model = None
            return False
    
    def get_word_vector(self, word):
        """Get vector for a single word"""
        if self.model:
            try:
                return self.model.get_word_vector(word)
            except:
                return np.random.randn(self.embedding_dim)
        else:
            return np.random.randn(self.embedding_dim)
    
    def get_embedding_matrix(self, word_to_idx):
        """Create embedding matrix for vocabulary"""
        vocab_size = len(word_to_idx)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))
        
        print(f"Building embedding matrix for {vocab_size} words...")
        words_processed = 0
        words_found = 0
        
        for word, idx in word_to_idx.items():
            if word not in ['<PAD>', '<UNK>']:
                if self.model:
                    try:
                        vector = self.model.get_word_vector(word)
                        embedding_matrix[idx] = vector
                        words_found += 1
                    except:
                        embedding_matrix[idx] = np.random.randn(self.embedding_dim)
                else:
                    embedding_matrix[idx] = np.random.randn(self.embedding_dim)
                words_processed += 1
                if words_processed % 5000 == 0:
                    print(f"Processed {words_processed}/{len(word_to_idx)} words")
        
        # Set padding index to zeros
        embedding_matrix[0] = np.zeros(self.embedding_dim)
        
        print(f"Embedding matrix built: {embedding_matrix.shape}")
        if self.model:
            print(f"Words found in FastText: {words_found}/{words_processed} ({words_found/words_processed*100:.1f}%)")
        else:
            print(f"Using random embeddings for all {words_processed} words")
        return embedding_matrix

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

class TextDataset(Dataset):
    """Custom dataset for text classification"""
    
    def __init__(self, texts, labels, word_to_idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in text.split()]
        
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
            seq_len = self.max_len
        else:
            seq_len = len(indices)
            indices = indices + [self.word_to_idx['<PAD>']] * (self.max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(seq_len, dtype=torch.long), torch.tensor(label, dtype=torch.float)
    
def build_vocabulary(texts):
    """Build vocabulary from texts - INCLUDES ALL WORDS (NO frequency filtering)"""
    print("Building vocabulary...")
    
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    # REMOVED: if count >= min_freq condition - now includes EVERY word
    for word, count in word_counts.items():
        vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)} (ALL words included - NO frequency filtering)")
    print(f"Total unique words found: {len(word_counts)}")
    return vocab

def load_pre_split_data(train_path='/content/drive/MyDrive/BigData_11_2026/data/train.csv', 
                        val_path='/content/drive/MyDrive/BigData_11_2026/data/validation.csv', 
                        test_path='/content/drive/MyDrive/BigData_11_2026/data/test.csv',
                        text_column='text', 
                        label_column='label'):
    """Load pre-split CSV files (train, validation, test)"""
    
    print(f"\nLoading pre-split datasets...")
    
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path, sep=';')
    print(f"Loaded {len(train_df)} training samples")
    
    print(f"\nLoading validation data from: {val_path}")
    val_df = pd.read_csv(val_path, sep=';')
    print(f"Loaded {len(val_df)} validation samples")
    
    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path, sep=';')
    print(f"Loaded {len(test_df)} test samples")
    
    for df, name in [(train_df, "Training"), (val_df, "Validation"), (test_df, "Test")]:
        if text_column not in df.columns:
            raise ValueError(f"{name}: Text column '{text_column}' not found. Available: {df.columns.tolist()}")
        if label_column not in df.columns:
            raise ValueError(f"{name}: Label column '{label_column}' not found. Available: {df.columns.tolist()}")
    
    X_train = train_df[text_column].astype(str).tolist()
    y_train = train_df[label_column].tolist()
    
    X_val = val_df[text_column].astype(str).tolist()
    y_val = val_df[label_column].tolist()
    
    X_test = test_df[text_column].astype(str).tolist()
    y_test = test_df[label_column].tolist()
    
    print(f"\nClass distribution - Training:")
    train_counts = pd.Series(y_train).value_counts()
    for label, count in train_counts.items():
        print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\nClass distribution - Validation:")
    val_counts = pd.Series(y_val).value_counts()
    for label, count in val_counts.items():
        print(f"  {label}: {count} ({count/len(y_val)*100:.1f}%)")
    
    print(f"\nClass distribution - Test:")
    test_counts = pd.Series(y_test).value_counts()
    for label, count in test_counts.items():
        print(f"  {label}: {count} ({count/len(y_test)*100:.1f}%)")
    
    label_encoder = LabelEncoder()
    all_labels = y_train + y_val + y_test
    label_encoder.fit(all_labels)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\nData loaded successfully!")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Classes: {label_encoder.classes_}")
    
    return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded, label_encoder

# ============================================================================
# PART 3: GRU MODEL (FORCED UNIDIRECTIONAL - NO BIDIRECTIONAL)
# ============================================================================

class GRUTextClassifier(nn.Module):
    """GRU-based text classifier with FastText embeddings - FORCED UNIDIRECTIONAL GRU"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, 
                 dropout_pre_gru=0.3, dropout_gru=0.5, 
                 dropout_post_gru=0.3,
                 pretrained_embeddings=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = False
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_pre_gru = nn.Dropout(dropout_pre_gru)
        
        if pretrained_embeddings is not None:
            print("Loading FastText Arabic pre-trained embeddings...")
            assert pretrained_embeddings.shape[0] == vocab_size
            assert pretrained_embeddings.shape[1] == embedding_dim
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = False
            print(" FastText embeddings are FROZEN ")
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout_gru if n_layers > 1 else 0,
            batch_first=True
        )
        
        gru_output_dim = hidden_dim
        self.dropout_post_gru = nn.Dropout(dropout_post_gru)
        self.fc = nn.Linear(gru_output_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        embedded = self.dropout_pre_gru(embedded)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, hidden = self.gru(packed_embedded)
        
        hidden = hidden[-1, :, :]
        
        hidden = self.dropout_post_gru(hidden)
        output = self.fc(hidden)
        output = self.sigmoid(output)
        
        return output.squeeze()

# ============================================================================
# PART 4: TRAINING FUNCTIONS WITH COMPLETE METRICS (NO LABEL SMOOTHING)
# ============================================================================

def train_epoch_with_all_metrics(model, dataloader, optimizer, criterion):
    """Train epoch and return ALL metrics (loss, accuracy, precision, recall, f1)"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch_idx, (texts, lengths, labels) in enumerate(dataloader):
        texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        
        loss = criterion(outputs, labels.float())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def evaluate_with_all_metrics(model, dataloader, criterion, threshold=0.5):
    """Evaluate and return ALL metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for texts, lengths, labels in dataloader:
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
            outputs = model(texts, lengths)
            
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            
            probabilities = outputs.cpu().numpy()
            predicted = (outputs > threshold).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels, all_probabilities

def train_model_with_all_metrics(model, train_loader, val_loader, epochs, lr, weight_decay, scheduler_factor=0.5, scheduler_patience=3):
    """
    NO EARLY STOPPING - Runs all specified epochs
    WITH REDUCELROnPLATEAU - Adaptive learning rate
    NO LABEL SMOOTHING - Standard BCE Loss
    """
    criterion = nn.BCELoss()  # ← Standard BCE Loss (no label smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # ADDED: ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Reduce when validation loss stops decreasing
        factor=scheduler_factor,  # Multiply lr by factor (default: 0.5)
        patience=scheduler_patience,  # Number of epochs with no improvement
        min_lr=1e-7          # Minimum learning rate
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    lr_history = []  # Track learning rate changes
    
    # Store ALL epochs
    all_epoch_history = []
    
    os.makedirs('DL Models/GRU with FastText', exist_ok=True)
    
    print("\n" + "="*80)
    print(f"TRAINING FOR {epochs} EPOCHS (No Early Stopping)")
    print("="*80)
    print(f"✅ Standard BCE Loss (No label smoothing)")
    print(f"✅ ReduceLROnPlateau: factor={scheduler_factor}, patience={scheduler_patience}")
    print("="*80)
    print("Metrics tracked for BOTH Training and Validation:")
    print("  📊 Loss | Accuracy | Precision | Recall | F1-Score")
    print("="*80 + "\n")
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch_with_all_metrics(
            model, train_loader, optimizer, criterion
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, _ = evaluate_with_all_metrics(
            model, val_loader, criterion, threshold=0.5
        )
        
        # Update scheduler with validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_reduced = (new_lr != old_lr)
        
        # Record ALL epochs
        current_epoch_data = {
            'epoch': epoch + 1,
            'learning_rate': current_lr,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1,
            'lr_reduced': lr_reduced,
            'is_best': val_loss < best_val_loss
        }
        all_epoch_history.append(current_epoch_data)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'val_precision': val_prec,
                'val_recall': val_rec
            }, 'DL Models/GRU with FastText/best_gru_model.pth')
            
            print(f"\n🌟 EPOCH {epoch+1}/{epochs} - NEW BEST! (Loss: {val_loss:.6f})")
            print(f"   📈 TRAIN: Loss={train_loss:.4f} | Acc={train_acc:.4f} | Prec={train_prec:.4f} | Rec={train_rec:.4f} | F1={train_f1:.4f}")
            print(f"   📉 VALID: Loss={val_loss:.4f} | Acc={val_acc:.4f} | Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}")
            if lr_reduced:
                print(f"   🔽 Learning rate reduced to: {new_lr:.2e}")
            
        else:
            print(f"\n📊 EPOCH {epoch+1}/{epochs} - Current Loss: {val_loss:.6f} (Best: {best_val_loss:.6f})")
            print(f"   📈 TRAIN: Loss={train_loss:.4f} | Acc={train_acc:.4f} | Prec={train_prec:.4f} | Rec={train_rec:.4f} | F1={train_f1:.4f}")
            print(f"   📉 VALID: Loss={val_loss:.4f} | Acc={val_acc:.4f} | Prec={val_prec:.4f} | Rec={val_rec:.4f} | F1={val_f1:.4f}")
            if lr_reduced:
                print(f"   🔽 Learning rate reduced to: {new_lr:.2e}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"📊 Total epochs run: {len(all_epoch_history)}")
    print(f"🏆 Best epoch: {best_epoch} (Lowest validation loss: {best_val_loss:.6f})")
    print(f"📈 Learning rate schedule:")
    unique_lrs = sorted(set(lr_history))
    for lr_val in unique_lrs:
        count = lr_history.count(lr_val)
        print(f"   - {lr_val:.2e} applied for {count} epochs")
    print("="*80)
    
    return best_val_loss, best_epoch, all_epoch_history

def objective(trial):
    """Objective function for Optuna hyperparameter optimization"""
    
    global global_train_loader, global_val_loader, global_vocab_size, global_embedding_matrix
    
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128, step=16)
    n_layers = trial.suggest_int('n_layers', 1, 2)
    dropout_pre_gru = trial.suggest_float('dropout_pre_gru', 0.2, 0.6, step=0.1)
    dropout_gru = trial.suggest_float('dropout_gru', 0.3, 0.7, step=0.1)
    dropout_post_gru = trial.suggest_float('dropout_post_gru', 0.2, 0.6, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # NEW: Scheduler parameters
    scheduler_factor = trial.suggest_float('scheduler_factor', 0.3, 0.8, step=0.1)
    scheduler_patience = trial.suggest_int('scheduler_patience', 2, 5)
    
    epochs = 20  # ← FIXED: 20 epochs
    
    train_loader = DataLoader(global_train_loader.dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(global_val_loader.dataset, batch_size=batch_size, shuffle=False)
    
    model = GRUTextClassifier(
        vocab_size=global_vocab_size,
        embedding_dim=300,
        hidden_dim=hidden_dim,
        output_dim=1,
        n_layers=n_layers,
        dropout_pre_gru=dropout_pre_gru,
        dropout_gru=dropout_gru,
        dropout_post_gru=dropout_post_gru,
        pretrained_embeddings=global_embedding_matrix
    ).to(device)
    
    best_val_loss, _, _ = train_model_with_all_metrics(
        model, train_loader, val_loader, epochs, learning_rate, weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience
    )
    
    return best_val_loss

# ============================================================================
# PART 5: OPTUNA OPTIMIZATION WRAPPER
# ============================================================================

def run_optuna_optimization(train_loader, val_loader, vocab_size, embedding_matrix, n_trials=30):
    """Run Optuna to find best hyperparameters by minimizing validation loss"""
    
    global global_train_loader, global_val_loader, global_vocab_size, global_embedding_matrix
    
    global_train_loader = train_loader
    global_val_loader = val_loader
    global_vocab_size = vocab_size
    global_embedding_matrix = embedding_matrix
    
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Running {n_trials} trials to find best parameters...")
    print("OPTIMIZING FOR: VALIDATION LOSS (MINIMIZATION)")
    print("✅ FORCED UNIDIRECTIONAL GRU (No BiGRU)")
    print("✅ TRAINABLE EMBEDDINGS (Fine-tuning enabled)")
    print("✅ STANDARD BCE LOSS (No label smoothing)")
    print("✅ REDUCELROnPLATEAU scheduler")
    print("✅ FIXED 20 EPOCHS (No early stopping)")
    print("="*70)
    
    study = optuna.create_study(
        direction='minimize',
        study_name='gru_text_classification_loss',
        storage=None,
        load_if_exists=False
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_trial = study.best_trial
    
    print("\n" + "="*70)
    print("OPTUNA OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Best Validation Loss achieved: {best_trial.value:.4f}")
    print("\nBest hyperparameters found:")
    print("-"*50)
    
    best_params = {}
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
        best_params[key] = value
    best_params['epochs'] = 20
    
    results_dir = 'DL Models/GRU with FastText/Results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{results_dir}/optuna_trials_{timestamp}.csv", index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot([t.value for t in study.trials if t.state == TrialState.COMPLETE], 'b-', linewidth=1)
    axes[0].scatter(range(len([t for t in study.trials if t.state == TrialState.COMPLETE])), 
                   [t.value for t in study.trials if t.state == TrialState.COMPLETE], 
                   c='blue', s=30, alpha=0.6)
    axes[0].axhline(y=best_trial.value, color='r', linestyle='--', label=f'Best Loss: {best_trial.value:.4f}')
    axes[0].set_xlabel('Trial', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Optuna Optimization Progress', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    importances = optuna.importance.get_param_importances(study)
    params = list(importances.keys())
    values = list(importances.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
    axes[1].barh(params, values, color=colors)
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/optuna_results_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Optuna results saved to: {results_dir}/optuna_trials_{timestamp}.csv")
    print(f"✓ Optimization plot saved to: {results_dir}/optuna_results_{timestamp}.png")
    
    return best_params, study

# ============================================================================
# PART 6: METRICS PLOTTING (Normal - shows ALL epochs)
# ============================================================================

def plot_metrics(all_epoch_history, save_path):
    """
    Plot ALL metrics - normal plotting, all epochs shown
    """
    
    if not all_epoch_history:
        print("Warning: No epoch data to plot!")
        return None
    
    # Extract all epoch data
    epochs = [e['epoch'] for e in all_epoch_history]
    
    train_loss = [e['train_loss'] for e in all_epoch_history]
    train_acc = [e['train_accuracy'] for e in all_epoch_history]
    train_f1 = [e['train_f1'] for e in all_epoch_history]
    train_prec = [e['train_precision'] for e in all_epoch_history]
    train_rec = [e['train_recall'] for e in all_epoch_history]
    
    val_loss = [e['val_loss'] for e in all_epoch_history]
    val_acc = [e['val_accuracy'] for e in all_epoch_history]
    val_f1 = [e['val_f1'] for e in all_epoch_history]
    val_prec = [e['val_precision'] for e in all_epoch_history]
    val_rec = [e['val_recall'] for e in all_epoch_history]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # 1. Loss Curve
    axes[0].plot(epochs, train_loss, 'b-o', linewidth=2, markersize=6, label='Training', alpha=0.8)
    axes[0].plot(epochs, val_loss, 'r-s', linewidth=2.5, markersize=8, label='Validation', alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)
    
    # 2. Accuracy Curve
    axes[1].plot(epochs, train_acc, 'b-o', linewidth=2, markersize=6, label='Training', alpha=0.8)
    axes[1].plot(epochs, val_acc, 'r-s', linewidth=2.5, markersize=8, label='Validation', alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curve', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)
    axes[1].set_ylim([0.5, 1.05])
    
    # 3. F1 Score Curve
    axes[2].plot(epochs, train_f1, 'b-o', linewidth=2, markersize=6, label='Training', alpha=0.8)
    axes[2].plot(epochs, val_f1, 'r-s', linewidth=2.5, markersize=8, label='Validation', alpha=0.8)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('F1-Score Curve', fontsize=13, fontweight='bold')
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(epochs)
    axes[2].set_ylim([0.5, 1.05])
    
    # 4. Precision Curve
    axes[3].plot(epochs, train_prec, 'b-o', linewidth=2, markersize=6, label='Training', alpha=0.8)
    axes[3].plot(epochs, val_prec, 'r-s', linewidth=2.5, markersize=8, label='Validation', alpha=0.8)
    axes[3].set_xlabel('Epoch', fontsize=12)
    axes[3].set_ylabel('Precision', fontsize=12)
    axes[3].set_title('Precision Curve', fontsize=13, fontweight='bold')
    axes[3].legend(loc='lower right', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xticks(epochs)
    axes[3].set_ylim([0.5, 1.05])
    
    # 5. Recall Curve
    axes[4].plot(epochs, train_rec, 'b-o', linewidth=2, markersize=6, label='Training', alpha=0.8)
    axes[4].plot(epochs, val_rec, 'r-s', linewidth=2.5, markersize=8, label='Validation', alpha=0.8)
    axes[4].set_xlabel('Epoch', fontsize=12)
    axes[4].set_ylabel('Recall', fontsize=12)
    axes[4].set_title('Recall Curve', fontsize=13, fontweight='bold')
    axes[4].legend(loc='lower right', fontsize=10)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xticks(epochs)
    axes[4].set_ylim([0.5, 1.05])
    
    # 6. Hide empty subplot
    axes[5].set_visible(False)
    
    plt.suptitle('Training and Validation Metrics - GRU with FastText Embeddings', 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Metrics plot saved: {save_path}")
    print(f"   📊 Shows ALL {len(all_epoch_history)} epochs")
    
    # Save full epoch data
    all_metrics_df = pd.DataFrame(all_epoch_history)
    all_metrics_df.to_csv(save_path.replace('.png', '_data.csv'), index=False)
    print(f"✅ Epoch data saved: {save_path.replace('.png', '_data.csv')}")
    
    return all_metrics_df

# ============================================================================
# PART 7: ROC CURVE FUNCTIONS (PROFESSIONAL STYLE)
# ============================================================================

def plot_roc_curve(true_labels, probabilities, label_encoder, save_path):
    """Plot ROC curve with professional styling"""
    
    true_labels_array = np.array(true_labels).astype(int)
    probabilities_array = np.array(probabilities)
    
    fpr, tpr, thresholds = roc_curve(true_labels_array, probabilities_array)
    auc = roc_auc_score(true_labels_array, probabilities_array)
    
    if auc == 0.5:
        auc_interpretation = "No discrimination"
    elif 0.5 < auc <= 0.6:
        auc_interpretation = "Poor discrimination"
    elif 0.6 < auc <= 0.7:
        auc_interpretation = "Acceptable discrimination"
    elif 0.7 < auc <= 0.8:
        auc_interpretation = "Excellent discrimination"
    else:
        auc_interpretation = "Outstanding discrimination"
    
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_specificity = 1 - optimal_fpr
    
    predictions_at_optimal = (probabilities_array >= optimal_threshold).astype(int)
    accuracy_at_optimal = accuracy_score(true_labels_array, predictions_at_optimal)
    
    print(f"\n" + "="*70)
    print("ROC ANALYSIS (Receiver Operating Characteristic)")
    print("="*70)
    print(f"AUC (Area Under Curve): {auc:.4f}")
    print(f"Interpretation: {auc_interpretation}")
    print(f"\nOptimal threshold (Youden's index): {optimal_threshold:.4f}")
    print(f"  - Sensitivity (Recall) at this threshold: {optimal_tpr:.4f}")
    print(f"  - Specificity at this threshold: {optimal_specificity:.4f}")
    print(f"  - Accuracy at this threshold: {accuracy_at_optimal:.4f}")
    print(f"  - Youden's index (J): {youden_j[optimal_idx]:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random classifier (AUC = 0.5)')
    ax.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, markerfacecolor='red', 
            markeredgecolor='darkred', markeredgewidth=1.5, zorder=10)
    
    ax.plot([optimal_fpr, optimal_fpr], [0, optimal_tpr], 'r:', linewidth=1, alpha=0.5)
    ax.plot([0, optimal_fpr], [optimal_tpr, optimal_tpr], 'r:', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve - GRU with FastText', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(labelsize=10)
    
    bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='blue', alpha=0.85)
    ax.annotate(f'Optimal cut-off = {optimal_threshold:.3f}\nSensitivity = {optimal_tpr:.3f}\nSpecificity = {optimal_specificity:.3f}',
                xy=(optimal_fpr, optimal_tpr), 
                xytext=(optimal_fpr + 0.15, optimal_tpr - 0.12),
                fontsize=9, fontweight='normal',
                bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))
    
    auc_text = f'AUC = {auc:.4f}\n{auc_interpretation}'
    ax.text(0.02, 0.02, auc_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
    
    legend = ax.legend(loc='lower right', framealpha=0.95, edgecolor='black', 
                       fancybox=False, fontsize=10)
    legend.get_frame().set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nROC graph saved: {save_path}")
    
    return {
        'auc': auc,
        'auc_interpretation': auc_interpretation,
        'optimal_threshold': optimal_threshold,
        'optimal_sensitivity': optimal_tpr,
        'optimal_specificity': optimal_specificity,
        'optimal_youden': youden_j[optimal_idx],
        'accuracy_at_optimal': accuracy_at_optimal,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

def save_roc_data(fpr, tpr, thresholds, auc, output_path):
    """Save ROC curve data to CSV"""
    
    min_len = min(len(fpr), len(tpr), len(thresholds))
    fpr = fpr[:min_len]
    tpr = tpr[:min_len]
    thresholds = thresholds[:min_len]
    
    roc_df = pd.DataFrame({
        'false_positive_rate': fpr,
        'true_positive_rate': tpr,
        'threshold': thresholds
    })
    roc_df['specificity'] = 1 - roc_df['false_positive_rate']
    roc_df['sensitivity'] = roc_df['true_positive_rate']
    roc_df['youden_j'] = roc_df['true_positive_rate'] - roc_df['false_positive_rate']
    roc_df.to_csv(output_path, index=False)
    print(f"ROC data saved: {output_path}")
    
    return roc_df

# ============================================================================
# PART 8: EVALUATION AND VISUALIZATION
# ============================================================================

def save_evaluation_results(test_texts, true_labels, predictions, probabilities, label_encoder, output_path):
    results_df = pd.DataFrame({
        'text': test_texts,
        'true_label': label_encoder.inverse_transform(np.array(true_labels).astype(int)),
        'predicted_label': label_encoder.inverse_transform(np.array(predictions).astype(int)),
        'is_correct': (np.array(true_labels).astype(int) == np.array(predictions).astype(int)),
        'probability_positive': np.array(probabilities),
        'probability_negative': 1 - np.array(probabilities)
    })
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved: {output_path}")
    return results_df

def calculate_metrics(true_labels, predictions, label_encoder):
    true_labels_array = np.array(true_labels).astype(int)
    predictions_array = np.array(predictions).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(true_labels_array, predictions_array),
        'precision': precision_score(true_labels_array, predictions_array, average='binary', zero_division=0),
        'recall': recall_score(true_labels_array, predictions_array, average='binary', zero_division=0),
        'f1_score': f1_score(true_labels_array, predictions_array, average='binary', zero_division=0)
    }
    return metrics

def save_confusion_matrix(true_labels, predictions, label_encoder, output_path):
    cm = confusion_matrix(true_labels, predictions)
    class_names = label_encoder.classes_
    
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_path)
    print(f"Confusion matrix saved: {output_path}")
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print(cm_df)
    
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    return cm_df

def plot_confusion_matrix_graph(true_labels, predictions, label_encoder, save_path):
    cm = confusion_matrix(true_labels, predictions)
    class_names = label_encoder.classes_
    
    per_class_recall = []
    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        per_class_recall.append(recall)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    
    recall_text = f"Avg Recall: {np.mean(per_class_recall):.3f}\n"
    for i, (name, rec) in enumerate(zip(class_names, per_class_recall)):
        recall_text += f"{name}: {rec:.3f}  "
    ax.text(1.05, 0.5, recall_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Prediction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - GRU with FastText', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix graph: {save_path}")
    
    return per_class_recall

def save_misclassification_csvs(test_texts, true_labels, predictions, probabilities, label_encoder, output_dir, timestamp):
    true_labels_array = np.array(true_labels).astype(int)
    predictions_array = np.array(predictions).astype(int)
    probabilities_array = np.array(probabilities)
    
    original_true_labels = label_encoder.inverse_transform(true_labels_array)
    original_pred_labels = label_encoder.inverse_transform(predictions_array)
    
    full_df = pd.DataFrame({
        'text': test_texts,
        'true_label': original_true_labels,
        'true_label_numeric': true_labels_array,
        'predicted_label': original_pred_labels,
        'predicted_label_numeric': predictions_array,
        'probability_positive': probabilities_array,
        'probability_negative': 1 - probabilities_array,
        'is_correct': (true_labels_array == predictions_array)
    })
    
    false_positives = full_df[(full_df['true_label_numeric'] == 0) & (full_df['predicted_label_numeric'] == 1)]
    false_negatives = full_df[(full_df['true_label_numeric'] == 1) & (full_df['predicted_label_numeric'] == 0)]
    
    if len(false_positives) > 0:
        fp_path = os.path.join(output_dir, f'false_positives_{timestamp}.csv')
        false_positives.to_csv(fp_path, index=False, encoding='utf-8-sig')
        print(f"False positives: {len(false_positives)} -> {fp_path}")
    
    if len(false_negatives) > 0:
        fn_path = os.path.join(output_dir, f'false_negatives_{timestamp}.csv')
        false_negatives.to_csv(fn_path, index=False, encoding='utf-8-sig')
        print(f"False negatives: {len(false_negatives)} -> {fn_path}")
    
    return false_positives, false_negatives

def save_metrics_report(metrics, output_path):
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)
    print(f"Report saved: {output_path}")
    
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

# ============================================================================
# PART 9: PERMUTATION TEST AND STATISTICAL TESTS
# ============================================================================

def permutation_test_accuracy(model, test_loader, criterion, n_permutations=999, threshold=0.5):
    """Permutation test to check if accuracy is significantly better than random"""
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TEST (RANDOMIZATION)")
    print("="*70)
    
    model.eval()
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for texts, lengths, labels in test_loader:
            texts, lengths = texts.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    observed_predictions = (all_probabilities >= threshold).astype(int)
    observed_accuracy = accuracy_score(all_labels, observed_predictions)
    
    print(f"Observed accuracy: {observed_accuracy:.4f}")
    
    null_distribution = []
    for i in range(n_permutations):
        shuffled_labels = np.random.permutation(all_labels)
        permuted_accuracy = accuracy_score(shuffled_labels, observed_predictions)
        null_distribution.append(permuted_accuracy)
        if (i + 1) % 100 == 0:
            print(f"  Permutations: {i+1}/{n_permutations}")
    
    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= observed_accuracy)
    
    if p_value < 0.001:
        significance = "*** (very significant)"
    elif p_value < 0.01:
        significance = "** (significant)"
    elif p_value < 0.05:
        significance = "* (significant)"
    else:
        significance = "n.s. (not significant)"
    
    print(f"\np-value: {p_value:.4f} - {significance}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_distribution, bins=30, alpha=0.7, color='gray', edgecolor='black', label='Null distribution')
    ax.axvline(observed_accuracy, color='red', linewidth=2, label=f'Observed (p={p_value:.4f})')
    ax.axvline(null_distribution.mean(), color='blue', linestyle='--', label=f'Mean = {null_distribution.mean():.3f}')
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Randomization Test - Null Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return {
        'observed_accuracy': observed_accuracy,
        'p_value': p_value,
        'significance': significance,
        'null_mean': null_distribution.mean(),
        'null_std': null_distribution.std(),
        'figure': fig
    }

def mantel_test_predictions(true_labels, predictions, probabilities, n_permutations=999):
    """Mantel test for distance matrix correlation"""
    print("\n" + "="*70)
    print("MANTEL TEST - DISTANCE CORRELATION")
    print("="*70)
    
    n = len(true_labels)
    from scipy.spatial.distance import pdist
    
    true_dist = pdist(true_labels.reshape(-1, 1), metric='hamming')
    prob_dist = pdist(probabilities.reshape(-1, 1), metric='euclidean')
    pred_dist = pdist(predictions.reshape(-1, 1), metric='hamming')
    
    observed_corr_prob = np.corrcoef(true_dist, prob_dist)[0, 1]
    observed_corr_pred = np.corrcoef(true_dist, pred_dist)[0, 1]
    
    print(f"Correlation (probabilities vs true): {observed_corr_prob:.4f}")
    print(f"Correlation (predictions vs true): {observed_corr_pred:.4f}")
    
    null_correlations_prob = []
    null_correlations_pred = []
    
    for i in range(n_permutations):
        perm_idx = np.random.permutation(n)
        permuted_true_dist = pdist(true_labels[perm_idx].reshape(-1, 1), metric='hamming')
        corr_prob = np.corrcoef(permuted_true_dist, prob_dist)[0, 1]
        corr_pred = np.corrcoef(permuted_true_dist, pred_dist)[0, 1]
        null_correlations_prob.append(corr_prob)
        null_correlations_pred.append(corr_pred)
    
    null_correlations_prob = np.array(null_correlations_prob)
    null_correlations_pred = np.array(null_correlations_pred)
    
    p_value_prob = np.mean(null_correlations_prob >= observed_corr_prob)
    p_value_pred = np.mean(null_correlations_pred >= observed_corr_pred)
    
    print(f"\np-value (probabilities): {p_value_prob:.4f}")
    print(f"p-value (predictions): {p_value_pred:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(null_correlations_prob, bins=30, alpha=0.7, color='gray', edgecolor='black')
    axes[0].axvline(observed_corr_prob, color='red', linewidth=2)
    axes[0].set_title(f'Probabilities (p={p_value_prob:.4f})', fontsize=12)
    axes[0].set_xlabel('Mantel Correlation')
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(null_correlations_pred, bins=30, alpha=0.7, color='gray', edgecolor='black')
    axes[1].axvline(observed_corr_pred, color='red', linewidth=2)
    axes[1].set_title(f'Predictions (p={p_value_pred:.4f})', fontsize=12)
    axes[1].set_xlabel('Mantel Correlation')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    
    return {
        'observed_corr_prob': observed_corr_prob,
        'observed_corr_pred': observed_corr_pred,
        'p_value_prob': p_value_prob,
        'p_value_pred': p_value_pred,
        'figure': fig
    }

def test_error_structure(true_labels, predictions, n_permutations=999):
    """Test if errors are structured or random"""
    print("\n" + "="*70)
    print("ERROR STRUCTURE TEST")
    print("="*70)
    
    errors = (true_labels != predictions).astype(int)
    n = len(errors)
    
    observed_cooccurrence = 0
    for i in range(n):
        for j in range(i+1, n):
            if errors[i] == 1 and errors[j] == 1:
                observed_cooccurrence += 1
    
    print(f"Observed error pairs: {observed_cooccurrence}")
    
    null_cooccurrence = []
    for i in range(n_permutations):
        shuffled_errors = np.random.permutation(errors)
        cooccur = 0
        for j in range(n):
            for k in range(j+1, n):
                if shuffled_errors[j] == 1 and shuffled_errors[k] == 1:
                    cooccur += 1
        null_cooccurrence.append(cooccur)
    
    null_cooccurrence = np.array(null_cooccurrence)
    p_value = np.mean(null_cooccurrence >= observed_cooccurrence)
    
    print(f"Null mean: {null_cooccurrence.mean():.1f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("→ Errors tend to cluster together")
    else:
        print("→ Random error distribution")
    
    return {
        'observed_cooccurrence': observed_cooccurrence,
        'p_value': p_value,
        'null_mean': null_cooccurrence.mean()
    }

# ============================================================================
# PART 10: MAIN EXECUTION
# ============================================================================

def train_final_model_with_best_params(train_loader, val_loader, vocab_size, embedding_matrix, best_params):
    """Train final model with best hyperparameters found by Optuna"""
    
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*70)
    
    model = GRUTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=best_params['hidden_dim'],
        output_dim=1,
        n_layers=best_params['n_layers'],
        dropout_pre_gru=best_params['dropout_pre_gru'],
        dropout_gru=best_params['dropout_gru'],
        dropout_post_gru=best_params['dropout_post_gru'],
        pretrained_embeddings=embedding_matrix
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Force 20 epochs for final training with scheduler
    best_val_loss, best_epoch, all_epoch_history = train_model_with_all_metrics(
        model, train_loader, val_loader, 
        epochs=20,
        lr=best_params['learning_rate'], 
        weight_decay=best_params['weight_decay'],
        scheduler_factor=best_params.get('scheduler_factor', 0.5),
        scheduler_patience=best_params.get('scheduler_patience', 3)
    )
    
    if os.path.exists('DL Models/GRU with FastText/best_gru_model.pth'):
        checkpoint = torch.load('DL Models/GRU with FastText/best_gru_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_epoch} with validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, all_epoch_history

def main():
    """Main execution function with Optuna optimization"""
    
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("="*70)
    print("GRU with FastText Arabic Embeddings - Text Classification with OPTUNA")
    print("OPTIMIZING FOR: VALIDATION LOSS (MINIMIZATION)")
    print("✅ FORCED UNIDIRECTIONAL GRU (No BiGRU)")
    print("✅ TRAINABLE EMBEDDINGS (Fine-tuning enabled)")
    print("✅ STANDARD BCE LOSS (No label smoothing)")
    print("✅ REDUCELROnPLATEAU scheduler")
    print("✅ ALL WORDS INCLUDED (NO min_freq filtering)")
    print("✅ FIXED 20 EPOCHS (No early stopping)")
    print("="*70)
    print(f"\nStart: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_data_dir = '/content/drive/MyDrive/BigData_11_2026/data'
    train_path = f'{base_data_dir}/train.csv'
    val_path = f'{base_data_dir}/validation.csv'
    test_path = f'{base_data_dir}/test.csv'
    output_dir = 'DL Models/GRU with FastText'
    
    missing_files = []
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print(f"\nError: Missing files: {missing_files}")
        print("Please check that your data files exist in the correct location.")
        return None, None, None, None, None
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_pre_split_data(
        train_path, val_path, test_path, 'text', 'label'
    )
    
    all_texts = X_train
    # REMOVED: min_freq parameter - now includes ALL words
    vocab = build_vocabulary(all_texts)
    
    results_dir = os.path.join(output_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open(f'{results_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("\nInitializing FastText embeddings...")
    ft = FastTextArabicEmbeddings(embedding_dim=300, model_path=f'{base_data_dir}/cc.ar.300.bin')
    ft.load_fasttext_model()
    embedding_matrix = ft.get_embedding_matrix(vocab)
    
    print("\nCreating datasets...")
    max_sequence_length = 100
    
    train_dataset = TextDataset(X_train, y_train, vocab, max_len=max_sequence_length)
    val_dataset = TextDataset(X_val, y_val, vocab, max_len=max_sequence_length)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len=max_sequence_length)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n" + "="*70)
    print("STARTING OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    n_trials = 30
    best_params, study = run_optuna_optimization(
        train_loader, val_loader, len(vocab), embedding_matrix, n_trials=n_trials
    )
    
    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(f"{results_dir}/best_hyperparameters_{timestamp}.csv", index=False)
    
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH OPTUNA-BEST PARAMETERS")
    print("="*70)
    
    train_loader_opt = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader_opt = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    trained_model, all_epoch_history = train_final_model_with_best_params(
        train_loader_opt, val_loader_opt, len(vocab), embedding_matrix, best_params
    )
    
    if all_epoch_history:
        all_metrics_df = plot_metrics(all_epoch_history, f"{results_dir}/metrics_curves_{timestamp}.png")
        print(f"\n✓ Plotted ALL {len(all_epoch_history)} epochs")
        best_epoch_idx = np.argmin([e['val_loss'] for e in all_epoch_history])
        print(f"✓ Best epoch: {all_epoch_history[best_epoch_idx]['epoch']} with val_loss: {all_epoch_history[best_epoch_idx]['val_loss']:.4f}")
    else:
        print("\n⚠️ No epochs to plot!")
    
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    print("\n" + "="*70)
    print("ROC ANALYSIS ON VALIDATION SET")
    print("="*70)
    
    criterion = nn.BCELoss()  # Standard BCE Loss
    
    val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels, val_probabilities = evaluate_with_all_metrics(
        trained_model, val_loader_opt, criterion, threshold=0.5
    )
    
    roc_results = plot_roc_curve(
        val_labels, val_probabilities, label_encoder,
        f"{results_dir}/roc_curve_{timestamp}.png"
    )
    
    roc_data_df = save_roc_data(
        roc_results['fpr'], roc_results['tpr'], roc_results['thresholds'],
        roc_results['auc'], f"{results_dir}/roc_data_{timestamp}.csv"
    )
    
    optimal_threshold = roc_results['optimal_threshold']
    print(f"\n✓ Optimal threshold found on validation: {optimal_threshold:.4f}")
    
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET WITH OPTIMAL THRESHOLD")
    print("="*70)
    
    test_loss, test_acc, test_prec, test_rec, test_f1, predictions, true_labels, probabilities = evaluate_with_all_metrics(
        trained_model, test_loader, criterion, threshold=optimal_threshold
    )
    
    print(f"\nTEST SET PERFORMANCE:")
    print(f"  Loss:      {test_loss:.6f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    test_probs_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_probability': probabilities,
        'predicted_class': predictions
    })
    test_probs_df.to_csv(f"{results_dir}/test_predictions_{timestamp}.csv", index=False)
    print(f"Test predictions saved: {results_dir}/test_predictions_{timestamp}.csv")
    
    roc_summary_df = pd.DataFrame([{
        'auc': roc_results['auc'],
        'auc_interpretation': roc_results['auc_interpretation'],
        'optimal_threshold': optimal_threshold,
        'optimal_sensitivity': roc_results['optimal_sensitivity'],
        'optimal_specificity': roc_results['optimal_specificity'],
        'optimal_youden': roc_results['optimal_youden'],
        'test_accuracy_at_optimal': test_acc,
        'test_precision_at_optimal': test_prec,
        'test_recall_at_optimal': test_rec,
        'test_f1_at_optimal': test_f1,
        'scheduler_factor': best_params.get('scheduler_factor', 0.5),
        'scheduler_patience': best_params.get('scheduler_patience', 3)
    }])
    roc_summary_df.to_csv(f"{results_dir}/roc_summary_{timestamp}.csv", index=False)
    print(f"ROC summary saved: {results_dir}/roc_summary_{timestamp}.csv")
    
    save_misclassification_csvs(X_test, true_labels, predictions, probabilities, 
                                label_encoder, results_dir, timestamp)
    
    results_df = save_evaluation_results(X_test, true_labels, predictions, probabilities, 
                                         label_encoder, f"{results_dir}/evaluation_results_{timestamp}.csv")
    
    metrics = calculate_metrics(true_labels, predictions, label_encoder)
    save_metrics_report(metrics, f"{results_dir}/metrics_report_{timestamp}.csv")
    
    save_confusion_matrix(true_labels, predictions, label_encoder, 
                         f"{results_dir}/confusion_matrix_{timestamp}.csv")
    plot_confusion_matrix_graph(true_labels, predictions, label_encoder,
                                f"{results_dir}/confusion_matrix_{timestamp}.png")
    
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    perm_results = permutation_test_accuracy(
        trained_model, test_loader, criterion,
        n_permutations=999, threshold=optimal_threshold
    )
    plt.savefig(f"{results_dir}/permutation_test_{timestamp}.png")
    plt.close()
    
    mantel_results = mantel_test_predictions(
        np.array(true_labels),
        np.array(predictions),
        np.array(probabilities),
        n_permutations=999
    )
    plt.savefig(f"{results_dir}/mantel_test_{timestamp}.png")
    plt.close()
    
    error_results = test_error_structure(
        np.array(true_labels),
        np.array(predictions),
        n_permutations=999
    )
    
    stats_results = {
        'permutation_test': {
            'observed_accuracy': perm_results['observed_accuracy'],
            'p_value': perm_results['p_value'],
            'significance': perm_results['significance'],
            'null_mean': perm_results['null_mean'],
            'null_std': perm_results['null_std']
        },
        'mantel_test': {
            'correlation_probabilities': mantel_results['observed_corr_prob'],
            'correlation_predictions': mantel_results['observed_corr_pred'],
            'p_value_probabilities': mantel_results['p_value_prob'],
            'p_value_predictions': mantel_results['p_value_pred']
        },
        'error_structure': {
            'observed_cooccurrence': error_results['observed_cooccurrence'],
            'p_value': error_results['p_value'],
            'null_mean': error_results['null_mean']
        }
    }
    
    with open(f"{results_dir}/statistical_tests_{timestamp}.pkl", 'wb') as f:
        pickle.dump(stats_results, f)
    
    end_time = time.time()
    total_time_seconds = end_time - start_time
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"📊 Total epochs run: {len(all_epoch_history)}")
    best_epoch_idx = np.argmin([e['val_loss'] for e in all_epoch_history]) if all_epoch_history else 0
    best_epoch_num = all_epoch_history[best_epoch_idx]['epoch'] if all_epoch_history else 0
    print(f"🏆 Best epoch: {best_epoch_num}")
    print(f"\n🏆 TEST SET PERFORMANCE (at threshold {optimal_threshold:.4f}):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\n📊 ROC-AUC (Validation): {roc_results['auc']:.4f} - {roc_results['auc_interpretation']}")
    print(f"\n🔧 TRAINING CONFIGURATION:")
    print(f"  Loss Function: Standard BCE Loss (No label smoothing)")
    print(f"  Scheduler: ReduceLROnPlateau (factor={best_params.get('scheduler_factor', 0.5)}, patience={best_params.get('scheduler_patience', 3)})")
    print(f"\n🔬 STATISTICAL TESTS:")
    print(f"  Permutation test p-value: {perm_results['p_value']:.4f} - {perm_results['significance']}")
    print(f"  Mantel test (probabilities): r={mantel_results['observed_corr_prob']:.4f}, p={mantel_results['p_value_prob']:.4f}")
    print(f"  Error structure p-value: {error_results['p_value']:.4f}")
    print(f"\n⏱️  Total time: {total_time_seconds:.2f}s ({total_time_seconds/60:.2f} min)")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated files:")
    print(f"  • Metrics Curves: {results_dir}/metrics_curves_{timestamp}.png")
    print(f"  • Epoch Data: {results_dir}/metrics_curves_{timestamp}_data.csv")
    print(f"  • ROC Curve: {results_dir}/roc_curve_{timestamp}.png")
    print(f"  • ROC Data: {results_dir}/roc_data_{timestamp}.csv")
    print(f"  • Confusion Matrix: {results_dir}/confusion_matrix_{timestamp}.png")
    print(f"  • Evaluation Results: {results_dir}/evaluation_results_{timestamp}.csv")
    print(f"  • Permutation Test: {results_dir}/permutation_test_{timestamp}.png")
    print(f"  • Mantel Test: {results_dir}/mantel_test_{timestamp}.png")
    print("="*70)
    
    return trained_model, vocab, label_encoder, results_df, metrics, total_time_seconds, roc_results

if __name__ == "__main__":
    import os
    os.chdir('/content/drive/MyDrive/BigData_11_2026')
    print(f"Working directory: {os.getcwd()}")
    
    try:
        try:
            import optuna
        except ImportError:
            print("Optuna not installed. Installing...")
            os.system("pip install optuna")
            import optuna
        
        model, vocab, label_encoder, results_df, metrics, total_time, roc_results = main()
        print(f"\nExecution time: {total_time:.2f}s")
        print(f"AUC (Validation): {roc_results['auc']:.4f}")
        print(f"Optimal threshold used for test: {roc_results['optimal_threshold']:.4f}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()