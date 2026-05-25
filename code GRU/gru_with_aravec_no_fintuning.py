"""
GRU avec plongements AraVec (GELÉS - Pas de fine-tuning) pour la classification de texte - Version finale
Support: Arabizi, Arabe algérien, Anglais, Français
Entrée: Fichier CSV kfoldsdata.csv (toutes les données pour la validation croisée) et test.csv
Sortie: Fichier CSV d'évaluation avec prédictions et métriques + Courbe ROC (UNIQUEMENT SUR TEST.CSV)
OPTIMISATION DES HYPERPARAMÈTRES AVEC OPTUNA - MAXIMISATION DU F1 DE VALIDATION
PLONGEMENTS NON ENTRAÎNABLES (Gelés - Pas de Fine-tuning)
EPOCHES FIXES = 10 (Pas d'arrêt précoce pour la recherche d'hyperparamètres)
PAS DE LISSAGE DES LABELS (Perte BCE standard)
REDUCELRONPLATEAU (Réduction adaptative du taux d'apprentissage) - MONITORE LA PERTE DE VALIDATION
TOUS LES MOTS INCLUS (Pas de filtrage par fréquence minimale)
SAUVEGARDE DE LA VECTORISATION
VALIDATION CROISÉE K-FOLDS (K=5) AVEC STRATIFIEDKFOLD
VOCABULAIRE CONSTRUIT UNIQUEMENT SUR LES DONNÉES D'ENTRAÎNEMENT (KFOLDSDATA.CSV)
SEUIL FIXE = 0.5
ÉPOQUE OPTIMALE = MIN VAL_LOSS (VALIDATION CROISÉE FINALE UNIQUEMENT)
AUC = VALEUR SUR TEST.CSV (UNIQUEMENT)
COURBE ROC SUR TEST.CSV (UNIQUEMENT)
PLONGEMENTS CONVERTIS EN TENSEUR PYTORCH UNE SEULE FOIS
SCHEDULER SUR LA PERTE de validation 
GRAINE PAR PLI (per-fold seed) POUR REPRODUCTIBILITÉ
NOMBRE D'ESSAIS OPTUNA = 15 (5 aléatoires + 10 TPE)
GRADIENT CLIPPING: HYPERPARAMÈTRE (log-uniform entre 0.5 et 5.0)
GRAPHES SÉPARÉS: 10 ÉPOQUES COMPLET + GRAPHE JUSQU'À L'ÉPOQUE OPTIMALE (SANS AUC)
ROC SUR TEST.CSV UNIQUEMENT (SANS SEUIL, AUC SEULEMENT)
GRAPHIQUE DE TEMPS: UN SEUL GRAPHE AVEC DEUX BARRES (ENTRAÎNEMENT FINAL + TOTAL)
GRU UNIDIRECTIONNEL (FORCÉ) - PAS DE BIDIRECTIONNEL
PLONGEMENTS ARAVEC (GELÉS - PAS DE FINE-TUNING)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_curve, roc_auc_score,
                              classification_report)
from sklearn.model_selection import StratifiedKFold
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
import json
import random

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ── Graine globale pour la reproductibilité ──
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé: {device}")

# ── Variables globales partagées avec Optuna ──
global_train_texts = None
global_train_labels = None
global_vocab = None
global_embedding_tensor = None

# ── Constantes ──
FIXED_EPOCHS = 10          # Nombre fixe d'époques pour la recherche
K_FOLDS = 5                # Nombre de plis pour la validation croisée
ARAVEC_DATA_DIR = '/content/drive/MyDrive/BigData_11_2026/data'


# ══════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════

def set_seed(seed):
    """Fixer toutes les graines aléatoires pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════
# GESTION DES PLONGEMENTS ARAVEC
# ══════════════════════════════════════════════════════

class AraVecEmbeddings:
    """Gestionnaire des plongements AraVec - GELÉS (pas de fine-tuning)."""

    def __init__(self, embedding_dim=300, data_dir=ARAVEC_DATA_DIR):
        self.embedding_dim = embedding_dim
        self.data_dir = data_dir
        self.word_vectors = None
        self.word_to_idx = {}
        self.idx_to_word = {}

    def load_model(self):
        """Charger le modèle AraVec pré-entraîné depuis le disque."""
        try:
            syn0_path = os.path.join(self.data_dir, 'Twt-CBOW.wv.syn0.npy')
            vocab_path = os.path.join(self.data_dir, 'Twt-CBOW')

            if not os.path.exists(syn0_path):
                print(f"Fichier non trouvé: {syn0_path}")
                return False

            print(f"Chargement des plongements AraVec depuis {self.data_dir}...")

            self.word_vectors = np.load(syn0_path)
            print(f"Matrice des vecteurs chargée: {self.word_vectors.shape}")

            vocab_loaded = False

            # Essayer différents formats de vocabulaire
            vocab_file = os.path.join(self.data_dir, 'Twt-CBOW.vocab')
            if os.path.exists(vocab_file):
                try:
                    with open(vocab_file, 'r', encoding='utf-8', errors='ignore') as f:
                        words = [line.strip().split()[0] for line in f if line.strip()]
                        self.idx_to_word = {i: word for i, word in enumerate(words)}
                        self.word_to_idx = {word: i for i, word in enumerate(words)}
                        vocab_loaded = True
                        print(f"Vocabulaire chargé depuis {vocab_file}")
                except Exception as e:
                    print(f"Erreur chargement vocabulaire: {e}")

            if not vocab_loaded and os.path.exists(vocab_path):
                try:
                    with open(vocab_path, 'rb') as f:
                        vocab_data = pickle.load(f)
                        if isinstance(vocab_data, dict):
                            self.word_to_idx = vocab_data
                            self.idx_to_word = {v: k for k, v in vocab_data.items()}
                        elif isinstance(vocab_data, list):
                            self.idx_to_word = {i: word for i, word in enumerate(vocab_data)}
                            self.word_to_idx = {word: i for i, word in enumerate(vocab_data)}
                        else:
                            raise ValueError("Format de vocabulaire non reconnu")
                        vocab_loaded = True
                        print("Vocabulaire chargé depuis fichier pickle")
                except Exception as e:
                    print(f"Erreur chargement pickle: {e}")

            if not vocab_loaded:
                try:
                    from gensim.models import KeyedVectors
                    if os.path.exists(vocab_path):
                        model = KeyedVectors.load(vocab_path)
                        self.word_to_idx = {word: idx for idx, word in enumerate(model.index_to_key)}
                        self.idx_to_word = {idx: word for idx, word in enumerate(model.index_to_key)}
                        self.word_vectors = model.vectors
                        vocab_loaded = True
                        print("Vocabulaire chargé via Gensim")
                except Exception as e:
                    print(f"Erreur chargement Gensim: {e}")

            if not vocab_loaded:
                print("Création d'un vocabulaire basé sur la taille des vecteurs")
                vocab_size = self.word_vectors.shape[0]
                self.idx_to_word = {i: f"word_{i}" for i in range(vocab_size)}
                self.word_to_idx = {f"word_{i}": i for i in range(vocab_size)}
                vocab_loaded = True

            # Alignement des dimensions
            if self.word_vectors.shape[0] != len(self.word_to_idx):
                print(f"Alignement: vocab={len(self.word_to_idx)}, vecteurs={self.word_vectors.shape[0]}")
                min_size = min(self.word_vectors.shape[0], len(self.word_to_idx))
                self.word_vectors = self.word_vectors[:min_size]
                if len(self.word_to_idx) > min_size:
                    words_list = list(self.word_to_idx.keys())[:min_size]
                    self.word_to_idx = {word: i for i, word in enumerate(words_list)}
                    self.idx_to_word = {i: word for i, word in enumerate(words_list)}

            print(f"Vocabulaire chargé: {len(self.word_to_idx)} mots")
            print(f"Dimension des plongements AraVec: {self.word_vectors.shape[1]}")
            print("Modèle AraVec chargé avec succès!")
            print("Support: Arabe, Arabizi et texte multilingue")
            return True

        except Exception as e:
            print(f"Erreur lors du chargement du modèle AraVec: {e}")
            print("Utilisation de plongements aléatoires à la place")
            self.word_vectors = None
            return False

    def get_word_vector(self, word):
        """Obtenir le vecteur d'un mot. Retourne un vecteur aléatoire si absent."""
        if self.word_vectors is not None and word in self.word_to_idx:
            idx = self.word_to_idx[word]
            if idx < len(self.word_vectors):
                return self.word_vectors[idx]
        return np.random.randn(self.embedding_dim) * 0.01

    def get_embedding_tensor(self, vocab):
        """
        Construire la matrice de plongement pour le vocabulaire entier
        et la retourner sous forme de tenseur PyTorch (CPU).
        """
        vocab_size = len(vocab)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))

        print(f"Construction de la matrice de plongement pour {vocab_size} mots...")
        words_processed = 0
        words_found = 0

        for word, idx in vocab.items():
            # <PAD> et <UNK> restent à zéro / aléatoire par défaut
            if word not in ['<PAD>', '<UNK>']:
                vector = self.get_word_vector(word)
                embedding_matrix[idx] = vector
                words_processed += 1
                if not np.all(vector == 0):
                    words_found += 1
                if words_processed % 5000 == 0:
                    print(f"Traitement: {words_processed}/{vocab_size} mots")

        # Le vecteur <PAD> est explicitement mis à zéro
        embedding_matrix[0] = np.zeros(self.embedding_dim)

        print(f"Matrice de plongement construite: {embedding_matrix.shape}")
        if words_processed > 0:
            print(f"Mots trouvés dans AraVec: {words_found}/{words_processed} "
                  f"({words_found / words_processed * 100:.1f}%)")
        print("NOTE: Les plongements seront GELÉS (pas de fine-tuning)")

        # Conversion en tenseur PyTorch une seule fois
        return torch.from_numpy(embedding_matrix).float()

    def save_metadata(self, save_dir):
        """Sauvegarder les métadonnées AraVec."""
        os.makedirs(save_dir, exist_ok=True)

        if self.word_vectors is not None:
            np.save(os.path.join(save_dir, 'aravec_word_vectors.npy'), self.word_vectors)
            print(f"Plongements sauvegardés dans {save_dir}/aravec_word_vectors.npy")

        with open(os.path.join(save_dir, 'aravec_word_to_idx.pkl'), 'wb') as f:
            pickle.dump(self.word_to_idx, f)
        with open(os.path.join(save_dir, 'aravec_idx_to_word.pkl'), 'wb') as f:
            pickle.dump(self.idx_to_word, f)

        metadata = {
            'dim_plongement': self.embedding_dim,
            'taille_vocabulaire': len(self.word_to_idx),
            'forme_vecteurs': self.word_vectors.shape if self.word_vectors is not None else None,
            'chemin_modele': self.data_dir
        }
        with open(os.path.join(save_dir, 'metadonnees_aravec.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Métadonnées AraVec sauvegardées dans {save_dir}/metadonnees_aravec.json")


# ══════════════════════════════════════════════════════
# DATASET PYTORCH
# ══════════════════════════════════════════════════════

class TextDataset(Dataset):
    """Dataset PyTorch pour la classification de texte."""

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

        # Conversion des mots en indices — <UNK> si mot hors vocabulaire
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
                   for word in text.split()]

        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
            seq_len = self.max_len
        else:
            seq_len = len(indices)
            # Rembourrage avec <PAD>
            indices += [self.word_to_idx['<PAD>']] * (self.max_len - len(indices))

        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(seq_len,  dtype=torch.long),
            torch.tensor(label,    dtype=torch.float)
        )


# ══════════════════════════════════════════════════════
# CONSTRUCTION DU VOCABULAIRE
# ══════════════════════════════════════════════════════

def build_vocabulary(texts):
    """
    Construire le vocabulaire à partir des textes d'entraînement.
    TOUS les mots sont inclus (pas de seuil de fréquence minimale).
    """
    print("Construction du vocabulaire...")

    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # Index 0 = <PAD>, index 1 = <UNK>
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in word_counts:
        vocab[word] = len(vocab)

    print(f"Taille du vocabulaire: {len(vocab)} (TOUS les mots inclus)")
    print(f"Mots uniques trouvés: {len(word_counts)}")
    return vocab


# ══════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════

def load_data(
        kfold_path='/content/drive/MyDrive/BigData_11_2026/data/kfoldsdata.csv',
        test_path='/content/drive/MyDrive/BigData_11_2026/data/test.csv',
        text_col='text',
        label_col='label'):
    """
    Charger les données K-Fold depuis kfoldsdata.csv
    et les données de test depuis test.csv.
    """
    print(f"\nChargement des données K-Folds depuis: {kfold_path}")
    if not os.path.exists(kfold_path):
        raise FileNotFoundError(f"Fichier non trouvé: {kfold_path}")

    df_kfold = pd.read_csv(kfold_path, sep=';')
    print(f"Chargé: {len(df_kfold)} échantillons pour la validation croisée")

    for col in [text_col, label_col]:
        if col not in df_kfold.columns:
            raise ValueError(f"Colonne '{col}' non trouvée. Disponibles: {df_kfold.columns.tolist()}")

    X_kfold = df_kfold[text_col].astype(str).tolist()
    y_kfold = df_kfold[label_col].tolist()

    print("\nDistribution des classes dans kfoldsdata.csv:")
    for label, count in pd.Series(y_kfold).value_counts().items():
        print(f"  {label}: {count} ({count / len(y_kfold) * 100:.1f}%)")

    # ── Chargement du jeu de test (optionnel) ──
    X_test, y_test = [], []
    if os.path.exists(test_path):
        print(f"\nChargement des données de test depuis: {test_path}")
        df_test = pd.read_csv(test_path, sep=';')
        print(f"Chargé: {len(df_test)} échantillons de test")
        for col in [text_col, label_col]:
            if col not in df_test.columns:
                raise ValueError(f"Colonne '{col}' non trouvée dans le fichier test")
        X_test = df_test[text_col].astype(str).tolist()
        y_test = df_test[label_col].tolist()
        print("\nDistribution des classes - Test:")
        for label, count in pd.Series(y_test).value_counts().items():
            print(f"  {label}: {count} ({count / len(y_test) * 100:.1f}%)")
    else:
        print(f"\nAttention: test.csv non trouvé à {test_path}. Évaluation finale ignorée.")

    # ── Encodage des labels (fit sur kfold + test pour couvrir toutes les classes) ──
    label_encoder = LabelEncoder()
    label_encoder.fit(y_kfold + y_test)

    y_kfold_enc = label_encoder.transform(y_kfold)
    y_test_enc  = label_encoder.transform(y_test) if y_test else []

    print(f"\nDonnées chargées avec succès!")
    print(f"  K-Folds: {len(X_kfold)} échantillons")
    print(f"  Test:    {len(X_test)} échantillons")
    print(f"  Classes: {label_encoder.classes_}")

    return X_kfold, y_kfold_enc, X_test, y_test_enc, label_encoder


# ══════════════════════════════════════════════════════
# ARCHITECTURE GRU
# ══════════════════════════════════════════════════════

class GRUClassifier(nn.Module):
    """
    Classifieur GRU unidirectionnel avec plongements AraVec GELÉS (pas de fine-tuning).
    Le bidirectionnel est désactivé de force.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2,
                 dropout_pre_gru=0.3,
                 dropout_gru=0.5,
                 dropout_post_gru=0.3,
                 pretrained_embeddings=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = False  # FORCÉ UNIDIRECTIONNEL

        # ── Couche d'embedding ──
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_pre_gru = nn.Dropout(dropout_pre_gru)

        # Chargement des plongements pré-entraînés si fournis
        if pretrained_embeddings is not None:
            print("Chargement des plongements AraVec pré-entraînés...")
            assert pretrained_embeddings.shape[0] == vocab_size
            assert pretrained_embeddings.shape[1] == embedding_dim
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # GELÉ (pas de fine-tuning)
            print("Plongements AraVec GELÉS (fine-tuning désactivé)")

        # ── GRU ──
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout_gru if n_layers > 1 else 0,
            batch_first=True
        )

        self.dropout_post_gru = nn.Dropout(dropout_post_gru)
        self.fc      = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # Plongements + dropout
        embedded = self.dropout_pre_gru(self.embedding(text))

        # Séquences compressées pour gérer le rembourrage
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)

        # Dernière couche cachée de la dernière couche GRU
        out = self.dropout_post_gru(hidden[-1, :, :])
        out = self.sigmoid(self.fc(out))
        return out.squeeze()


# ══════════════════════════════════════════════════════
# BOUCLES D'ENTRAÎNEMENT ET D'ÉVALUATION
# ══════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, clip_norm):
    """
    Entraîner une époque complète.
    Retourne: perte_moy, exactitude, précision, rappel, f1
    """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for texts, lengths, labels in loader:
        texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels.float())
        loss.backward()

        # Gradient clipping avec la norme optimisée par Optuna
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall    = recall_score(all_labels, all_preds,    average='binary', zero_division=0)
    f1        = f1_score(all_labels, all_preds,        average='binary', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1


def evaluate(model, loader, criterion, threshold=0.5):
    """
    Évaluer le modèle sur un jeu de données.
    Retourne: perte_moy, exactitude, précision, rappel, f1, auc, prédictions, vrais_labels, probabilités
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for texts, lengths, labels in loader:
            texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
            outputs = model(texts, lengths)

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            probs = outputs.cpu().numpy()
            preds = (outputs > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, precision, recall, f1, auc, all_preds, all_labels, all_probs


def train_with_history(model, train_loader, val_loader, epochs, lr, weight_decay,
                       factor, patience, clip_norm):
    """
    Entraîner le modèle et enregistrer l'historique complet par époque.
    Le scheduler ReduceLROnPlateau surveille la PERTE val.
    Retourne: meilleure_perte_val, historique, probabilités_val_par_époque
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler sur la perte validation 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor = factor,
        patience = patience,
        min_lr=1e-7
    )

    
    history = []
    val_probs_per_epoch = []
    reduction_epochs_fold = []
    prev_lr = lr

    for epoch in range(epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, clip_norm
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, _, val_labels, val_probs = evaluate(
            model, val_loader, criterion, threshold=0.5
        )
        scheduler.step(val_loss)
        # Mise à jour du scheduler sur la perte validation

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            reduction_epochs_fold.append(epoch)
        prev_lr = current_lr

        history.append({
            'epoch':          epoch + 1,
            'train_loss':     train_loss,
            'train_accuracy': train_acc,
            'train_precision':train_prec,
            'train_recall':   train_rec,
            'train_f1':       train_f1,
            'val_loss':       val_loss,
            'val_accuracy':   val_acc,
            'val_precision':  val_prec,
            'val_recall':     val_rec,
            'val_f1':         val_f1,
            'val_auc':        val_auc
        })

        val_probs_per_epoch.append(val_probs)

    return history, val_probs_per_epoch, reduction_epochs_fold


# ══════════════════════════════════════════════════════
# FONCTION OBJECTIF OPTUNA — MAXIMISATION DU F1 DE VALIDATION
# ══════════════════════════════════════════════════════

def optuna_objective(trial):
    """
    Fonction objectif Optuna: maximise le F1 moyen de validation
    sur K_FOLDS plis avec FIXED_EPOCHS époques chacun.
    """
    global global_train_texts, global_train_labels, global_vocab, global_embedding_tensor

    # ── Espace de recherche des hyperparamètres ──
    hidden_dim         = trial.suggest_int('hidden_dim',          32,   128,  step=16)
    n_layers           = trial.suggest_int('n_layers',             1,     2)
    dropout_pre_gru    = trial.suggest_float('dropout_pre_gru',   0.2,   0.6, step=0.1)
    dropout_gru        = trial.suggest_float('dropout_gru',        0.3,   0.7, step=0.1)
    dropout_post_gru   = trial.suggest_float('dropout_post_gru',   0.2,   0.6, step=0.1)
    learning_rate      = trial.suggest_float('learning_rate',      5e-5,  5e-3, log=True)
    weight_decay       = trial.suggest_float('weight_decay',       1e-4,  1e-2, log=True)
    batch_size         = trial.suggest_categorical('batch_size', [32, 64, 128])
    factor   = trial.suggest_float('factor', 0.1, 0.9, step=0.05)
    patience = trial.suggest_int('patience', 0, 4)
    clip_norm          = trial.suggest_float('clip_norm',          0.5,   5.0, log=True)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    f1_per_fold_per_epoch = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(global_train_texts, global_train_labels)):

        set_seed(SEED + fold)
        print(f"  Pli {fold + 1} - graine: {SEED + fold}")

        train_texts = [global_train_texts[i] for i in train_idx]
        train_labels= [global_train_labels[i] for i in train_idx]
        val_texts   = [global_train_texts[i] for i in val_idx]
        val_labels  = [global_train_labels[i] for i in val_idx]

        train_loader = DataLoader(
            TextDataset(train_texts, train_labels, global_vocab, max_len=100),
            batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            TextDataset(val_texts, val_labels, global_vocab, max_len=100),
            batch_size=batch_size, shuffle=False)

        model = GRUClassifier(
            vocab_size=len(global_vocab),
            embedding_dim=300,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers,
            dropout_pre_gru=dropout_pre_gru,
            dropout_gru=dropout_gru,
            dropout_post_gru=dropout_post_gru,
            pretrained_embeddings=global_embedding_tensor
        ).to(device)

        history, _, _ = train_with_history(
            model, train_loader, val_loader, FIXED_EPOCHS,        
            learning_rate, weight_decay,
            factor, patience, clip_norm
        )

        f1_per_fold_per_epoch.append([e['val_f1'] for e in history])

    # F1 moyen sur les plis pour chaque époque → on retourne le maximum
    mean_f1_per_epoch = [
        np.mean([f1_per_fold_per_epoch[p][e] for p in range(K_FOLDS)])
        for e in range(FIXED_EPOCHS)
    ]
    best_mean_f1 = max(mean_f1_per_epoch)

    print(f"  -> Validation croisée ({K_FOLDS} plis):")
    print(f"     F1 moyen par époque: {mean_f1_per_epoch}")
    print(f"     F1 maximum (moyenne plis): {best_mean_f1:.6f}")

    return best_mean_f1


# ══════════════════════════════════════════════════════
# K-FOLD AVEC LES MEILLEURS HYPERPARAMÈTRES (PAS DE ROC)
# ══════════════════════════════════════════════════════

def run_kfold_best_params(fold_histories, results_dir, timestamp):
    """
    Agrège les fold_histories pour calculer les métriques, trouver l'époque optimale,
    et générer les graphiques.
    
    Args:
        fold_histories: Liste des historiques d'entraînement pour chaque fold
                       (déjà calculés par la phase 2)
        results_dir: Répertoire pour sauvegarder les résultats
        timestamp: Horodatage pour les noms de fichiers
    
    Returns:
        df_agg: DataFrame avec métriques agrégées par époque
        mean_auc: AUC moyenne à l'époque optimale
        optimal_epoch: Époque avec la plus faible perte de validation moyenne
    """
    print("\n" + "=" * 70)
    print("AGRÉGATION DES MÉTRIQUES K-FOLD")
    print("=" * 70)
    print(f"Nombre de plis: {len(fold_histories)}")
    print(f"Nombre d'époques par pli: {len(fold_histories[0]) if fold_histories else 0}")
    
    if not fold_histories:
        raise ValueError("fold_histories est vide")
    
    K_FOLDS = len(fold_histories)
    FIXED_EPOCHS = len(fold_histories[0])
    
    # ── Agrégation des métriques par époque ──
    metrics_names = [
        'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1',
        'val_loss',   'val_accuracy',   'val_precision',   'val_recall',   'val_f1', 'val_auc'
    ]
    
    agg = {'epoque': list(range(1, FIXED_EPOCHS + 1))}
    
    for metric in metrics_names:
        epoch_vals = [
            [fold_histories[p][e][metric] for p in range(K_FOLDS)]
            for e in range(FIXED_EPOCHS)
        ]
        agg[f'{metric}_moy'] = [np.mean(v) for v in epoch_vals]
        agg[f'{metric}_std'] = [np.std(v)  for v in epoch_vals]
    
    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(f"{results_dir}/kfold_metriques_agregees_{timestamp}.csv", index=False)
    print(f"Métriques agrégées sauvegardées: {results_dir}/kfold_metriques_agregees_{timestamp}.csv")
    
    # ── Époque optimale = min val_loss moyenne ──
    optimal_epoch = int(np.argmin(df_agg['val_loss_moy'])) + 1
    print(f"\nÉpoque optimale (min perte_val moyenne): {optimal_epoch} "
          f"(perte_val = {df_agg['val_loss_moy'][optimal_epoch - 1]:.6f})")
    
    # ── AUC moyenne à l'époque optimale ──
    auc_at_optimal = [fold_histories[p][optimal_epoch - 1]['val_auc'] for p in range(K_FOLDS)]
    mean_auc = np.mean(auc_at_optimal)
    std_auc = np.std(auc_at_optimal)
    print(f"AUC validation à l'époque {optimal_epoch}: {mean_auc:.4f} ± {std_auc:.4f}")
    
    # ── Graphiques ──
    _plot_kfold_metrics(df_agg, df_agg, K_FOLDS,
                        save_path=f"{results_dir}/kfold_metriques_complet_{timestamp}.png")
    
    _plot_kfold_metrics(df_agg, df_agg.iloc[:optimal_epoch], K_FOLDS,
                        save_path=f"{results_dir}/kfold_metriques_optimal_{timestamp}.png")
    
    return df_agg, mean_auc, optimal_epoch



def run_kfold_find_reduction_epochs(all_texts, all_labels, vocab, embedding_tensor,
                                    best_params, best_factor, best_patience, results_dir, timestamp):
    print("\n" + "=" * 70)
    print("PHASE 2 - K-FOLD: RECHERCHE DES ÉPOQUES DE RÉDUCTION")
    print(f"   best_factor = {best_factor}, best_patience = {best_patience}")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    all_fold_val_losses = []
    all_fold_histories = []
    all_fold_reduction_epochs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
        set_seed(SEED + fold_idx)
        print(f"\n--- Pli {fold_idx+1}/{K_FOLDS} ---")

        tr_texts  = [all_texts[i]  for i in train_idx]
        tr_labels = [all_labels[i] for i in train_idx]
        vl_texts  = [all_texts[i]  for i in val_idx]
        vl_labels = [all_labels[i] for i in val_idx]

        train_loader = DataLoader(
            TextDataset(tr_texts, tr_labels, vocab, max_len=100),
            batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(
            TextDataset(vl_texts, vl_labels, vocab, max_len=100),
            batch_size=best_params['batch_size'], shuffle=False)

        model = GRUClassifier(
            vocab_size=len(vocab), embedding_dim=300,
            hidden_dim=best_params['hidden_dim'], output_dim=1,
            n_layers=best_params['n_layers'],
            dropout_pre_gru=best_params['dropout_pre_gru'],
            dropout_gru=best_params['dropout_gru'],
            dropout_post_gru=best_params['dropout_post_gru'],
            pretrained_embeddings=embedding_tensor
        ).to(device)

        history, _, reduction_epochs_fold = train_with_history(
            model, train_loader, val_loader, FIXED_EPOCHS,
            best_params['learning_rate'], best_params['weight_decay'],
            best_factor, best_patience, best_params['clip_norm']
        )
        all_fold_val_losses.append([e['val_loss'] for e in history])
        all_fold_histories.append(history)
        all_fold_reduction_epochs.append(reduction_epochs_fold)

    mean_val_losses = np.mean(all_fold_val_losses, axis=0)
    std_val_losses  = np.std(all_fold_val_losses,  axis=0)

    all_reductions = [e for sublist in all_fold_reduction_epochs for e in sublist]
    if all_reductions:
        reduction_counter = Counter(all_reductions)
        reduction_epochs = sorted([e for e, count in reduction_counter.items()
                                   if count > K_FOLDS / 2])
        print(f"\nÉpoques de réduction (majorité): {[e+1 for e in reduction_epochs]}")
    else:
        reduction_epochs = []
        print("\nAucune réduction détectée")

    with open(f"{results_dir}/reduction_epochs_{timestamp}.json", 'w') as f:
        json.dump({'best_factor': best_factor,
                   'reduction_epochs': reduction_epochs,
                   'mean_val_losses': mean_val_losses.tolist()}, f, indent=2)

    return reduction_epochs, mean_val_losses, std_val_losses, all_fold_histories


def _plot_kfold_metrics(df_full, df_plot, k_folds, save_path):
    """Tracer les métriques K-Fold."""
    epochs = df_plot['epoque']
    pairs = [
        ('train_loss_moy',      'val_loss_moy',      'Perte',      'Perte'),
        ('train_accuracy_moy',  'val_accuracy_moy',  'Exactitude', 'Exactitude'),
        ('train_precision_moy', 'val_precision_moy', 'Précision',  'Précision'),
        ('train_recall_moy',    'val_recall_moy',    'Rappel',     'Rappel'),
        ('train_f1_moy',        'val_f1_moy',        'F1-score',   'F1-score'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (train_col, val_col, ylabel, title_label) in enumerate(pairs):
        axes[i].plot(epochs, df_plot[train_col], 'b-o',
                     label='Entraînement', linewidth=2, markersize=6)
        axes[i].plot(epochs, df_plot[val_col],   'r-s',
                     label='Validation',   linewidth=2.5, markersize=8)
        axes[i].set_xlabel('Époque', fontsize=12)
        axes[i].set_ylabel(ylabel,   fontsize=12)
        axes[i].set_title(f'{title_label} (moy. sur {k_folds} plis)',
                          fontsize=11, fontweight='bold')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    axes[5].set_visible(False)
    plt.suptitle('Métriques validation croisée pour GRU avec AraVec',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé: {save_path}")


# ══════════════════════════════════════════════════════
# FONCTION POUR TRACER LA ROC SUR TEST.CSV
# ══════════════════════════════════════════════════════

def plot_roc_test(true_labels, probabilities, save_path):
    """
    Tracer la courbe ROC pour l'ensemble de test (test.csv).
    """
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc_score = roc_auc_score(true_labels, probabilities)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Courbe ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7,
             label='Classifieur aléatoire (AUC = 0.5)')
    plt.xlabel('Taux de faux positifs', fontsize=12)
    plt.ylabel('Taux de vrais positifs', fontsize=12)
    plt.title(f'Courbe ROC - Ensemble de Test (test.csv)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    print(f"AUC (test.csv): {auc_score:.4f}")
    
    # Sauvegarder les données ROC
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(save_path.replace('.png', '_donnees.csv'), index=False)
    
    return auc_score


# ══════════════════════════════════════════════════════
# ENTRAÎNEMENT FINAL
# ══════════════════════════════════════════════════════

def train_final_model(train_texts, train_labels, vocab, embedding_tensor,
                      best_params, reduction_epochs, best_factor , n_epochs):
    """
    Entraîner le modèle final sur TOUTES les données (kfoldsdata.csv)
    pendant n_epochs époques (déterminé par le K-Fold via min val_loss).
    """
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE FINAL SUR TOUTES LES DONNÉES (KFOLDSDATA.CSV)")
    print(f"NOMBRE FIXE D'ÉPOQUES: {n_epochs} (issu de la validation croisée - min perte_val)")
    print("=" * 70)
    print(f"Entraînement sur {len(train_texts)} échantillons")

    set_seed(SEED)

    train_loader = DataLoader(
        TextDataset(train_texts, train_labels, vocab, max_len=100),
        batch_size=best_params['batch_size'], shuffle=True
    )

    model = GRUClassifier(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=best_params['hidden_dim'],
        output_dim=1,
        n_layers=best_params['n_layers'],
        dropout_pre_gru=best_params['dropout_pre_gru'],
        dropout_gru=best_params['dropout_gru'],
        dropout_post_gru=best_params['dropout_post_gru'],
        pretrained_embeddings=embedding_tensor
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux:      {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,} (embeddings exclus)")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=best_params['learning_rate'],
                           weight_decay=best_params['weight_decay'])
    
    lr_reducer = ManualLRReducer(optimizer, reduction_epochs, best_factor)

    history = []
    print("\n" + "=" * 80)
    print(f"ENTRAÎNEMENT FINAL PENDANT {n_epochs} ÉPOQUES")
    print("=" * 80 + "\n")

    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, best_params['clip_norm']
        )
        lr_reducer.step(epoch)

        history.append({
            'epoch':          epoch + 1,
            'train_loss':     train_loss,
            'train_accuracy': train_acc,
            'train_precision':train_prec,
            'train_recall':   train_rec,
            'train_f1':       train_f1,
        })
        print(f"ÉPOQUE {epoch + 1}/{n_epochs} - "
              f"Perte: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"Prec: {train_prec:.4f} | Rappel: {train_rec:.4f} | F1: {train_f1:.4f}")

    training_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ENTRAÎNEMENT FINAL TERMINÉ APRÈS {n_epochs} ÉPOQUES")
    print(f"Temps d'entraînement: {training_time:.2f}s ({training_time / 60:.2f} min)")
    print("=" * 80)

    print("\nVérification des plongements:")
    for name, param in model.named_parameters():
        if 'embedding' in name:
            print(f"  {name}: requires_grad = {param.requires_grad}")

    return model, history, training_time


# ══════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════

def plot_final_metrics(history, save_path):
    """Tracer les métriques d'entraînement du modèle final."""
    if not history:
        print("Attention: Aucune donnée d'époque à tracer!")
        return None

    epochs     = [e['epoch']           for e in history]
    train_loss = [e['train_loss']      for e in history]
    train_acc  = [e['train_accuracy']  for e in history]
    train_f1   = [e['train_f1']        for e in history]
    train_prec = [e['train_precision'] for e in history]
    train_rec  = [e['train_recall']    for e in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    series = [
        (train_loss, 'Perte',      'Perte (entraînement final)'),
        (train_acc,  'Exactitude', 'Exactitude (entraînement final)'),
        (train_f1,   'F1-score',   'F1-score (entraînement final)'),
        (train_prec, 'Précision',  'Précision (entraînement final)'),
        (train_rec,  'Rappel',     'Rappel (entraînement final)'),
    ]
    for i, (data, ylabel, title) in enumerate(series):
        axes[i].plot(epochs, data, 'b-o', linewidth=2, markersize=6,
                     label='Entraînement', alpha=0.8)
        axes[i].set_xlabel('Époque', fontsize=12)
        axes[i].set_ylabel(ylabel,   fontsize=12)
        axes[i].set_title(title,     fontsize=13, fontweight='bold')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
        axes[i].set_xticks(epochs)

    axes[5].set_visible(False)
    plt.suptitle('Métriques d\'entraînement final - GRU avec AraVec gelés',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nGraphique des métriques sauvegardé: {save_path}")
    df_metrics = pd.DataFrame(history)
    csv_path = save_path.replace('.png', '_donnees.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f"Données des époques sauvegardées: {csv_path}")
    return df_metrics


def plot_time_chart(training_time, total_time, optimal_epochs, save_path):
    """Graphique à deux barres: temps d'entraînement final et temps total."""
    labels = [f"Entraînement final\n({optimal_epochs} époques)", "Temps total exécution"]
    times  = [training_time, total_time]
    colors = ['#2E7D32', '#1565C0']

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, times, width=0.4, color=colors, edgecolor='black')

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{t:.1f}s\n({t / 60:.2f} min)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Secondes', fontsize=12, fontweight='bold')
    ax.set_title('Temps d\'exécution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique de temps sauvegardé: {save_path}")


# ══════════════════════════════════════════════════════
# ÉVALUATION SUR LE JEU DE TEST
# ══════════════════════════════════════════════════════

def save_evaluation_results(test_texts, true_labels, predictions, probabilities,
                             label_encoder, output_path):
    """Sauvegarder les prédictions complètes sur le jeu de test."""
    df = pd.DataFrame({
        'texte':                test_texts,
        'vrai_label':           label_encoder.inverse_transform(np.array(true_labels).astype(int)),
        'label_predit':         label_encoder.inverse_transform(np.array(predictions).astype(int)),
        'est_correct':          (np.array(true_labels).astype(int) == np.array(predictions).astype(int)),
        'probabilite_positive': np.array(probabilities),
        'probabilite_negative': 1 - np.array(probabilities)
    })
    df.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés: {output_path}")
    return df


def compute_metrics(true_labels, predictions):
    """Calculer exactitude, précision, rappel et F1."""
    true = np.array(true_labels).astype(int)
    pred = np.array(predictions).astype(int)
    return {
        'exactitude': accuracy_score(true, pred),
        'precision':  precision_score(true, pred, average='binary', zero_division=0),
        'rappel':     recall_score(true, pred,    average='binary', zero_division=0),
        'f1_score':   f1_score(true, pred,        average='binary', zero_division=0)
    }


def save_confusion_matrix(true_labels, predictions, label_encoder, output_path):
    """Sauvegarder la matrice de confusion en CSV."""
    cm = confusion_matrix(true_labels, predictions)
    class_names = label_encoder.classes_
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(output_path)
    print(f"Matrice de confusion sauvegardée: {output_path}")
    print("\n" + "=" * 70)
    print("MATRICE DE CONFUSION")
    print("=" * 70)
    print(df_cm)
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"\nVN: {tn}, FP: {fp}, FN: {fn}, VP: {tp}")
    return df_cm


def plot_confusion_matrix(true_labels, predictions, label_encoder, save_path):
    """Tracer la matrice de confusion."""
    cm = confusion_matrix(true_labels, predictions)
    class_names = label_encoder.classes_

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Réel',       fontsize=12, fontweight='bold')
    ax.set_title('Matrice de confusion - GRU avec AraVec gelés',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique matrice de confusion: {save_path}")


def save_metrics_report(metrics, output_path):
    """Sauvegarder le rapport de métriques."""
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    print(f"Rapport de métriques sauvegardé: {output_path}")
    print("\n" + "=" * 70)
    print("MÉTRIQUES D'ÉVALUATION")
    print("=" * 70)
    print(f"Exactitude:  {metrics['exactitude']:.4f}")
    print(f"Précision:   {metrics['precision']:.4f}")
    print(f"Rappel:      {metrics['rappel']:.4f}")
    print(f"F1-score:    {metrics['f1_score']:.4f}")


def save_misclassifications(test_texts, true_labels, predictions, probabilities,
                             label_encoder, output_dir, timestamp):
    """Sauvegarder les faux positifs et faux négatifs."""
    true  = np.array(true_labels).astype(int)
    pred  = np.array(predictions).astype(int)
    probs = np.array(probabilities)

    df_full = pd.DataFrame({
        'texte':                    test_texts,
        'vrai_label':               label_encoder.inverse_transform(true),
        'vrai_label_numerique':     true,
        'label_predit':             label_encoder.inverse_transform(pred),
        'label_predit_numerique':   pred,
        'probabilite_positive':     probs,
        'probabilite_negative':     1 - probs,
        'est_correct':              (true == pred)
    })

    fp = df_full[(df_full['vrai_label_numerique'] == 0) & (df_full['label_predit_numerique'] == 1)]
    fn = df_full[(df_full['vrai_label_numerique'] == 1) & (df_full['label_predit_numerique'] == 0)]

    if len(fp) > 0:
        fp_path = os.path.join(output_dir, f'faux_positifs_{timestamp}.csv')
        fp.to_csv(fp_path, index=False, encoding='utf-8-sig')
        print(f"Faux positifs: {len(fp)} → {fp_path}")
    if len(fn) > 0:
        fn_path = os.path.join(output_dir, f'faux_negatifs_{timestamp}.csv')
        fn.to_csv(fn_path, index=False, encoding='utf-8-sig')
        print(f"Faux négatifs: {len(fn)} → {fn_path}")

    return fp, fn


# ══════════════════════════════════════════════════════
# TEST DE PERMUTATION
# ══════════════════════════════════════════════════════

def permutation_test(model, test_loader, criterion, n_permutations=999, threshold=0.5):
    """Test de significativité statistique par randomisation."""
    print("\n" + "=" * 70)
    print("TEST DE SIGNIFICATIVITÉ STATISTIQUE (RANDOMISATION)")
    print("=" * 70)

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for texts, lengths, labels in test_loader:
            texts, lengths = texts.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    obs_preds  = (all_probs >= threshold).astype(int)
    obs_acc    = accuracy_score(all_labels, obs_preds)
    print(f"Exactitude observée: {obs_acc:.4f}")

    null_dist = []
    for i in range(n_permutations):
        shuffled = np.random.permutation(all_labels)
        null_dist.append(accuracy_score(shuffled, obs_preds))
        if (i + 1) % 100 == 0:
            print(f"  Permutations: {i + 1}/{n_permutations}")

    null_dist = np.array(null_dist)
    p_value   = np.mean(null_dist >= obs_acc)

    if   p_value < 0.001: significance = "*** (hautement significatif)"
    elif p_value < 0.01:  significance = "** (significatif)"
    elif p_value < 0.05:  significance = "* (significatif)"
    else:                 significance = "n.s. (non significatif)"

    print(f"\np-valeur: {p_value:.4f} - {significance}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(null_dist, bins=30, alpha=0.7, color='gray', edgecolor='black',
            label='Distribution nulle')
    ax.axvline(obs_acc, color='red', linewidth=2,
               label=f'Observée (p={p_value:.4f})')
    ax.axvline(null_dist.mean(), color='blue', linestyle='--',
               label=f'Moyenne = {null_dist.mean():.3f}')
    ax.set_xlabel('Exactitude', fontsize=12)
    ax.set_ylabel('Fréquence',  fontsize=12)
    ax.set_title('Test de randomisation - Distribution nulle',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return {
        'exactitude_observee': obs_acc,
        'p_valeur':            p_value,
        'signification':       significance,
        'moy_nulle':           null_dist.mean(),
        'std_nulle':           null_dist.std(),
        'figure':              fig
    }


# ══════════════════════════════════════════════════════
# SAUVEGARDE DE LA VECTORISATION
# ══════════════════════════════════════════════════════

def save_vectorization(vocab, embedding_tensor, label_encoder, aravec_obj,
                        save_dir='DL Models/GRU with AraVec/vectorisation'):
    """Sauvegarder tous les composants de vectorisation."""
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'vocabulaire.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulaire sauvegardé: {save_dir}/vocabulaire.pkl")

    np.save(os.path.join(save_dir, 'matrice_plongements.npy'), embedding_tensor.numpy())
    print(f"Matrice de plongement sauvegardée: {save_dir}/matrice_plongements.npy")

    with open(os.path.join(save_dir, 'encodeur_labels.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Encodeur de labels sauvegardé: {save_dir}/encodeur_labels.pkl")

    if aravec_obj:
        aravec_obj.save_metadata(os.path.join(save_dir, 'aravec_sauvegarde'))

    info = {
        'taille_vocab':             len(vocab),
        'dim_plongement':           embedding_tensor.shape[1],
        'nb_classes':               len(label_encoder.classes_),
        'classes':                  label_encoder.classes_.tolist(),
        'plongements_entrainables': False,
        'date_sauvegarde':          datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(save_dir, 'info_vectorisation.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)

    print(f"Informations de vectorisation sauvegardées: {save_dir}/info_vectorisation.json")

def save_classification_report(true_labels, predictions, probabilities, label_encoder, output_dir, timestamp):
    from sklearn.metrics import classification_report
    
    true  = np.array(true_labels).astype(int)
    pred  = np.array(predictions).astype(int)
    probs = np.array(probabilities)
    
    target_names = [str(cls) for cls in label_encoder.classes_]
    
    per_class_acc = {}
    for cls_idx in range(len(target_names)):
        mask = (true == cls_idx)
        per_class_acc[cls_idx] = accuracy_score(true[mask], pred[mask]) if mask.sum() > 0 else 0.0

    per_class_loss = {}
    criterion = nn.BCELoss(reduction='none')
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    true_tensor  = torch.tensor(true,  dtype=torch.float32)
    losses = criterion(probs_tensor, true_tensor).numpy()
    for cls_idx in range(len(target_names)):
        mask = (true == cls_idx)
        per_class_loss[cls_idx] = float(losses[mask].mean()) if mask.sum() > 0 else 0.0

    per_class_auc = {}
    for cls_idx in range(len(target_names)):
        try:
            scores = probs if cls_idx == 1 else 1 - probs
            per_class_auc[cls_idx] = roc_auc_score((true == cls_idx).astype(int), scores)
        except Exception:
            per_class_auc[cls_idx] = 0.0

    report_dict = classification_report(true, pred, target_names=target_names, output_dict=True)
    
    rows = []
    for cls_idx, cls_name in enumerate(target_names):
        r = report_dict[cls_name]
        rows.append({
            'classe':      cls_name,
            'précision':   r['precision'],
            'rappel':      r['recall'],
            'f1_score':    r['f1-score'],
            'exactitude':  per_class_acc[cls_idx],
            'perte':       per_class_loss[cls_idx],
            'auc':         per_class_auc[cls_idx],
            'support':     int(r['support'])
        })
    
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT PAR CLASSE:")
    print("=" * 70)
    print(df.to_string(index=False))
    
    report_path = os.path.join(output_dir, f'classification_report_{timestamp}.csv')
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\nClassification report sauvegardé : {report_path}")
    
    return df



# ============================================================
# MANUAL LR REDUCER
# ============================================================


class ManualLRReducer:
    def __init__(self, optimizer, reduction_epochs, factor):
        self.optimizer = optimizer
        self.reduction_epochs = set(reduction_epochs)
        self.factor = factor

    def step(self, epoch):
        if epoch in self.reduction_epochs:
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(old_lr * self.factor, 1e-7)
            for pg in self.optimizer.param_groups:
                pg['lr'] = new_lr
            print(f"    LR réduit epoch {epoch+1}: {old_lr:.2e} → {new_lr:.2e}")
            return True
        return False

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

# ══════════════════════════════════════════════════════
# OPTIMISATION OPTUNA
# ══════════════════════════════════════════════════════

def run_optuna(train_texts, train_labels, vocab, embedding_tensor,
               n_trials=15, n_startup_trials=5):
    """Lancer l'optimisation des hyperparamètres avec Optuna."""
    global global_train_texts, global_train_labels, global_vocab, global_embedding_tensor
    global_train_texts      = train_texts
    global_train_labels     = train_labels
    global_vocab            = vocab
    global_embedding_tensor = embedding_tensor

    print("\n" + "=" * 70)
    print("OPTIMISATION DES HYPERPARAMÈTRES AVEC OPTUNA")
    print("=" * 70)
    print(f"Exécution de {n_trials} essais ({n_startup_trials} aléatoires, puis TPE)...")
    print("OBJECTIF OPTUNA: MAXIMISATION DU F1 DE VALIDATION")
    print(f"VALIDATION CROISÉE STRATIFIÉE K-FOLDS K={K_FOLDS}")
    print("GRU UNIDIRECTIONNEL - PLONGEMENTS GELÉS")
    print(f"{FIXED_EPOCHS} ÉPOQUES FIXES")
    print("=" * 70)

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials, seed=SEED)
    study = optuna.create_study(
        direction='maximize',
        study_name='gru_aravec_f1_maximisation',
        sampler=sampler
    )
    study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    print("\n" + "=" * 70)
    print("RÉSULTATS DE L'OPTIMISATION OPTUNA")
    print("=" * 70)
    print(f"Meilleur F1 de validation (moy. {K_FOLDS} plis): {best_trial.value:.4f}")
    print("\nMeilleurs hyperparamètres trouvés:")

    best_factor   = best_trial.params['factor']
    best_patience = best_trial.params['patience']
    best_params   = {k: v for k, v in best_trial.params.items()
                 if k not in ('factor', 'patience')}
    for key, val in best_params.items():
        print(f"  {key}: {val}")
        
    best_params['epochs'] = FIXED_EPOCHS

    results_dir = 'DL Models/GRU with AraVec/Résultats'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    study.trials_dataframe().to_csv(f"{results_dir}/optuna_essais_{timestamp}.csv", index=False)

    # Graphiques Optuna
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    completed_values = [t.value for t in study.trials if t.state == TrialState.COMPLETE]
    axes[0].plot(completed_values, 'b-', linewidth=1)
    axes[0].scatter(range(len(completed_values)), completed_values, c='blue', s=30, alpha=0.6)
    axes[0].axhline(y=best_trial.value, color='r', linestyle='--',
                    label=f'Meilleur F1: {best_trial.value:.4f}')
    axes[0].set_xlabel('Essai', fontsize=12)
    axes[0].set_ylabel(f'F1 validation moyen ({K_FOLDS} plis)', fontsize=12)
    axes[0].set_title("Progrès de l'optimisation Optuna", fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    importances = optuna.importance.get_param_importances(study)
    params_list = list(importances.keys())
    values_list = list(importances.values())
    colors_imp = plt.cm.viridis(np.linspace(0, 1, len(params_list)))
    axes[1].barh(params_list, values_list, color=colors_imp)
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Importance des hyperparamètres', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/optuna_resultats_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return best_params, best_factor, best_patience, study


# ══════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════

def main():
    start_time = time.time()
    start_dt   = datetime.now()

    print("=" * 70)
    print("GRU avec plongements AraVec (GELÉS) - Classification de texte avec OPTUNA")
    print("=" * 70)
    print("⚠️  COURBE ROC UNIQUEMENT SUR TEST.CSV ⚠️")
    print("=" * 70)
    print(f"\nDébut: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Chemins ──
    data_dir    = '/content/drive/MyDrive/BigData_11_2026/data'
    kfold_path  = f'{data_dir}/kfoldsdata.csv'
    test_path   = f'{data_dir}/test.csv'
    output_dir  = 'DL Models/GRU with AraVec'

    if not os.path.exists(kfold_path):
        print(f"\nErreur: Fichier manquant: {kfold_path}")
        return None, None, None, None, None

    if not os.path.exists(test_path):
        print(f"Attention: test.csv non trouvé à {test_path} - évaluation finale ignorée.")

    os.makedirs(output_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, 'Résultats')
    os.makedirs(results_dir, exist_ok=True)

    # ── Chargement des données ──
    X_kfold, y_kfold, X_test, y_test, label_encoder = load_data(
        kfold_path=kfold_path, test_path=test_path,
        text_col='text', label_col='label'
    )

    # ── Vocabulaire ──
    vocab = build_vocabulary(X_kfold)

    # ── Plongements AraVec ──
    print("\nInitialisation des plongements AraVec...")
    aravec_emb = AraVecEmbeddings(embedding_dim=300, data_dir=data_dir)
    aravec_emb.load_model()
    embedding_tensor = aravec_emb.get_embedding_tensor(vocab)

    save_vectorization(vocab, embedding_tensor, label_encoder, aravec_emb,
                       os.path.join(output_dir, 'vectorisation'))

    # ── Optimisation Optuna ──
    print("\n" + "=" * 70)
    print(f"DÉBUT OPTIMISATION OPTUNA AVEC K-FOLDS (K={K_FOLDS})")
    print("=" * 70)

    best_params , best_factor, best_patience, study = run_optuna(X_kfold, y_kfold, vocab, embedding_tensor,
                                     n_trials=15, n_startup_trials=5)

    pd.DataFrame([{**best_params, 'best_factor': best_factor, 'best_patience': best_patience }]).to_csv( 
        f"{results_dir}/meilleurs_hyperparametres_{timestamp}.csv", index=False)

    # ── K-Fold final (PAS DE ROC) ──
    reduction_epochs, mean_val_losses, _, fold_histories = run_kfold_find_reduction_epochs(
        X_kfold, list(y_kfold), vocab, embedding_tensor,
        best_params, best_factor, best_patience, results_dir, timestamp
    )
    df_agg, mean_auc_val, optimal_epochs = run_kfold_best_params(
        fold_histories, results_dir, timestamp
    )
    # ── Entraînement du modèle final ──
    final_model, history, training_time = train_final_model(
        X_kfold, list(y_kfold), vocab, embedding_tensor,
        best_params, reduction_epochs, best_factor, optimal_epochs
    )

    model_path = f"{output_dir}/modele_final_{timestamp}.pth"
    torch.save(final_model.state_dict(), model_path)
    print(f"Modèle final sauvegardé: {model_path}")

    if history:
        plot_final_metrics(history, f"{results_dir}/courbes_metriques_finales_{timestamp}.png")

    total_time = time.time() - start_time
    plot_time_chart(training_time, total_time, optimal_epochs,
                    f"{results_dir}/comparaison_temps_{timestamp}.png")

    # ── ÉVALUATION SUR TEST.CSV AVEC COURBE ROC ──
    if X_test and len(y_test) > 0:
        test_loader = DataLoader(
            TextDataset(X_test, y_test, vocab, max_len=100),
            batch_size=best_params['batch_size'], shuffle=False
        )

        print("\n" + "=" * 70)
        print("📊 ÉVALUATION SUR TEST.CSV AVEC COURBE ROC 📊")
        print("=" * 70)

        criterion = nn.BCELoss()
        (test_loss, test_acc, test_prec, test_rec, test_f1, test_auc,
         predictions, true_labels, probabilities) = evaluate(
            final_model, test_loader, criterion, threshold=0.5
        )

        print(f"\nPERFORMANCE SUR TEST.CSV (seuil 0.5):")
        print(f"  Perte:      {test_loss:.6f}")
        print(f"  Exactitude: {test_acc:.4f}")
        print(f"  Précision:  {test_prec:.4f}")
        print(f"  Rappel:     {test_rec:.4f}")
        print(f"  F1-score:   {test_f1:.4f}")
        print(f"  🎯 AUC-TEST: {test_auc:.4f}")

        # ⭐ TRACER LA COURBE ROC SUR TEST.CSV ⭐
        roc_test_path = f"{results_dir}/roc_test_{timestamp}.png"
        plot_roc_test(true_labels, probabilities, roc_test_path)

        # Sauvegarder les prédictions
        pd.DataFrame({
            'vrai_label':          true_labels,
            'probabilite_predite': probabilities,
            'classe_predite':      predictions
        }).to_csv(f"{results_dir}/predictions_test_{timestamp}.csv", index=False)

        save_misclassifications(X_test, true_labels, predictions, probabilities,
                                label_encoder, results_dir, timestamp)
        
        class_report_df = save_classification_report(
                true_labels, predictions, probabilities, label_encoder, results_dir, timestamp
        )

        df_results = save_evaluation_results(
            X_test, true_labels, predictions, probabilities, label_encoder,
            f"{results_dir}/resultats_evaluation_{timestamp}.csv"
        )

        metrics = compute_metrics(true_labels, predictions)
        save_metrics_report(metrics, f"{results_dir}/rapport_metriques_{timestamp}.csv")

        save_confusion_matrix(true_labels, predictions, label_encoder,
                              f"{results_dir}/matrice_confusion_{timestamp}.csv")
        plot_confusion_matrix(true_labels, predictions, label_encoder,
                              f"{results_dir}/matrice_confusion_{timestamp}.png")

        # Test de permutation
        print("\n" + "=" * 70)
        print("TESTS STATISTIQUES")
        print("=" * 70)
        perm_results = permutation_test(final_model, test_loader, criterion,
                                        n_permutations=999, threshold=0.5)
        plt.savefig(f"{results_dir}/test_permutation_{timestamp}.png")
        plt.close()

        # Informations d'exécution
        runtime_info = {
            'temps_entrainement_secondes': training_time,
            'temps_total_secondes':        total_time,
            'epoques_optimales':           optimal_epochs,
            'seuil_utilise':               0.5,
            'auc_test':                    float(test_auc),
            'auc_validation_info':         float(mean_auc_val),
            'essais_optuna':               15,
            'plongements':                 'geles_pas_de_finetuning'
        }
        pd.DataFrame([runtime_info]).to_csv(
            f"{results_dir}/info_execution_{timestamp}.csv", index=False)

        # ── Résumé final ──
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ FINAL 📊")
        print("=" * 70)
        print(f"✅ Époques du modèle final: {optimal_epochs}")
        print(f"\n🎯 PERFORMANCE SUR TEST.CSV:")
        print(f"   Exactitude: {metrics['exactitude']:.4f}")
        print(f"   Précision:  {metrics['precision']:.4f}")
        print(f"   Rappel:     {metrics['rappel']:.4f}")
        print(f"   F1-score:   {metrics['f1_score']:.4f}")
        print(f"   🏆 AUC-TEST: {test_auc:.4f}")
        print(f"\n⏱️  TEMPS:")
        print(f"   Entraînement: {training_time:.2f}s")
        print(f"   Total:        {total_time:.2f}s")
        print(f"\n📈 COURBE ROC générée sur test.csv")
        print("=" * 70)

        return (final_model, vocab, label_encoder, df_results, metrics,
                total_time, {'auc_test': test_auc})

    else:
        print("\nPas d'ensemble de test. Pipeline terminé.")
        return final_model, vocab, label_encoder, None, None, None, None


# ══════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir('/content/drive/MyDrive/BigData_11_2026')
    print(f"Répertoire de travail: {os.getcwd()}")
    try:
        import optuna
    except ImportError:
        print("Optuna non installé. Installation...")
        os.system("pip install optuna")
        import optuna

    model, vocab, label_encoder, df_results, metrics, total_time, test_results = main()

    if df_results is not None and test_results:
        print(f"\n✅ AUC sur test.csv: {test_results['auc_test']:.4f}")