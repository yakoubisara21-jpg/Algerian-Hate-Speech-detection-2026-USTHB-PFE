"""
FAD-SA-GRU: Fusion Multi-Embedding avec self attention GRU (FastText + AraVec + DziriBERT)
Architecture MLP Fusion pour l'arabe algérien / Arabizi / Darija

PIPELINE COMPLET:
  Phrase → Prétraitement → Tokenisation
  → [FastText(300), AraVec(300), DziriBERT(768)]
  → Concaténation (1368)
  → MLP Fusion (1368→768→512)
  → FAD-SA-GRU (512→256×2=512)
  → Pooling → Dense → Softmax

CONFIGURATION:
  - Plongements FastText      : ENTRAÎNABLES (fine-tuning)
  - Plongements AraVec        : ENTRAÎNABLES (fine-tuning)
  - DziriBERT                 : modèle fine-tuné
  - Optimisation Optuna       : 15 essais (5 aléatoires + 10 TPE)
  - Validation croisée        : StratifiedKFold K=5
  - Époques fixes (Optuna)    : 10
  - Seuil de classification   : 0.5
  - Scheduler                 : ReduceLROnPlateau sur val_loss
  - Gradient clipping         : hyperparamètre Optuna
  - ROC Curve                 : TEST.CSV uniquement
"""

# ══════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
import warnings
import os
import pickle
import json
import random
import time
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.trial import TrialState

import fasttext
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ══════════════════════════════════════════════════════
# NOM DE L'APPROCHE
# ══════════════════════════════════════════════════════

MODEL_NAME = "Fusion Multi-Embedding avec Self-Attention GRU (FAD-SA-GRU)"
MODEL_NAME_SHORT = "FAD-SA-GRU"

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════

SEED              = 42
K_FOLDS           = 5
FIXED_EPOCHS      = 10
N_TRIALS          = 15
N_STARTUP_TRIALS  = 5

FASTTEXT_MODEL_PATH  = '/content/drive/MyDrive/BigData_11_2026/data/cc.ar.300.bin'
ARAVEC_DATA_DIR      = '/content/drive/MyDrive/BigData_11_2026/data'
# Chemin vers le modèle DziriBERT fine-tuné (dossier contenant config.json, model.safetensors…)
DZIRIBERT_MODEL_PATH = '/content/drive/MyDrive/BigData_11_2026/DziriBert/Résultats/modele_final'

DATA_DIR    = '/content/drive/MyDrive/BigData_11_2026/data'
KFOLD_PATH  = f'{DATA_DIR}/kfoldsdata.csv'
TEST_PATH   = f'{DATA_DIR}/test.csv'
OUTPUT_DIR  = 'DL Models/FADSAGRU '

FASTTEXT_DIM  = 300
ARAVEC_DIM    = 300
DZIRIBERT_DIM = 768
FUSION_DIM    = FASTTEXT_DIM + ARAVEC_DIM + DZIRIBERT_DIM  # 1368
MLP_MID_DIM   = 768
MLP_OUT_DIM   = 512
MAX_SEQ_LEN   = 100
DZIRIBERT_MAX_LEN = 128

# ── Graine globale ──
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Appareil utilisé : {device}")

# ── Variables globales partagées Optuna ──
global_train_texts       = None
global_train_labels      = None
global_vocab             = None
global_ft_tensor         = None   # FastText embedding matrix
global_aravec_tensor     = None   # AraVec embedding matrix
global_dziribert_cache   = None   # Dict {text: np.array(768,)} — cache DziriBERT


# ══════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════
# 1. FASTTEXT EMBEDDINGS
# ══════════════════════════════════════════════════════

class FastTextEmbeddings:
    """Gestionnaire des plongements FastText arabes — ENTRAÎNABLES."""

    def __init__(self, model_path=FASTTEXT_MODEL_PATH, embedding_dim=FASTTEXT_DIM):
        self.model_path    = model_path
        self.embedding_dim = embedding_dim
        self.model         = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[FastText] Fichier non trouvé : {self.model_path}")
            return False
        print(f"[FastText] Chargement depuis {self.model_path} ...")
        self.model = fasttext.load_model(self.model_path)
        print("[FastText] Modèle chargé ✓")
        return True

    def get_word_vector(self, word):
        if self.model:
            try:
                return self.model.get_word_vector(word)
            except Exception:
                pass
        return np.random.randn(self.embedding_dim) * 0.01

    def build_embedding_tensor(self, vocab):
        vocab_size       = len(vocab)
        matrix           = np.zeros((vocab_size, self.embedding_dim))
        found = processed = 0
        print(f"[FastText] Construction matrice pour {vocab_size} mots ...")
        for word, idx in vocab.items():
            if word in ('<PAD>', '<UNK>'):
                continue
            vec = self.get_word_vector(word)
            matrix[idx] = vec
            processed += 1
            if not np.all(vec == 0):
                found += 1
        matrix[0] = 0.0  # <PAD>
        # OOV → bruit gaussien
        for word, idx in vocab.items():
            if word == '<PAD>':
                continue
            if np.all(matrix[idx] == 0):
                matrix[idx] = np.random.randn(self.embedding_dim) * 0.01
        if processed:
            print(f"[FastText] Trouvés : {found}/{processed} ({found/processed*100:.1f}%)")
        print("[FastText] Plongements ENTRAÎNABLES ✓")
        return torch.from_numpy(matrix).float()

 
# ══════════════════════════════════════════════════════
# 2. ARAVEC EMBEDDINGS
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
# 3. DZIRIBERT — EMBEDDINGS CONTEXTUELS
# ══════════════════════════════════════════════════════

class DziriBERTEmbedder:
    """
    Charge le modèle DziriBERT fine-tuné et produit des embeddings
    contextuels (moyenne des tokens) de dimension 768.
    Tous les textes sont mis en cache pour éviter les recalculs.
    """

    def __init__(self, model_path=DZIRIBERT_MODEL_PATH, max_len=DZIRIBERT_MAX_LEN):
        self.model_path = model_path
        self.max_len    = max_len
        self.tokenizer  = None
        self.model      = None
        self.cache      = {}

    def load(self):
        if not os.path.exists(self.model_path):
            print(f"[DziriBERT] Chemin non trouvé : {self.model_path}")
            return False
        print(f"[DziriBERT] Chargement du modèle fine-tuné depuis {self.model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model     = AutoModel.from_pretrained(self.model_path)
        self.model.to(device)
        self.model.eval()
        print("[DziriBERT] Modèle chargé ✓")
        return True

    @torch.no_grad()
    def encode_batch(self, texts, batch_size=32):
        """Encode une liste de textes → np.array (N, 768)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc   = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
            enc   = {k: v.to(device) for k, v in enc.items()}
            out   = self.model(**enc)
            # Moyenne des embeddings de tokens (hors padding)
            mask  = enc['attention_mask'].unsqueeze(-1).float()
            embs  = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(embs.cpu().numpy())
        return np.vstack(all_embeddings)

    def build_cache(self, texts):
        """Précalcule et met en cache tous les embeddings DziriBERT."""
        unique_texts = list(set(texts))
        not_cached   = [t for t in unique_texts if t not in self.cache]

        if not_cached:
            print(f"[DziriBERT] Encodage de {len(not_cached)} textes uniques ...")
            embeddings = self.encode_batch(not_cached)
            for text, emb in zip(not_cached, embeddings):
                self.cache[text] = emb
            print(f"[DziriBERT] Cache complet : {len(self.cache)} entrées ✓")

    def get(self, text):
        """Retourne l'embedding DziriBERT d'un texte (depuis le cache)."""
        if text in self.cache:
            return self.cache[text]
        # Si non en cache, encoder à la volée
        emb = self.encode_batch([text])[0]
        self.cache[text] = emb
        return emb


# ══════════════════════════════════════════════════════
# VOCABULAIRE
# ══════════════════════════════════════════════════════

def build_vocabulary(texts):
    """Construit le vocabulaire (TOUS les mots, pas de seuil)."""
    print("Construction du vocabulaire ...")
    word_counts = Counter()
    for text in texts:
        word_counts.update(str(text).split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word in word_counts:
        vocab[word] = len(vocab)
    print(f"Vocabulaire : {len(vocab)} entrées (TOUS les mots inclus) ✓")
    return vocab


# ══════════════════════════════════════════════════════
# DATASET PYTORCH
# ══════════════════════════════════════════════════════

class HybridTextDataset(Dataset):
    """
    Dataset retournant pour chaque texte :
      - indices de tokens (pour FastText + AraVec via nn.Embedding)
      - longueur de séquence réelle
      - embedding DziriBERT pré-calculé (768,)
      - label
    """

    def __init__(self, texts, labels, word_to_idx, dziribert_cache,
                 max_len=MAX_SEQ_LEN):
        self.texts            = [str(t) for t in texts]
        self.labels           = labels
        self.word_to_idx      = word_to_idx
        self.dziribert_cache  = dziribert_cache
        self.max_len          = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]

        # Indices pour les embeddings FastText et AraVec
        unk = self.word_to_idx['<UNK>']
        pad = self.word_to_idx['<PAD>']
        indices = [self.word_to_idx.get(w, unk) for w in text.split()]

        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
            seq_len = self.max_len
        else:
            seq_len = max(len(indices), 1)
            indices += [pad] * (self.max_len - len(indices))

        # Embedding DziriBERT (depuis le cache)
        dziri_emb = self.dziribert_cache.get(text,
                        np.zeros(DZIRIBERT_DIM, dtype=np.float32))

        return (
            torch.tensor(indices,   dtype=torch.long),
            torch.tensor(seq_len,   dtype=torch.long),
            torch.tensor(dziri_emb, dtype=torch.float),
            torch.tensor(label,     dtype=torch.float)
        )



class SelfAttention(nn.Module):
    """
    Mécanisme d'attention sur la séquence GRU.
    Calcule un score d'importance pour chaque token,
    puis produit un vecteur de contexte pondéré.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, gru_output, mask):
        """
        gru_output : (B, T, hidden_dim)
        mask       : (B, T) — True pour les tokens réels, False pour le padding
        Retourne   : (B, hidden_dim) — vecteur de contexte pondéré
        """
        # Score d'attention brut → (B, T, 1)
        scores = self.attention(gru_output)

        # Masquer les tokens de padding avec -inf avant softmax
        scores = scores.squeeze(-1)                          # (B, T)
        scores = scores.masked_fill(~mask, float('-inf'))    # (B, T)

        weights = torch.softmax(scores, dim=1)               # (B, T)

        # Vecteur de contexte : somme pondérée
        context = (gru_output * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        return context, weights

# ══════════════════════════════════════════════════════
# ARCHITECTURE HYBRIDE GRU
# ══════════════════════════════════════════════════════

class HybridGRUClassifier(nn.Module):
    """
    Architecture MLP Fusion :

    FastText(300) ─┐
    AraVec(300)  ──┼─ cat(1368) ─► MLP(1368→768→512) ─► GRU ─► Pooling ─► Classifier
    DziriBERT(768)─┘

    Les plongements FastText et AraVec sont ENTRAÎNABLES (fine-tuning).
    DziriBERT est utilisé comme features pré-calculées (vecteur de phrase).
    """

    def __init__(self, vocab_size, output_dim,
                 ft_embeddings,    # Tenseur FastText (vocab_size, 300)
                 aravec_embeddings,# Tenseur AraVec  (vocab_size, 300)
                 # MLP Fusion
                 mlp_dropout=0.3,
                 # GRU
                 hidden_dim=256,
                 n_layers=2,
                 dropout_gru=0.3,
                 dropout_pre_gru=0.3,
                 dropout_post_gru=0.3,
                 # Classifier
                 clf_hidden=256,
                 clf_dropout=0.3):


                
        super().__init__()

        # ── Couches d'embedding (ENTRAÎNABLES) ──
        self.ft_embedding = nn.Embedding(vocab_size, FASTTEXT_DIM, padding_idx=0)
        self.ft_embedding.weight = nn.Parameter(ft_embeddings.clone())
        self.ft_embedding.weight.requires_grad = True

        self.aravec_embedding = nn.Embedding(vocab_size, ARAVEC_DIM, padding_idx=0)
        self.aravec_embedding.weight = nn.Parameter(aravec_embeddings.clone())
        self.aravec_embedding.weight.requires_grad = True

        # ── MLP Fusion : 1368 → 768 → 512 ──
        # (La dim DziriBERT 768 vient du vecteur de phrase, répété sur chaque token)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(FUSION_DIM, MLP_MID_DIM),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(MLP_MID_DIM, MLP_OUT_DIM),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
        )

        self.dropout_pre_gru = nn.Dropout(dropout_pre_gru)

        # ── GRU ──
        self.gru = nn.GRU(
            input_size=MLP_OUT_DIM,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_gru if n_layers > 1 else 0,
        )


        # hidden_dim 
        gru_out_dim = hidden_dim

        
                         # ── Self-Attention ──
        self.self_attention = SelfAttention(gru_out_dim)

        self.dropout_post_gru = nn.Dropout(dropout_post_gru)

        # ── Classifieur ──
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, clf_hidden),
            nn.ReLU(),
            nn.Dropout(clf_dropout),
            nn.Linear(clf_hidden, output_dim),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, token_ids, seq_lens, dziri_embs):
        """
        token_ids  : (B, T) — indices des tokens
        seq_lens   : (B,)   — longueurs réelles
        dziri_embs : (B, 768) — embeddings DziriBERT de la phrase entière
        """
        B, T = token_ids.shape

        # Embeddings token-level
        ft_emb     = self.ft_embedding(token_ids)        # (B, T, 300)
        aravec_emb = self.aravec_embedding(token_ids)    # (B, T, 300)

        # Broadcast de l'embedding DziriBERT sur tous les tokens
        dziri_exp  = dziri_embs.unsqueeze(1).expand(B, T, DZIRIBERT_DIM)  # (B, T, 768)

        # Concaténation → (B, T, 1368)
        fused = torch.cat([ft_emb, aravec_emb, dziri_exp], dim=-1)

        # MLP Fusion → (B, T, 512)
        fused = self.fusion_mlp(fused)
        fused = self.dropout_pre_gru(fused)

        # BiGRU avec pack/unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            fused, seq_lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        output, _     = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # (B, T, hidden_dim*2)

        # Self-Attention pooling sur les tokens réels
        mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) \
               < seq_lens.unsqueeze(1)                       # (B, T) — booléen
        pooled, attn_weights = self.self_attention(output, mask)
        # pooled : (B, hidden_dim)

        pooled = self.dropout_post_gru(pooled)
        logits = self.classifier(pooled)         # (B, output_dim)
        return self.sigmoid(logits).squeeze(-1)


# ══════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════════════

def load_data(kfold_path=KFOLD_PATH, test_path=TEST_PATH,
              text_col='text', label_col='label'):
    print(f"\nChargement K-Folds : {kfold_path}")
    df_kfold = pd.read_csv(kfold_path, sep=';')
    print(f"  {len(df_kfold)} échantillons chargés")

    X_kfold = df_kfold[text_col].astype(str).tolist()
    y_kfold = df_kfold[label_col].tolist()

    print("Distribution des classes (K-Folds):")
    for lab, cnt in pd.Series(y_kfold).value_counts().items():
        print(f"  {lab}: {cnt} ({cnt/len(y_kfold)*100:.1f}%)")

    X_test, y_test = [], []
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path, sep=';')
        X_test  = df_test[text_col].astype(str).tolist()
        y_test  = df_test[label_col].tolist()
        print(f"\nTest : {len(X_test)} échantillons")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_kfold + y_test)
    y_kfold_enc = label_encoder.transform(y_kfold)
    y_test_enc  = label_encoder.transform(y_test) if y_test else []

    print(f"\nClasses : {label_encoder.classes_}")
    return X_kfold, y_kfold_enc, X_test, y_test_enc, label_encoder


# ══════════════════════════════════════════════════════
# BOUCLES D'ENTRAÎNEMENT / ÉVALUATION
# ══════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, clip_norm):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for token_ids, seq_lens, dziri_embs, labels in loader:
        token_ids  = token_ids.to(device)
        seq_lens   = seq_lens.to(device)
        dziri_embs = dziri_embs.to(device)
        labels     = labels.to(device)

        optimizer.zero_grad()
        outputs = model(token_ids, seq_lens, dziri_embs)
        loss    = criterion(outputs, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall    = recall_score(all_labels,   all_preds, average='binary', zero_division=0)
    f1        = f1_score(all_labels,       all_preds, average='binary', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1


def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for token_ids, seq_lens, dziri_embs, labels in loader:
            token_ids  = token_ids.to(device)
            seq_lens   = seq_lens.to(device)
            dziri_embs = dziri_embs.to(device)
            labels     = labels.to(device)

            outputs    = model(token_ids, seq_lens, dziri_embs)
            loss       = criterion(outputs, labels.float())
            total_loss += loss.item()

            probs = outputs.cpu().numpy()
            preds = (outputs > threshold).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss  = total_loss / len(loader)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall    = recall_score(all_labels,   all_preds, average='binary', zero_division=0)
    f1        = f1_score(all_labels,       all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return avg_loss, accuracy, precision, recall, f1, auc, all_preds, all_labels, all_probs



def train_with_history(model, train_loader, val_loader, epochs, lr, weight_decay,
                       factor, patience, clip_norm):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=1e-7
    )

    history = []
    val_probs_per_epoch = []
    reduction_epochs_fold = []
    prev_lr = lr

    for epoch in range(epochs):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_epoch(
            model, train_loader, optimizer, criterion, clip_norm
        )
        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1, vl_auc, _, vl_labels, vl_probs = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(vl_loss)  # monitors val_loss (same as GRU)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            reduction_epochs_fold.append(epoch)  # 0-indexed
        prev_lr = current_lr

        history.append({
            'epoch': epoch + 1,
            'train_loss': tr_loss, 'train_accuracy': tr_acc,
            'train_precision': tr_prec, 'train_recall': tr_rec, 'train_f1': tr_f1,
            'val_loss': vl_loss, 'val_accuracy': vl_acc,
            'val_precision': vl_prec, 'val_recall': vl_rec,
            'val_f1': vl_f1, 'val_auc': vl_auc,
        })
        val_probs_per_epoch.append(vl_probs)

    return history, val_probs_per_epoch, reduction_epochs_fold


# ══════════════════════════════════════════════════════
# OPTUNA — OBJECTIF
# ══════════════════════════════════════════════════════

def make_model(vocab_size, output_dim, ft_tensor, aravec_tensor, params):
    return HybridGRUClassifier(
        vocab_size=vocab_size,
        output_dim=output_dim,
        ft_embeddings=ft_tensor,
        aravec_embeddings=aravec_tensor,
        mlp_dropout=params['mlp_dropout'],
        hidden_dim=params['hidden_dim'],
        n_layers=params['n_layers'],
        dropout_gru=params['dropout_gru'],
        dropout_pre_gru=params['dropout_pre_gru'],
        dropout_post_gru=params['dropout_post_gru'],
        clf_hidden=params['clf_hidden'],
        clf_dropout=params['clf_dropout'],
    ).to(device)

def optuna_objective_phase1(trial):
    global global_train_texts, global_train_labels, global_vocab
    global global_ft_tensor, global_aravec_tensor, global_dziribert_cache

    params = {
        'hidden_dim':        trial.suggest_int('hidden_dim',        64,  256, step=32),
        'n_layers':          trial.suggest_int('n_layers',           1,    2),
        'mlp_dropout':       trial.suggest_float('mlp_dropout',     0.1,  0.5, step=0.1),
        'dropout_pre_gru':   trial.suggest_float('dropout_pre_gru', 0.1,  0.5, step=0.1),
        'dropout_gru':       trial.suggest_float('dropout_gru',     0.2,  0.6, step=0.1),
        'dropout_post_gru':  trial.suggest_float('dropout_post_gru',0.1,  0.5, step=0.1),
        'clf_hidden':        trial.suggest_int('clf_hidden',         64,  256, step=64),
        'clf_dropout':       trial.suggest_float('clf_dropout',      0.1,  0.5, step=0.1),
        'learning_rate':     trial.suggest_float('learning_rate',    5e-5, 5e-3, log=True),
        'weight_decay':      trial.suggest_float('weight_decay',     1e-4, 1e-2, log=True),
        'batch_size':        trial.suggest_categorical('batch_size', [32, 64, 128]),
        'clip_norm':         trial.suggest_float('clip_norm',        0.5,  5.0, log=True),
        'factor':            trial.suggest_float('factor',           0.1,  0.9, step=0.05),
        'patience':          trial.suggest_int('patience',           0,    4),   # ← added
    }

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    all_fold_val_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(global_train_texts, global_train_labels)):
        set_seed(SEED + fold)

        tr_texts  = [global_train_texts[i] for i in train_idx]
        tr_labels = [global_train_labels[i] for i in train_idx]
        vl_texts  = [global_train_texts[i] for i in val_idx]
        vl_labels = [global_train_labels[i] for i in val_idx]

        tr_loader = DataLoader(
            HybridTextDataset(tr_texts, tr_labels, global_vocab,
                              global_dziribert_cache, MAX_SEQ_LEN),
            batch_size=params['batch_size'], shuffle=True
        )
        vl_loader = DataLoader(
            HybridTextDataset(vl_texts, vl_labels, global_vocab,
                              global_dziribert_cache, MAX_SEQ_LEN),
            batch_size=params['batch_size'], shuffle=False
        )

        model = make_model(len(global_vocab), 1,
                           global_ft_tensor, global_aravec_tensor, params)

        history, _, _ = train_with_history(   # reductions discarded in Optuna
            model, tr_loader, vl_loader, FIXED_EPOCHS,
            params['learning_rate'], params['weight_decay'],
            params['factor'], params['patience'], params['clip_norm']
        )
        all_fold_val_f1s.append([e['val_f1'] for e in history])

    mean_val_f1s = np.mean(all_fold_val_f1s, axis=0)
    return float(max(mean_val_f1s))


def run_optuna_phase1(train_texts, train_labels, vocab, ft_tensor, aravec_tensor, dziribert_cache):
    """PHASE 1 - Optimisation Optuna pour trouver le meilleur factor"""
    global global_train_texts, global_train_labels, global_vocab
    global global_ft_tensor, global_aravec_tensor, global_dziribert_cache

    global_train_texts     = train_texts
    global_train_labels    = train_labels
    global_vocab           = vocab
    global_ft_tensor       = ft_tensor
    global_aravec_tensor   = aravec_tensor
    global_dziribert_cache = dziribert_cache

    print("\n" + "=" * 70)
    print("🔬 PHASE 1 - OPTUNA")
    print(f"   ReduceLROnPlateau live sur val_loss")
    print(f"   K={K_FOLDS} folds, {FIXED_EPOCHS} époques fixes")
    print(f"   patience et factor = hyperparamètres Optuna")
    print(f"   Objectif: maximisation du F1 de validation")
    print("=" * 70)

    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=SEED)
    study   = optuna.create_study(
        direction='maximize',
        study_name='fadgru_f1_maximisation',
        sampler=sampler
    )
    study.optimize(optuna_objective_phase1, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    best_factor   = best.params['factor']
    best_patience = best.params['patience']   # ← extracted separately

    best_params = {k: v for k, v in best.params.items()
                   if k not in ('factor', 'patience')}   # ← both excluded
    best_params['epochs'] = FIXED_EPOCHS

    print(f"\n✅ Meilleur F1: {best.value:.4f} | factor={best_factor} | patience={best_patience}")
    return best_params, best_factor, best_patience, study   # ← 4 return values

# ══════════════════════════════════════════════════════
# K-FOLD FINAL AVEC LES MEILLEURS HYPERPARAMÈTRES
# ══════════════════════════════════════════════════════
def run_kfold_find_reduction_epochs(all_texts, all_labels, vocab, ft_tensor, aravec_tensor,
                                    dziribert_cache, best_params, best_factor, best_patience,
                                    results_dir, timestamp):
    print("\n" + "=" * 70)
    print("PHASE 2 - K-FOLD: RECHERCHE DES ÉPOQUES DE RÉDUCTION")
    print(f"   best_factor={best_factor}, best_patience={best_patience}")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    all_fold_val_losses = []
    all_fold_histories  = []
    all_fold_reduction_epochs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
        set_seed(SEED + fold_idx)
        print(f"\n--- Fold {fold_idx+1}/{K_FOLDS} ---")

        tr_texts  = [all_texts[i] for i in train_idx]
        tr_labels = [all_labels[i] for i in train_idx]
        vl_texts  = [all_texts[i] for i in val_idx]
        vl_labels = [all_labels[i] for i in val_idx]

        tr_loader = DataLoader(
            HybridTextDataset(tr_texts, tr_labels, vocab, dziribert_cache, MAX_SEQ_LEN),
            batch_size=best_params['batch_size'], shuffle=True
        )
        vl_loader = DataLoader(
            HybridTextDataset(vl_texts, vl_labels, vocab, dziribert_cache, MAX_SEQ_LEN),
            batch_size=best_params['batch_size'], shuffle=False
        )

        model = make_model(len(vocab), 1, ft_tensor, aravec_tensor, best_params)

        history, _, reduction_epochs_fold = train_with_history(
            model, tr_loader, vl_loader, FIXED_EPOCHS,
            best_params['learning_rate'], best_params['weight_decay'],
            best_factor, best_patience, best_params['clip_norm']
        )
        all_fold_val_losses.append([e['val_loss'] for e in history])
        all_fold_histories.append(history)
        all_fold_reduction_epochs.append(reduction_epochs_fold)
        print(f"  Fold {fold_idx+1} reduction epochs: {[e+1 for e in reduction_epochs_fold]}")

    mean_val_losses = np.mean(all_fold_val_losses, axis=0)
    std_val_losses  = np.std(all_fold_val_losses,  axis=0)

    # Majority vote — same logic as GRU
    all_reductions = [e for sublist in all_fold_reduction_epochs for e in sublist]
    if all_reductions:
        reduction_counter = Counter(all_reductions)
        reduction_epochs  = sorted([e for e, count in reduction_counter.items()
                                    if count > K_FOLDS / 2])
        print(f"\nÉpoques de réduction (majorité): {[e+1 for e in reduction_epochs]}")
    else:
        reduction_epochs = []
        print("\nAucune réduction détectée")

    with open(f"{results_dir}/reduction_epochs_{timestamp}.json", 'w') as f:
        json.dump({
            'best_factor':      best_factor,
            'best_patience':    best_patience,
            'reduction_epochs': reduction_epochs,
            'mean_val_losses':  mean_val_losses.tolist()
        }, f, indent=2)

    return reduction_epochs, mean_val_losses, std_val_losses, all_fold_histories

def _plot_kfold_metrics(df, save_path):
    epochs = df['epoque']
    pairs  = [
        ('train_loss_moy',      'val_loss_moy',      'Perte'),
        ('train_accuracy_moy',  'val_accuracy_moy',  'Exactitude'),
        ('train_precision_moy', 'val_precision_moy', 'Précision'),
        ('train_recall_moy',    'val_recall_moy',    'Rappel'),
        ('train_f1_moy',        'val_f1_moy',        'F1-score'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (tr_col, vl_col, lbl) in enumerate(pairs):
        axes[i].plot(epochs, df[tr_col], 'b-o', label='Entraînement', linewidth=2, markersize=5)
        axes[i].plot(epochs, df[vl_col], 'r-s', label='Validation',   linewidth=2.5, markersize=7)
        axes[i].set_xlabel('Époque')
        axes[i].set_ylabel(lbl)
        axes[i].set_title(f'{lbl} — (moy. sur {K_FOLDS} plis)', fontweight='bold')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    axes[5].set_visible(False)
    plt.suptitle(f'Métriques de la validation croisée — {MODEL_NAME}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé : {save_path}")


class ManualLRReducer:
    """Gestionnaire manuel des réductions du learning rate"""
    
    def __init__(self, optimizer, reduction_epochs, factor):
        self.optimizer = optimizer
        self.reduction_epochs = set(reduction_epochs)
        self.factor = factor
        self.reduction_count = 0
        
    def step(self, epoch):
        if epoch in self.reduction_epochs:
            old_lr = self.optimizer.param_groups[0]['lr']
            new_lr =  max(old_lr * self.factor, 1e-7)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.reduction_count += 1
            print(f" MANUAL LR REDUCTION at epoch {epoch+1}: {old_lr:.2e} → {new_lr:.2e}")
            return True
        return False
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

# ══════════════════════════════════════════════════════
# ENTRAÎNEMENT FINAL
# ══════════════════════════════════════════════════════
def train_final_model_with_manual_scheduler(train_texts, train_labels, vocab, ft_tensor, aravec_tensor,
                                            dziribert_cache, best_params, reduction_epochs, 
                                            best_factor, results_dir, timestamp, final_epochs=None):
    """
    PHASE 3 - Entraînement final avec réductions manuelles du LR
    """
    print("\n" + "=" * 70)
    print("🔬 PHASE 3 - ENTRAÎNEMENT FINAL")
    print(f"   Réduction manuelle du LR aux epochs: {[e+1 for e in reduction_epochs]}")
    print(f"   Facteur de réduction: {best_factor}")
    print("=" * 70)

    set_seed(SEED)
    tr_loader = DataLoader(
        HybridTextDataset(train_texts, train_labels, vocab, dziribert_cache, MAX_SEQ_LEN),
        batch_size=best_params['batch_size'], shuffle=True
    )
    model = make_model(len(vocab), 1, ft_tensor, aravec_tensor, best_params)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux      : {total_params:,}")
    print(f"Paramètres entraînables: {trainable_params:,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=best_params['learning_rate'],
                           weight_decay=best_params['weight_decay'])
    
    lr_reducer = ManualLRReducer(optimizer, reduction_epochs, best_factor)

    history = []
    start_time = time.time()
    
     # Utiliser le nombre d'époques passé en paramètre ou la valeur par défaut
    if final_epochs is None:
        n_epochs = FIXED_EPOCHS
    else:
        n_epochs = final_epochs

    print(f"\nDémarrage entraînement final...")
    print(f"LR initial: {best_params['learning_rate']:.2e}")
    
    for epoch in range(n_epochs):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_epoch(
            model, tr_loader, optimizer, criterion, best_params['clip_norm']
        )
        
        lr_reducer.step(epoch)
        current_lr = lr_reducer.get_current_lr()
        
        history.append({
            'epoch': epoch + 1, 
            'train_loss': tr_loss, 
            'train_accuracy': tr_acc,
            'train_precision': tr_prec, 
            'train_recall': tr_rec, 
            'train_f1': tr_f1,
            'learning_rate': current_lr
        })
        
        reduction_marker = " 🔻" if epoch in reduction_epochs else ""
        print(f"  Époque {epoch+1:2d}/{n_epochs} | "
              f"Perte:{tr_loss:.4f} Acc:{tr_acc:.4f} F1:{tr_f1:.4f} | "
              f"LR: {current_lr:.2e}{reduction_marker}")

    training_time = time.time() - start_time
    print(f"\n✅ Entraînement terminé : {training_time:.1f}s")

    model_path = f"{os.path.dirname(results_dir)}/modele_final_{timestamp}.pth"
    torch.save(model, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    return model, history, training_time


def _plot_final_metrics(history, save_path):
    epochs    = [e['epoch']           for e in history]
    series    = [
        ([e['train_loss']      for e in history], 'Perte'),
        ([e['train_accuracy']  for e in history], 'Exactitude'),
        ([e['train_f1']        for e in history], 'F1-score'),
        ([e['train_precision'] for e in history], 'Précision'),
        ([e['train_recall']    for e in history], 'Rappel'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (data, lbl) in enumerate(series):
        axes[i].plot(epochs, data, 'b-o', linewidth=2, markersize=5)
        axes[i].set_xlabel('Époque')
        axes[i].set_ylabel(lbl)
        axes[i].set_title(f'{lbl} (entraînement final)', fontweight='bold')
        axes[i].grid(alpha=0.3)
        axes[i].set_xticks(epochs)
    axes[5].set_visible(False)
    plt.suptitle(f'Entraînement final —  {MODEL_NAME} ',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════
# ÉVALUATION TEST & ROC
# ══════════════════════════════════════════════════════

def evaluate_on_test(model, test_texts, test_labels, vocab, dziribert_cache,
                      best_params, label_encoder, results_dir, timestamp):
    print("\n" + "=" * 70)
    print("ÉVALUATION SUR TEST.CSV")
    print("=" * 70)

    test_loader = DataLoader(
        HybridTextDataset(test_texts, test_labels, vocab, dziribert_cache, MAX_SEQ_LEN),
        batch_size=best_params['batch_size'], shuffle=False
    )
    criterion = nn.BCELoss()
    (te_loss, te_acc, te_prec, te_rec, te_f1, te_auc,
     predictions, true_labels, probabilities) = evaluate(model, test_loader, criterion)

    print(f"\nPERFORMANCE TEST.CSV:")
    print(f"  Perte      : {te_loss:.6f}")
    print(f"  Exactitude : {te_acc:.4f}")
    print(f"  Précision  : {te_prec:.4f}")
    print(f"  Rappel     : {te_rec:.4f}")
    print(f"  F1-score   : {te_f1:.4f}")
    print(f"  AUC        : {te_auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {te_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7,
            label='Random Classifier (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'ROC Curve — {MODEL_NAME}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = f"{results_dir}/roc_test_{timestamp}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC Curve sauvegardée : {roc_path}")

    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
        roc_path.replace('.png', '_data.csv'), index=False)

    # Matrice de confusion
    cm = confusion_matrix(true_labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Prédiction', fontweight='bold')
    ax.set_ylabel('Réel',       fontweight='bold')
    ax.set_title(f'Matrice de confusion —  {MODEL_NAME}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/matrice_confusion_{timestamp}.png", dpi=300)
    plt.close()

    # ADD:
    df_cm = pd.DataFrame(cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_)
    df_cm.to_csv(f"{results_dir}/matrice_confusion_{timestamp}.csv")

    print(f"Matrice de confusion sauvegardée : {results_dir}/matrice_confusion_{timestamp}.csv")

    # Sauvegarde CSV prédictions
    pd.DataFrame({
        'texte':                    test_texts,
        'vrai_label':               label_encoder.inverse_transform(np.array(true_labels).astype(int)),
        'label_predit':             label_encoder.inverse_transform(np.array(predictions).astype(int)),
        'est_correct':              (np.array(true_labels).astype(int) ==
                                     np.array(predictions).astype(int)),
        'probabilite_positive':     np.array(probabilities),
        'probabilite_negative': 1 - np.array(probabilities)

    }).to_csv(f"{results_dir}/predictions_test_{timestamp}.csv", index=False)

    metrics = {
        'exactitude': te_acc, 'precision': te_prec,
        'rappel': te_rec, 'f1_score': te_f1, 'auc': te_auc
    }
    rapport = {
        'Exactitude': te_acc,
        'Précision':  te_prec,
        'Rappel':     te_rec,
        'F1_score':   te_f1,
        'ROC AUC':    te_auc
    }
    pd.DataFrame([rapport]).to_csv(f"{results_dir}/rapport_metriques_{timestamp}.csv", index=False)   

    return metrics, true_labels, predictions, probabilities

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


def save_classification_report(true_labels, predictions, label_encoder, output_dir, timestamp):
    """Génère et sauvegarde le classification report en CSV."""
    from sklearn.metrics import classification_report
    
    # Obtenir les noms des classes
    target_names = label_encoder.classes_
    
    # Convertir en chaînes si nécessaire
    target_names = [str(cls) if not isinstance(cls, str) else cls for cls in target_names]
    
    # Générer le rapport
    class_report = classification_report(
        true_labels, 
        predictions, 
        target_names=target_names, 
        output_dict=True
    )
    
    # Convertir en DataFrame
    class_report_df = pd.DataFrame(class_report).transpose()
    
    # Afficher dans la console
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT:")
    print("=" * 70)
    print(classification_report(true_labels, predictions, target_names=target_names))
    
    # Sauvegarder en CSV
    report_path = os.path.join(output_dir, f'classification_report_{timestamp}.csv')
    class_report_df.to_csv(report_path, index=True, encoding='utf-8-sig')
    print(f"\n✅ Classification report sauvegardé : {report_path}")
    
    return class_report_df
    
# ══════════════════════════════════════════════════════
# SAUVEGARDE VECTORISATION
# ══════════════════════════════════════════════════════

def save_vectorization(vocab, ft_tensor, aravec_tensor, label_encoder,
                        save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'vocabulaire.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    np.save(os.path.join(save_dir, 'ft_matrice.npy'),     ft_tensor.numpy())
    np.save(os.path.join(save_dir, 'aravec_matrice.npy'), aravec_tensor.numpy())
    with open(os.path.join(save_dir, 'encodeur_labels.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    info = {
        'taille_vocab':     len(vocab),
        'dim_fasttext':     FASTTEXT_DIM,
        'dim_aravec':       ARAVEC_DIM,
        'dim_dziribert':    DZIRIBERT_DIM,
        'dim_fusion':       FUSION_DIM,
        'mlp_dims':         f'{FUSION_DIM}→{MLP_MID_DIM}→{MLP_OUT_DIM}',
        'architecture':     'FastText+AraVec+DziriBERT → MLP Fusion → BiGRU',
        'date_sauvegarde':  datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(save_dir, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"[Vectorisation] Sauvegardée dans {save_dir} ✓")



def plot_time_chart(training_time, total_time, optimal_epochs, save_path):
    """
    Graphique à deux barres: temps d'entraînement final et temps total d'exécution.
    """
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
    ax.set_title('Temps d\'exécution - FAD-SA-GRU', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique de temps sauvegardé: {save_path}")


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
        for token_ids, seq_lens, dziri_embs, labels in test_loader:
            token_ids  = token_ids.to(device)
            seq_lens   = seq_lens.to(device)
            dziri_embs = dziri_embs.to(device)
            outputs = model(token_ids, seq_lens, dziri_embs)
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
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════
def main():
    total_start = time.time()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    print("=" * 70)
    print(f"{MODEL_NAME} — STRATÉGIE EN 3 PHASES")
    print("=" * 70)
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_dir = os.path.join(OUTPUT_DIR, 'Résultats')
    os.makedirs(results_dir, exist_ok=True)
 
    # Chargement des données
    X_kfold, y_kfold, X_test, y_test, label_encoder = load_data()


    # Vocabulaire
    vocab = build_vocabulary(X_kfold)
 
    # FastText
    ft = FastTextEmbeddings()
    ft.load_model()
    ft_tensor = ft.build_embedding_tensor(vocab)
 
    # AraVec
    aravec = AraVecEmbeddings(embedding_dim=ARAVEC_DIM, data_dir=ARAVEC_DATA_DIR)
    aravec.load_model()
    aravec_tensor = aravec.get_embedding_tensor(vocab)
 
    # DziriBERT
    dziribert = DziriBERTEmbedder()
    dziribert.load()
    all_texts = list(set(X_kfold + (X_test if X_test else [])))
    dziribert.build_cache(all_texts)
 
    # Sauvegarde vectorisation
    save_vectorization(vocab, ft_tensor, aravec_tensor, label_encoder,
                        os.path.join(OUTPUT_DIR, 'vectorisation'))
 
    # ═══════════════════════════════════════════════════
    # PHASE 1 - OPTUNA (trouver le meilleur factor)
    # ═══════════════════════════════════════════════════
    best_params, best_factor, best_patience, study = run_optuna_phase1(
       X_kfold, list(y_kfold), vocab, ft_tensor, aravec_tensor, dziribert.cache
    )
    
    pd.DataFrame([{'best_factor': best_factor, **best_params}]).to_csv(
        f"{results_dir}/phase1_meilleurs_params_{timestamp}.csv", index=False)
 
    # ═══════════════════════════════════════════════════
    # PHASE 2 - Trouver les epochs de réduction
    # ═══════════════════════════════════════════════════
    reduction_epochs, mean_val_losses, std_val_losses, fold_histories = run_kfold_find_reduction_epochs(
        X_kfold, list(y_kfold), vocab, ft_tensor, aravec_tensor,
        dziribert.cache, best_params, best_factor, best_patience,   # ← best_patience added
        results_dir, timestamp
    )


    # === GRAPHIQUE 1: K-Fold (toutes les époques 1-10) ===
    print("\n📊 Génération du graphique K-Fold (toutes les époques)...")
    epochs_list = list(range(1, FIXED_EPOCHS + 1))
    
    # Calculer les métriques moyennes sur tous les folds
    mean_train_loss = np.mean([[h['train_loss'] for h in hist] for hist in fold_histories], axis=0)
    mean_val_loss = np.mean([[h['val_loss'] for h in hist] for hist in fold_histories], axis=0)
    mean_train_acc = np.mean([[h['train_accuracy'] for h in hist] for hist in fold_histories], axis=0)
    mean_val_acc = np.mean([[h['val_accuracy'] for h in hist] for hist in fold_histories], axis=0)
    mean_train_f1 = np.mean([[h['train_f1'] for h in hist] for hist in fold_histories], axis=0)
    mean_val_f1 = np.mean([[h['val_f1'] for h in hist] for hist in fold_histories], axis=0)

    mean_train_prec = np.mean([[h['train_precision'] for h in hist] for hist in fold_histories], axis=0)
    mean_val_prec   = np.mean([[h['val_precision']   for h in hist] for hist in fold_histories], axis=0)
    mean_train_rec  = np.mean([[h['train_recall']    for h in hist] for hist in fold_histories], axis=0)
    mean_val_rec    = np.mean([[h['val_recall']      for h in hist] for hist in fold_histories], axis=0)

    
    kfold_df = pd.DataFrame({
        'epoque': epochs_list,
        'train_loss_moy': mean_train_loss,
        'val_loss_moy': mean_val_loss,
        'train_accuracy_moy': mean_train_acc,
        'val_accuracy_moy': mean_val_acc,
        'train_precision_moy': mean_train_prec,  
        'val_precision_moy':   mean_val_prec,     
        'train_recall_moy':    mean_train_rec,    
        'val_recall_moy':      mean_val_rec,      
        'train_f1_moy': mean_train_f1,
        'val_f1_moy': mean_val_f1,
    })
    
    _plot_kfold_metrics(kfold_df, f"{results_dir}/kfold_metrics_all_epochs_{timestamp}.png")

    kfold_df.to_csv(f"{results_dir}/kfold_metriques_agregees_{timestamp}.csv", index=False)
    
# === GRAPHIQUE 2: K-Fold (jusqu'à l'époque optimale = min val loss moyenne) ===
    optimal_epoch = int(np.argmin(mean_val_losses))   # index 0-based
    print(f"\n📊 Époque optimale (min val loss moyenne): epoch {optimal_epoch + 1} "
        f"(val_loss={mean_val_losses[optimal_epoch]:.6f})")

    kfold_df_optimal = kfold_df[kfold_df['epoque'] <= optimal_epoch + 1].copy()
    _plot_kfold_metrics(kfold_df_optimal,
                f"{results_dir}/kfold_metrics_optimal_epoch_{timestamp}.png")
    
    # ═══════════════════════════════════════════════════
    # PHASE 3 - Entraînement final (jusqu'à l'époque optimale)
    # ═══════════════════════════════════════════════════
    # Calculer le nombre d'époques pour l'entraînement final
    # optimal_epoch est déjà calculé ci-dessus via np.argmin(mean_val_losses)
    final_epochs = optimal_epoch + 1   # 0-based index → nombre d'époques
    print(f"\n🎯 Entraînement final sur {final_epochs} époques "
      f"(époque optimale = {final_epochs}, val_loss min = {mean_val_losses[optimal_epoch]:.6f})")
    
    final_model, history, training_time = train_final_model_with_manual_scheduler(
        X_kfold, list(y_kfold), vocab, ft_tensor, aravec_tensor,
        dziribert.cache, best_params, reduction_epochs, best_factor, 
        results_dir, timestamp, final_epochs  # ← Passer le nombre d'époques
    )
    
    # === GRAPHIQUE 3: Entraînement final ===
    print("\n📊 Génération du graphique d'entraînement final...")
    _plot_final_metrics(history, f"{results_dir}/final_training_metrics_{timestamp}.png")

    
    pd.DataFrame(history).to_csv(
        f"{results_dir}/final_training_metrics_{timestamp}_donnees.csv", index=False)
    
    # Évaluation sur test
    if X_test and len(y_test) > 0:
        metrics, true_labels, predictions, probabilities = evaluate_on_test(
            final_model, X_test, list(y_test), vocab, dziribert.cache,
            best_params, label_encoder, results_dir, timestamp
        )
       
        fp, fn = save_misclassifications(
            X_test, true_labels, predictions, probabilities,
            label_encoder, results_dir, timestamp
        )

        class_report_df = save_classification_report(
            true_labels, predictions, label_encoder, results_dir, timestamp
        )


        test_loader_perm = DataLoader(
            HybridTextDataset(X_test, list(y_test), vocab, dziribert.cache, MAX_SEQ_LEN),
            batch_size=best_params['batch_size'], shuffle=False
        )

        criterion_perm = nn.BCELoss()

        perm_results = permutation_test(final_model, test_loader_perm, criterion_perm,
                                 n_permutations=999, threshold=0.5)
        
        plt.savefig(f"{results_dir}/test_permutation_{timestamp}.png")
        plt.close()
        
        
        # Afficher un résumé
        print(f"\n📊 Résumé des mauvaises classifications:")
        print(f"   Faux positifs (FP): {len(fp)}")
        print(f"   Faux négatifs (FN): {len(fn)}")
        


    else:
        metrics = {}
        print("Pas de test.csv — évaluation finale ignorée.")
    
    total_time = time.time() - total_start



 
    runtime_info = {
        'temps_entrainement_secondes': training_time,
        'temps_total_secondes':        total_time,
        'epoques_optimales':           final_epochs,
        'seuil_utilise':               0.5,
        'auc_test':                    float(metrics.get('auc', 0)),
        'essais_optuna':               N_TRIALS,
        'plongements':                 'fasttext+aravec_entrainables_dziribert_finetuned'
    }

    pd.DataFrame([runtime_info]).to_csv(
        f"{results_dir}/info_execution_{timestamp}.csv", index=False)
    

    plot_time_chart(training_time, total_time, final_epochs,
                f"{results_dir}/comparaison_temps_{timestamp}.png")
    
    # Résumé final
    print("\n" + "=" * 70)
    print(f"📊 RÉSUMÉ FINAL — {MODEL_NAME}")
    print("=" * 70)
    print(f"Phase 1 - Best factor: {best_factor} | patience: {best_patience}")
    print(f"Phase 2 - Epoch optimale (min val loss): {optimal_epoch + 1} "
        f"(val_loss={mean_val_losses[optimal_epoch]:.6f})")
    print(f"Phase 2 - Epochs de réduction LR: {[e+1 for e in reduction_epochs]}")

    if metrics:
        print(f"\n📈 Performance sur test.csv :")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
    print(f"\n⏱️  Temps total : {total_time:.1f}s ({total_time/60:.2f} min)")
    print("=" * 70)
 
    return final_model, vocab, label_encoder, metrics
# ══════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    os.chdir('/content/drive/MyDrive/BigData_11_2026')
    print(f"Répertoire de travail : {os.getcwd()}")

    for pkg in ['optuna', 'transformers', 'fasttext']:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installation de {pkg} ...")
            os.system(f"pip install {pkg} -q")

    try:
        model, vocab, label_encoder, metrics = main()
    except Exception as e:
        import traceback
        print(f"\nErreur : {e}")
        traceback.print_exc()
