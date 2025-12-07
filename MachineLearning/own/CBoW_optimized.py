# ✔ Actualiza solo 1 palabra positiva
# ✔ Y unas pocas palabras negativas (p. ej., 5)
# ✔ NO calcula softmax
# ✔ NO actualiza todo el vocabulario#
# ✔ Mucho más rápido y eficiente
# ✔ Es el algoritmo que implementa Word2Vec original (Mikolov et al., 2013)

# CBoW_optimized_safe.py
import numpy as np
import re
from collections import Counter, deque
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import traceback

# --------------------------
# Configuration (tweak here)
# --------------------------
BASE_DIR = "/home/sant_vz6/Escritorio/ProyectosInteresantes/MachineLearning/own"

# Carpeta segura para archivos npy
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATA_FILE = os.path.join(BASE_DIR , "dataset/spanish_corpora/TED.txt")
MAX_VOCAB = 30000
UNIGRAM_TABLE_SIZE = 1_000_000  # 1M entries (int32 ~ 4MB)
EMBEDDING_DIM = 50
BUFFER_MAX = 10000
CHECKPOINT_TOKENS = 100_000
INITIAL_LR = 0.025
MIN_LR = 0.0001
NEG_SAMPLES = 5
SEED = 1
WEIGHT_CLIP = 100.0            # absolute clip for weights to avoid inf
GRAD_CLIP = 5.0                # clip gradient g
SCORE_CLIP = 20.0              # clip scores before sigmoid
Vocab_file = f"{DATA_DIR}/vocab.npy"
counts_file = f"{DATA_DIR}/vocab_counts.npy"
unigram_file = f"{DATA_DIR}/unigram_table.npy"
W1_file = f"{DATA_DIR}/W1.npy"
W2_file = f"{DATA_DIR}/W2.npy"
processed_file = f"{DATA_DIR}/processed_tokens.npy"
total_tokens_file = f"{DATA_DIR}/total_tokens.npy"

# --------------------------
# Utility helpers
# --------------------------
def safe_load_npy(path, default=None):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        print(f"[WARN] Failed to load {path} (corrupt?). Will regenerate. Traceback:")
        traceback.print_exc()
        return default

def safe_save_npy(path, arr):
    tmp = path + ".tmp"
    if not tmp.endswith(".npy"):
        tmp += ".npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def safe_save_obj(path, obj):
    # uses numpy to allow dict save (allow_pickle)
    tmp = path + ".tmp"
    np.save(tmp, obj)
    os.replace(tmp + ".npy" if tmp.endswith(".npy") else tmp, path)

# --------------------------
# Model
# --------------------------
class CBOWNegSampling:
    def __init__(self, vocab_list, counts_dict, embedding_dim=EMBEDDING_DIM, seed=SEED, unigram_table=None):
        self.rng = np.random.default_rng(seed)
        self.embedding_dim = int(embedding_dim)
        self.vocab = list(vocab_list)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.V = len(self.vocab)

        # Ensure consistent dtype float32 for weights
        self.W1 = self.rng.normal(0, 0.001, (self.V, self.embedding_dim)).astype(np.float32)
        self.W2 = np.zeros((self.embedding_dim, self.V), dtype=np.float32)

        # unigram table
        if unigram_table is not None:
            self.unigram_table = np.array(unigram_table, dtype=np.int32)
        else:
            freqs = np.array([counts_dict[w] for w in self.vocab], dtype=np.float64)
            probs = freqs ** 0.75
            probs /= probs.sum()
            # sample once to create table (deterministic if seed fixed)
            self.unigram_table = self.rng.choice(self.V, size=UNIGRAM_TABLE_SIZE, p=probs).astype(np.int32)
        self.table_size = len(self.unigram_table)

    def sample_negatives(self, k, exclude_idx=None):
        """
        Samples k negatives from unigram table.
        If exclude_idx is provided, it may still appear (allowed by original word2vec).
        """
        return self.unigram_table[self.rng.integers(0, self.table_size, size=k)]

    def _sigmoid(self, x):
        # stable sigmoid with clipping
        x = np.clip(x, -SCORE_CLIP, SCORE_CLIP)
        return 1.0 / (1.0 + np.exp(-x))

    def train_step(self, context_idxs, target_idx, lr, neg_samples=NEG_SAMPLES):
        """
        context_idxs: list or iterable of ints (indices)
        target_idx: int
        lr: float
        """
        if len(context_idxs) == 0:
            return

        # compute average context embedding (float32)
        x = np.mean(self.W1[np.array(context_idxs, dtype=np.int64)], axis=0)  # (embedding_dim,)
        x = x.astype(np.float32)

        # positive
        score_pos = float(x @ self.W2[:, target_idx])          # scalar
        sig_pos = self._sigmoid(score_pos)
        grad_pos = 1.0 - sig_pos                               # derivative for log-sigmoid

        # negatives
        neg_idx = self.sample_negatives(neg_samples)
        score_neg = x @ self.W2[:, neg_idx]                    # (neg_samples,)
        sig_neg = self._sigmoid(-score_neg)                    # sigmoid(-score_neg)
        grad_neg = -sig_neg                                     # derivative piece

        # update W2 (gradient descent)
        # positive
        update_pos = (lr * grad_pos) * x                       # (embedding_dim,)
        self.W2[:, target_idx] -= update_pos.astype(np.float32)

        # negatives (broadcast): W2[:, neg_idx] -= lr * x[:,None] * grad_neg[None,:]
        grad_neg_f = grad_neg.astype(np.float32)[None, :]      # (1, neg_samples)
        self.W2[:, neg_idx] -= lr * (x[:, None].astype(np.float32) * grad_neg_f)

        # compute gradient for context embeddings
        # g = grad_pos * W2[:, target] + sum_i grad_neg[i]*W2[:, neg_idx[i]]
        g_pos = grad_pos * self.W2[:, target_idx]                          # (embedding_dim,)
        g_neg = np.sum(self.W2[:, neg_idx] * grad_neg_f, axis=1)           # (embedding_dim,)
        g = g_pos + g_neg

        # gradient clipping
        g = np.clip(g, -GRAD_CLIP, GRAD_CLIP).astype(np.float32)

        # average over context size
        denom = max(1, len(context_idxs))
        g_context = g / float(denom)

        # update W1 for each context index
        for ci in context_idxs:
            self.W1[ci] -= lr * g_context

        # clip weights to avoid runaway (prevent inf/nan)
        np.clip(self.W1, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W1)
        np.clip(self.W2, -WEIGHT_CLIP, WEIGHT_CLIP, out=self.W2)

    def most_similar(self, word, top_k=10):
        if word not in self.word2idx:
            return None
        wid = self.word2idx[word]
        vec = self.W1[wid].astype(np.float64)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        W1_norm = self.W1.astype(np.float64)
        W1_norm = W1_norm / (np.linalg.norm(W1_norm, axis=1, keepdims=True) + 1e-9)
        sims = W1_norm @ vec_norm
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(self.idx2word[i], float(sims[i])) for i in idxs]
    
    def get_embedding(self, word):
        """
        Devuelve el embedding de la palabra dada.
        Si la palabra no está en el vocabulario, devuelve None.
        """
        idx = self.word2idx.get(word)
        if idx is None:
            return None
        return self.W1[idx].copy()  # devuelve un array float32


# --------------------------
# Tokenizers and streaming
# --------------------------
def tokenize_line(line: str):
    # minimal cleaning: keep unicode word characters and whitespace
    line = re.sub(r"[^\w\s]", "", line.lower())
    toks = line.split()
    return toks

def tokenize_stream_from_dataset(dataset):
    for line in dataset["text"]:
        if not line.strip():
            continue
        for tok in tokenize_line(line):
            yield tok

# --------------------------
# Build / load vocab, counts, unigram
# --------------------------
dataset = load_dataset("text", data_files=DATA_FILE, split="train")

# safe load if exists
vocab_list = None
counts_dict = None
unigram_table = None

# Try load safely; if any fail, regenerate
loaded_ok = True
vocab_arr = safe_load_npy(Vocab_file)
counts_obj = safe_load_npy(counts_file)
unigram_arr = safe_load_npy(unigram_file)

if vocab_arr is None or counts_obj is None or unigram_arr is None:
    loaded_ok = False

if loaded_ok:
    try:
        vocab_list = vocab_arr.tolist()
        counts_dict = counts_obj.item() if hasattr(counts_obj, 'item') else counts_obj.tolist()
        unigram_table = np.array(unigram_arr, dtype=np.int32)
        print("Cargado vocab, counts y unigram table desde disco.")
    except Exception:
        print("[WARN] Falla al convertir archivos guardados a estructuras. Regenerando...")
        loaded_ok = False

if not loaded_ok:
    print("Construyendo vocab y conteos (primera vez o archivos corruptos). Esto puede tardar.")
    counter = Counter()
    for tok in tqdm(tokenize_stream_from_dataset(dataset), desc="Contando tokens"):
        counter[tok] += 1
    most = counter.most_common(MAX_VOCAB)
    vocab_list = [w for w, _ in most]
    counts_dict = {w: c for w, c in most}
    # save safely
    safe_save_npy(Vocab_file, np.array(vocab_list))
    safe_save_npy(counts_file, np.array(counts_dict, dtype=object))
    # build unigram
    freqs = np.array([counts_dict[w] for w in vocab_list], dtype=np.float64)
    probs = freqs ** 0.75
    probs /= probs.sum()
    rng = np.random.default_rng(SEED)
    unigram_table = rng.choice(len(vocab_list), size=UNIGRAM_TABLE_SIZE, p=probs).astype(np.int32)
    safe_save_npy(unigram_file, unigram_table)
    print(f"Vocab final: {len(vocab_list)} (saved)")

# --------------------------
# Total tokens estimate (for lr decay)
# --------------------------
total_tokens = None
tot_loaded = safe_load_npy(total_tokens_file)
if tot_loaded is not None:
    try:
        total_tokens = int(tot_loaded)
        print("Total tokens cargados desde archivo:", total_tokens)
    except Exception:
        total_tokens = None

if total_tokens is None:
    print("Contando tokens totales para lr-decay (esto es otra pasada, puede tardar)...")
    total_tokens = 0
    for line in tqdm(dataset["text"], desc="Contando tokens totales"):
        toks = tokenize_line(line)
        total_tokens += len(toks)
    safe_save_npy(total_tokens_file, np.array(total_tokens))
    print("Tokens totales (aprox):", total_tokens)

# --------------------------
# Initialize model (or load weights)
# --------------------------
model = CBOWNegSampling(vocab_list, counts_dict, embedding_dim=EMBEDDING_DIM, seed=SEED, unigram_table=unigram_table)

# load weights if present and valid
W1_loaded = safe_load_npy(W1_file)
W2_loaded = safe_load_npy(W2_file)
processed_tokens = 0
proc_loaded = safe_load_npy(processed_file)
if proc_loaded is not None:
    try:
        processed_tokens = int(proc_loaded)
    except Exception:
        processed_tokens = 0

if W1_loaded is not None and W2_loaded is not None:
    try:
        # validate shapes
        if W1_loaded.shape == model.W1.shape and W2_loaded.shape == model.W2.shape:
            model.W1 = W1_loaded.astype(np.float32)
            model.W2 = W2_loaded.astype(np.float32)
            print("Pesos W1/W2 cargados correctamente. Procesados previamente:", processed_tokens)
        else:
            print("[WARN] Pesos guardados tienen forma distinta. Ignorando y re-inicializando.")
    except Exception:
        print("[WARN] Error cargando pesos guardados. Ignorando y continuando.")
else:
    print("No hay pesos guardados, empezando desde 0.")

# --------------------------
# Training loop with checkpoints & resume
# --------------------------
def train_streaming(epochs=1, window=2, initial_lr=INITIAL_LR, min_lr=MIN_LR,
                    neg_samples=NEG_SAMPLES, save_every_tokens=CHECKPOINT_TOKENS):
    total_tokens_est = total_tokens * epochs
    processed = int(processed_tokens)

    # use deque as buffer of recent token indices; store -1 for unknowns
    buff = deque(maxlen=BUFFER_MAX)

    for epoch in range(epochs):
        print(f"\n=== ÉPOCA {epoch} ===")
        # iterate lines
        for line in tqdm(dataset["text"], desc=f"Epoch {epoch} lines"):
            toks = tokenize_line(line)
            for tok in toks:
                idx = model.word2idx.get(tok, -1)
                buff.append(idx)

                if idx == -1:
                    continue

                # build context from previous tokens only (streaming)
                context = []
                for j in range(1, window + 1):
                    if len(buff) > j and buff[-1 - j] != -1:
                        context.append(buff[-1 - j])

                if not context:
                    continue

                # update processed count and learning rate
                processed += 1
                progress = processed / max(1, total_tokens_est)
                lr = initial_lr * (1.0 - progress)
                if lr < min_lr:
                    lr = min_lr

                # train step
                model.train_step(context, idx, lr, neg_samples)

                # periodic checkpoints
                if processed % save_every_tokens == 0:
                    safe_save_npy(W1_file, model.W1)
                    safe_save_npy(W2_file, model.W2)
                    safe_save_npy(processed_file, np.array(processed))
                    print(f"[Checkpoint] guardados pesos tras {processed} tokens (lr={lr:.6f})")

        # end of epoch checkpoint
        safe_save_npy(W1_file, model.W1)
        safe_save_npy(W2_file, model.W2)
        safe_save_npy(processed_file, np.array(processed))
        print(f"Fin época {epoch}. Pesos guardados. tokens procesados hasta ahora: {processed}")

    return processed




# --------------------------
# Run training
# --------------------------
if __name__ == "__main__":
    # small safety check before long runs
    print("Configuración:")
    print(f"  DATA_FILE = {DATA_FILE}")
    print(f"  MAX_VOCAB = {MAX_VOCAB}, EMBEDDING_DIM = {EMBEDDING_DIM}")
    print(f"  UNIGRAM_TABLE_SIZE = {UNIGRAM_TABLE_SIZE}, BUFFER_MAX = {BUFFER_MAX}")
    print(f"  INITIAL_LR = {INITIAL_LR}, MIN_LR = {MIN_LR}, NEG_SAMPLES = {NEG_SAMPLES}")
    print("Iniciando entrenamiento... (presiona Ctrl+C para parar y mantener checkpoint)")

    # choose epochs as desired
    processed_final = train_streaming(epochs=2, window=2,
                                      initial_lr=INITIAL_LR, min_lr=MIN_LR,
                                      neg_samples=NEG_SAMPLES,
                                      save_every_tokens=CHECKPOINT_TOKENS)

    print("Entrenamiento finalizado. Tokens procesados:", processed_final)
    print("\nEjemplo most_similar('gente'):")
    print(model.most_similar("gente", top_k=10))
