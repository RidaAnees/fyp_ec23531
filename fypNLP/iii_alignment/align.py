
"""
Unified cross-lingual alignment with multiple methods:
  - 'procrustes': vanilla orthogonal Procrustes (supervised)
  - 'vecmap_supervised': VecMap supervised (Procrustes + iterative refine)
  - 'vecmap_semi': VecMap semi-supervised (small seed + self-learning)
  - 'vecmap_unsupervised':VecMap unsupervised (no seed dict)
Before alignment, the script measures isomorphism between source and target embedding spaces using:
  - RSA (Representational Similarity Analysis) on seed-dict word pairs
  - Eigenvalue similarity on top-k singular values
  - Hubness skewness comparison
"""
import os
import random
import sys
import tempfile
import time
import importlib.util
import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec, KeyedVectors
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr, skew

# Preprocessing
def preprocess_embeddings(vectors, method='center_whiten_renorm'):
    VALID = {'none', 'norm_only', 'center','center_whiten', 'center_whiten_renorm'}
    if method not in VALID:
        raise ValueError(
            f"embedding_preprocessing must be one of {VALID}, got '{method}'"
        )
    vectors = np.array(vectors, dtype=np.float32) #always work on a fresh float32 copy so it never mutates a memory-mapped array
    if method == 'none':
        return vectors
    #Step 1: unit-normalise rows
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-8)
    if method == 'norm_only':
        return vectors
    # Step 2: mean-centre 
    vectors = vectors - vectors.mean(axis=0, keepdims=True)
    if method == 'center':
        return vectors
    # Step 3: ZCA whitening 
    cov = np.cov(vectors.T)                              
    U, S, _ = np.linalg.svd(cov)
    W_zca = U @ np.diag(1.0 / np.sqrt(S + 1e-6)) @ U.T  
    vectors = vectors @ W_zca.T
    if method == 'center_whiten':
        return vectors
    # Step 4: re-normalise rows to unit sphere
    # (method== 'center_whiten_renorm')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-8)
    return vectors

def _apply_preprocessing(kv: KeyedVectors, method: str, label: str) -> KeyedVectors:
    """
    Preprocess a KeyedVectors object in-place (returns the same object).
    Handles the mmap='r' read-only case by always working on a fresh copy (preprocess_embeddings guarantees this)
    After assignment the vectors attribute is a writable float32 ndarray regardless of how kv was loaded.
    """
    if method== 'none':
        _step( f"Preprocessing skipped for {label} (method='none')")
        return kv
    _step (f"Preprocessing {label} with method='{method}' …" )
    processed = preprocess_embeddings(kv.vectors, method)   # always a fresh copy
    kv.vectors= processed
    _step (f"  ✓ {label}: shape={kv.vectors.shape}, "
          f"sample norm={np.linalg.norm(kv.vectors[0]):.4f}" )
    return kv

#helpers
def _stage(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"\n  [{ts}] -> {msg}", flush=True)
def _step(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

class _Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.time()
        _stage(f"{self.label} …")
        return self
    def __exit__(self, *exc):
        elapsed = time.time() - self.t0
        _step(f"✓ {self.label} done in {elapsed:.1f}s")

# measure Isomorphism 
def _rsa(src_kv, tgt_kv, word_pairs):
    """ RSA two embedding spaces.
    Returns pearson & spearman corr on pairwise cos distances  """
    src_words, tgt_words= zip(*word_pairs)
    src_vecs = np.array([src_kv[w] for w in src_words], dtype=np.float32)
    tgt_vecs= np.array([tgt_kv[w] for w in tgt_words], dtype=np.float32)
    src_vecs/= np.linalg.norm (src_vecs, axis=1, keepdims=True) + 1e-9
    tgt_vecs/= np.linalg.norm (tgt_vecs, axis=1, keepdims=True) + 1e-9
    src_dists= pdist  (src_vecs, metric='cosine')
    tgt_dists= pdist (tgt_vecs, metric='cosine' )
    pearson_r, _= pearsonr ( src_dists, tgt_dists )
    spearman_r, _= spearmanr (src_dists, tgt_dists )
    return {
        'rsa_n_pairs':len(word_pairs),
        'rsa_pearson': round(float(pearson_r),  4 ),
        'rsa_spearman': round(float(spearman_r), 4) ,
    }

def _eigenvalue_similarity(src_kv, tgt_kv, n_words=10000, k=20):
    """
    Compare normalized singular values of the top-n words in each space 
    Lower distance = more similar variance distribution -> more isomorphic
    """
    n_src = min(n_words, len(src_kv))
    n_tgt = min(n_words, len(tgt_kv))
    src_words = list(src_kv.index_to_key)[:n_src]
    tgt_words = list(tgt_kv.index_to_key)[:n_tgt]
    src_mat = np.array([src_kv[w] for w in src_words], dtype=np.float32)
    tgt_mat = np.array([tgt_kv[w] for w in tgt_words], dtype=np.float32)
    _, s1, _ = np.linalg.svd(src_mat, full_matrices=False)
    _, s2, _ = np.linalg.svd(tgt_mat, full_matrices=False)
    k = min(k, len(s1), len(s2))
    s1 = s1[:k] / s1[:k].sum()
    s2 = s2[:k] / s2[:k].sum()
    return {
        'eigval_distance': round(float(np.sum((s1 - s2) ** 2)), 6),
        'eigval_k':        k,
        'eigval_n_words':  min(n_src, n_tgt),
    }

def _hubness_compare(src_kv, tgt_kv, n_words=5000, k=10):
    """
    Compare hubness profiles of both spaces
    Similar skewness -> similar hubness
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        _step("scikit-learn not installed — skipping hubness comparison")
        return {}
    def _skew(kv, n):
        n = min(n, len(kv))
        words= list(kv.index_to_key)[:n]
        mat= np.array([kv[w] for w in words], dtype=np.float32)
        nn= NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(mat)
        _, idx = nn.kneighbors(mat)
        idx = idx[:, 1:] # excludes self
        counts = np.bincount(idx.flatten(), minlength=len(mat))
        return float(skew(counts))
    return {
        'src_hubness_skew': round(_skew(src_kv, n_words), 4 ),
        'tgt_hubness_skew': round(_skew(tgt_kv, n_words), 4 ),
    }

def _isomorphism_report(src_kv, tgt_kv, seed_pairs):
    """
    Run isomorphism measures BEFORE alignment, print a summary, and return a dict of results to merge into the eval log.
    """
    print(f"\n  ########################################################## ")
    print(f"  PRE-ALIGNMENT ISOMORPHISM MEASUREMENT")
    print(f" ##########################################################")
    valid_pairs= [(s, t) for s,t in seed_pairs
                   if s in src_kv.key_to_index and t in tgt_kv.key_to_index]
    results = {}
    #1. RSA
    if len(valid_pairs)>= 10:
        with _Timer(f"RSA on {len(valid_pairs)} seed pairs"):
            rsa_results = _rsa(src_kv, tgt_kv, valid_pairs)
            results.update(rsa_results)
            _step(f"RSA Pearson r : {rsa_results['rsa_pearson']} " )
            _step(f"RSA Spearman r : {rsa_results['rsa_spearman'] }" )
    else:
        _step(f"Skipping RSA, need more than 10 valid pairs, got {len(valid_pairs)} " )
    # 2.Eigenvalue similarity
    with _Timer("Eigenvalue similarity (top-20 singular values)" ):
        eig_results= _eigenvalue_similarity(src_kv, tgt_kv)
        results.update(eig_results)
        _step(f"Eigval distance: {eig_results['eigval_distance']}"
              f" (k={eig_results['eigval_k']}, n={eig_results['eigval_n_words']:,})" )
    # 3.Hubness
    with _Timer("Hubness skewness comparison"):
        hub_results = _hubness_compare(src_kv, tgt_kv )
        results.update(hub_results )
        if hub_results:
            _step (f"Source hubness skewness: {hub_results['src_hubness_skew']} ")
            _step (f"Target hubness skewness:{hub_results['tgt_hubness_skew']}" )
            _step (f"Δ skewness : "
                  f"{abs(hub_results['src_hubness_skew'] - hub_results['tgt_hubness_skew']):.4f}" )
    #Summary
    print(f"\n Isomorphism Summary ")
    if 'rsa_pearson' in results:
        rsa_p = results['rsa_pearson']
        if rsa_p >= 0.6:    rsa_lbl = "HIGH"
        elif rsa_p >= 0.3:  rsa_lbl = "MEDIUM"
        else:               rsa_lbl = "LOW"
        print(f"  RSA Pearson      : {rsa_p}  -> {rsa_lbl}")
    if 'eigval_distance' in results:
        ed = results['eigval_distance']
        if ed < 0.01:    ed_lbl = "HIGH"
        elif ed < 0.1:   ed_lbl = "MEDIUM"
        else:            ed_lbl = "LOW"
        print(f"  Eigval similarity: {ed}  -> {ed_lbl}")
    if 'src_hubness_skew' in results:
        d = abs(results['src_hubness_skew'] - results['tgt_hubness_skew'])
        if d < 1.0:    h_lbl = "HIGH"
        elif d < 3.0:  h_lbl = "MEDIUM"
        else:          h_lbl = "LOW"
        print(f"  Hubness match    : Δ={d:.4f}  -> {h_lbl}")
    print(f" ######################################################################\n")
    return results

# I/O helpers
def _kv_to_vec_file(kv: KeyedVectors, path: str) -> None:
    words = list(kv.index_to_key)
    n = len(words)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"{n} {kv.vector_size}\n")
        report_every = max(1, n // 10)
        for i, w in enumerate(words):
            fh.write(f"{w} {' '.join(f'{v:.6f}' for v in kv[w])}\n")
            if (i + 1) % report_every == 0 or (i + 1) == n:
                pct = 100 * (i + 1) / n
                _step(f"  written {i+1:,}/{n:,} ({pct:.0f}%)")

def _vec_file_to_kv(path: str) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(path, binary=False)

def _write_seed_file(pairs, path):
    with open(path, 'w', encoding='utf-8') as fh:
        for src, tgt in pairs:
            fh.write(f"{src}\t{tgt}\n")

# Seed dictionary
def _load_pairs(dict_path, src, tgt, use_translit=False):
    df = pd.read_csv(dict_path,
                     names=['english', 'transliteration', 'dhivehi'],
                     header=None,
                     encoding='utf-8')
    #drop the header row
    if df.iloc[0]['english'] == 'english':
        df = df.iloc[1:].reset_index(drop=True)
        _step("Detected and dropped CSV header row" )
    pairs, miss_s, miss_t = [], [], []
    for _, row in df.iterrows():
        eng = str(row['english']).strip().lower()
        s= (str(row['transliteration']).strip().lower()
               if use_translit else str(row['dhivehi']).strip())
        if s in src.key_to_index and eng in tgt.key_to_index:
            pairs.append((s, eng))
        else:
            if s   not in src.key_to_index: miss_s.append(s)
            if eng not in tgt.key_to_index: miss_t.append(eng)
    mode = 'transliteration' if use_translit else 'Thaana script'
    _step(f"Valid seed pairs   : {len(pairs):,}  (source: {mode})")
    if miss_s: _step(f"Missing in Dhivehi : {len(miss_s)}  e.g. {miss_s[:3]}" )
    if miss_t: _step(f"Missing in English : {len(miss_t)}  e.g. {miss_t[:3]}" )
    return pairs

def _split_pairs(pairs, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_train = max(10, int(len(shuffled) * train_ratio))
    return shuffled[:n_train], shuffled[n_train:]

#Procrustes
def _align_procrustes(src, tgt, train_pairs, mode='orthogonal'):
    """
    Procrustes alignment with two modes:
      'orthogonal' : W constrained to be orthogonal (preserves angles/distances). Solved via SVD: W = U @ V.T  where  X.T @ Y = U S V.T
      'linear' : unconstrained least-squares (any linear map). Solved via pseudo-inverse: W = pinv(X) @ Y
    """
    if mode not in ('orthogonal', 'linear'):
        raise ValueError(
            f"procrustes_mode must be 'orthogonal' or 'linear', got'{mode}'"
        )
    with _Timer(f"Procrustes ({mode}) on {len(train_pairs)} pairs"):
        X = np.array([src[s] for s, _ in train_pairs], dtype=np.float32 )
        Y = np.array([tgt[t] for _, t in train_pairs], dtype=np.float32 )
        _step("Normalizing seed vectors …")
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
        if mode == 'orthogonal':
            _step(f"Solving orthogonal Procrustes (SVD on {X.shape[1]}×{X.shape[1]}) …")
            W, _ = orthogonal_procrustes(X, Y)
        else:
            _step("Solving linear least-squares (unconstrained) …")
            W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        train_sim = float(np.mean(np.sum((X @ W) * Y, axis=1)))
        _step(f"Avg train cosine   : {train_sim:.4f}")
        src_words  = list(src.index_to_key)
        n          = len(src_words)
        _step(f"Applying mapping to {n:,} Dhivehi vectors …")
        src_mat    = np.array([src[w] for w in src_words], dtype=np.float32)
        src_mat   /= np.linalg.norm(src_mat, axis=1, keepdims=True) + 1e-9
        src_mapped_mat = src_mat @ W
        src_mapped = KeyedVectors(vector_size=src.vector_size)
        src_mapped.add_vectors(src_words, src_mapped_mat)
    return src_mapped, tgt

# VecMap
def _run_vecmap(vecmap_dir, mode, src_vec, tgt_vec, src_out, tgt_out, seed_file, use_cuda):
    vecmap_abs = os.path.abspath(vecmap_dir)
    map_script = os.path.join(vecmap_abs, 'map_embeddings.py')
    if not os.path.exists(map_script):
        raise FileNotFoundError(
            f"VecMap not found at {map_script}\n"
            f"Install with:  git clone https://github.com/artetxem/vecmap {vecmap_dir}  :) "
        )
    if vecmap_abs not in sys.path:
        sys.path.insert(0, vecmap_abs )
    argv = ['map_embeddings.py']
    if mode == 'supervised':
        argv+= ['--supervised', seed_file]
    elif mode == 'semi':
        argv+= ['--semi_supervised', seed_file]
    elif mode == 'unsupervised':
        argv+= ['--unsupervised']
    else:
        raise ValueError(f"Unknown VecMap mode: {mode}" )
    argv += ['--verbose']
    if use_cuda:
        argv += ['--cuda']
    argv += ['--csls', '10']
    argv += [src_vec, tgt_vec, src_out, tgt_out]
    original_argv = sys.argv[:]
    sys.argv = argv
    _step(f"VecMap argv: {' '.join(argv)}")
    print("\n VecMap output (live)",
          flush=True)
    try:
        spec = importlib.util.spec_from_file_location('map_embeddings', map_script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"VecMap exited with code {e.code}")
    finally:
        sys.argv = original_argv
        sys.stdout.flush()
        print("----------------------------------------------------------------------------\n",
              flush=True)

def _align_vecmap(src, tgt, train_pairs, mode, vecmap_dir, use_cuda):
    with tempfile.TemporaryDirectory() as tmp:
        src_vec= os.path.join(tmp, 'src.vec')
        tgt_vec = os.path.join(tmp, 'tgt.vec')
        seed_file= os.path.join(tmp, 'seed.txt')
        src_out = os.path.join(tmp, 'src_mapped.vec')
        tgt_out = os.path.join(tmp, 'tgt_mapped.vec')
        with _Timer(f"Exporting source ({len(src):,} vectors) to .vec"):
            _kv_to_vec_file(src, src_vec)
        with _Timer(f"Exporting target ({len(tgt):,} vectors) to .vec"):
            _kv_to_vec_file(tgt, tgt_vec)
        if mode != 'unsupervised':
            _step(f"Writing {len(train_pairs)} seed pairs")
            _write_seed_file(train_pairs, seed_file)
            seed_arg= seed_file
        else:
            seed_arg = None
            _step("Unsupervised mode:    no seed dictionary used " )
        with _Timer(f"VecMap mapping ({mode}, cuda={use_cuda})"):
            _run_vecmap(vecmap_dir, mode, src_vec, tgt_vec, src_out, tgt_out,
                        seed_arg, use_cuda)
        with _Timer("Loading mapped vectors back into KeyedVectors"):
            src_mapped = _vec_file_to_kv(src_out)
            tgt_mapped= _vec_file_to_kv(tgt_out )
            _step(f"Mapped src vocab : {len(src_mapped):,}")
            _step(f"Mapped tgt vocab : {len(tgt_mapped):,}")
    return src_mapped, tgt_mapped

# Evaluation
def _build_csls_cache(tgt_mat_n, k=10):
    n = tgt_mat_n.shape[0]
    r_t = np.zeros(n, dtype=np.float32)
    #Shrink batch to avoid OOM — 500 rows × 205k cols × 4 bytes ≈ 390 MB
    batch = 500
    n_batches = (n + batch - 1)// batch
    
    for b_idx, start in enumerate(range(0, n, batch)):
        end  = min(start + batch, n)
        sims = tgt_mat_n[start:end] @ tgt_mat_n.T # (batch, n)
        
        # Zero out self-similarity
        for i in range(end - start):
            sims[i, start + i] = 0.0
        
        # np.partition is still OOM-safe here because sims is already (batch, n) so only keeping top-k values, not the full sorted array
        top_k = np.partition(sims, -k, axis=1)[:, -k:]
        r_t[start:end] = np.mean(top_k, axis=1)
        
        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
            _step(f"  CSLS batch {b_idx+1}/{n_batches}")
    return r_t

def _evaluate_held_out(src_mapped, tgt_mapped, test_pairs, topn_list=(1, 5, 10)):
    if not test_pairs:
        return {}, None, None
    with _Timer(f"Evaluation on {len(test_pairs)} held-out pairs"):
        en_words  = list(tgt_mapped.index_to_key)
        en_mat = np.array([tgt_mapped[w] for w in en_words], dtype=np.float32)
        en_mat_n = en_mat / (np.linalg.norm(en_mat, axis=1, keepdims=True) + 1e-9)
        en_idx = {w: i for i, w in enumerate(en_words)}
        _step(f"Building CSLS cache for {len(en_words):,} English words …")
        r_t = _build_csls_cache(en_mat_n)
        max_topn= max(topn_list)
        hits_cos = {k: 0 for k in topn_list}
        hits_csls = {k: 0 for k in topn_list}
        cosines = []
        n_evaluated = 0
        _step("Scoring test pairs …")
        for dv, en_gold in test_pairs:
            if dv not in src_mapped.key_to_index or en_gold not in en_idx:
                continue
            proj   = src_mapped[dv].astype(np.float32)
            proj_n = proj / (np.linalg.norm(proj) + 1e-9)
            cos_sims    = en_mat_n @ proj_n
            csls_scores = 2.0 * cos_sims - r_t
            top_cos  = [en_words[j]
                        for j in np.argsort(cos_sims)[-max_topn:][::-1]]
            top_csls = [en_words[j]
                        for j in np.argsort(csls_scores)[-max_topn:][::-1]]
            cosines.append(float(cos_sims[en_idx[en_gold]]))
            for k in topn_list:
                if en_gold in top_cos[:k]:  hits_cos[k]  += 1
                if en_gold in top_csls[:k]: hits_csls[k] += 1
            n_evaluated += 1
    results = {}
    for k in topn_list:
        results[f'cosine_P@{k}']= (round(hits_cos[k]  / n_evaluated, 4)
                                    if n_evaluated else 0)
        results[f'csls_P@{k}']= (round(hits_csls[k] / n_evaluated, 4)
                                    if n_evaluated else 0)
    results['mean_cosine_gold']= round(float(np.mean(cosines)), 4) if cosines else 0
    results['n_test_pairs'] = n_evaluated
    print(f"\n Held-out Evaluation ({n_evaluated} pairs)")
    print(f"{'Metric':<10} {'Cosine':>8} {'CSLS':>8} {'Δ':>8}")
    print(f"{'-'*40}")
    for k in topn_list:
        c, s = results[f'cosine_P@{k}'], results[f'csls_P@{k}']
        print(f"  P@{k:<7d} {c:>8.4f} {s:>8.4f} {s-c:>+8.4f}")
    print(f"  Mean cos to gold : {results['mean_cosine_gold']:.4f}")
    return results, en_mat_n, r_t

def _print_examples(src_mapped, tgt_mapped, test_pairs, n=10,en_mat_n=None, r_t=None):
    """Print qualitative retrieval examples """
    en_words = list(tgt_mapped.index_to_key)
    if en_mat_n is None:
        en_mat   = np.array([tgt_mapped[w] for w in en_words], dtype=np.float32)
        en_mat_n = en_mat / (np.linalg.norm(en_mat, axis=1, keepdims=True) + 1e-9)
    if r_t is None:
        r_t = _build_csls_cache(en_mat_n)
    print("\n Qualitative examples (held-out)")
    for dv, en_gold in test_pairs[:n]:
        if dv not in src_mapped.key_to_index:
            print(f"\n  '{dv}' — not in mapped vocabulary")
            continue
        proj = src_mapped[dv].astype(np.float32)
        proj_n = proj / (np.linalg.norm(proj) + 1e-9)
        cos_sims = en_mat_n @ proj_n
        csls = 2.0 * cos_sims - r_t
        top_cos = [en_words[i] for i in np.argsort(cos_sims)[-5:][::-1]]
        top_csls = [en_words[i] for i in np.argsort(csls)[-5:][::-1]]
        print(f"\n  '{dv}'  (gold: {en_gold})")
        print(f"Cosine {'✓' if en_gold in top_cos  else '✗'} : {top_cos}" )
        print(f"CSLS   {'✓' if en_gold in top_csls else '✗'} : {top_csls}" )

# Save aligned space
def _create_aligned_space(src_mapped, tgt_mapped, output_path, train_pairs=None, test_pairs=None):
    with _Timer("Building combined aligned space"):
        os.makedirs(os.path.dirname(output_path) or '.',exist_ok=True)
        src_words= list(src_mapped.index_to_key)
        tgt_words = list(tgt_mapped.index_to_key )
        _step(f"Stacking {len(src_words):,} dv + {len(tgt_words):, } en vectors" )
        src_mat = np.array([src_mapped[w] for w in src_words], dtype=np.float32)
        tgt_mat= np.array([tgt_mapped[w] for w in tgt_words], dtype=np.float32  )
        all_words= ([f"dv_{w}" for w in src_words] +  [f"en_{w}" for w in tgt_words])
        all_vecs= np.vstack([src_mat, tgt_mat])
        kv = KeyedVectors(vector_size=all_vecs.shape[1])
        kv.add_vectors(all_words, all_vecs)
        kv.save(output_path)
        _step(f"Aligned space saved -> {output_path}")
        _step(f"Dhivehi: {len(src_words):,}  English: {len(tgt_words):,}")
        if train_pairs is not None or test_pairs is not None:
            all_seed_en = set()
            if train_pairs: all_seed_en |= {e for _, e in train_pairs}
            if test_pairs:  all_seed_en |= {e for _, e in test_pairs}
            seed_path = output_path.replace('.kv', '_seed_words.txt')
            with open(seed_path, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(sorted(all_seed_en)))
            _step(f"Seed list  -> {seed_path}  ({len(all_seed_en)} words)")

# Pipeline entry point
def run_alignment(cfg: dict) -> None:
    method = cfg.get('alignment_method', 'procrustes').lower()
    valid_methods = {'procrustes', 'vecmap_supervised','vecmap_semi', 'vecmap_unsupervised'}
    if method not in valid_methods:
        raise ValueError(
            f"alignment_method must be one of {valid_methods}, got '{method}'"
        )
    overall_start = time.time()
    print(f"\n  ALIGNMENT METHOD: {method.upper()}")
    print(f" -------------------------------------------------------------------")
    #Load the embeddings
    with _Timer(f"Loading source (Dhivehi): {cfg['source_embedding']}"):
        src_path = cfg['source_embedding']
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source embedding not found: {src_path}")
        if src_path.endswith('.kv'):
            # Load without mmap so vectors are writable from the start
            src_kv = KeyedVectors.load(src_path)
        elif src_path.endswith('.model'):
            try:
                model = FastText.load(src_path)
            except Exception:
                model = Word2Vec.load(src_path)
            src_kv = model.wv
        elif src_path.endswith('.vec'):
            src_kv = KeyedVectors.load_word2vec_format(src_path, binary=False)
        else:
            raise ValueError(f"Unknown embedding format: {src_path}")
        _step(f"Source vocab : {len(src_kv):,}  dim {src_kv.vector_size}")
    with _Timer(f"Loading target (English): {cfg['target_embedding']}"):
        tgt_path = cfg['target_embedding']
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(
                f"Target embedding not found: {tgt_path}\n"
                f"If you need to create a .kv file from a .vec file:\n"
                f" from gensim.models import KeyedVectors\n"
                f"  kv = KeyedVectors.load_word2vec_format('your_file.vec', binary=False)\n"
                f"kv.save('{tgt_path}')"
            )
        if tgt_path.endswith('.kv'):
            #Load without mmap so vectors are writable from the start
            tgt_kv = KeyedVectors.load(tgt_path)
            _step(f"Loaded .kv: {len(tgt_kv.key_to_index):,} words, "
                  f"dim {tgt_kv.vector_size}")
        elif tgt_path.endswith('.model'):
            try:
                model = FastText.load(tgt_path)
            except Exception:
                model = Word2Vec.load(tgt_path)
            tgt_kv = model.wv
            _step(f"Loaded .model: {len(tgt_kv.key_to_index):,} words, "
                  f"dim {tgt_kv.vector_size}")
        elif tgt_path.endswith('.vec'):
            tgt_kv = KeyedVectors.load_word2vec_format(tgt_path, binary=False)
            _step(f"Loaded .vec: {len(tgt_kv.key_to_index):,} words, "
                  f"dim {tgt_kv.vector_size}")
        else:
            raise ValueError(
                f"Unknown embedding format: {tgt_path}. Use .kv, .model, or .vec"
            )
    # Embedding preprocessing
    preproc_method = cfg.get('embedding_preprocessing', 'none')
    with _Timer(f"Preprocessing embeddings (method='{preproc_method}')"):
        src_kv = _apply_preprocessing(src_kv, preproc_method, label='Dhivehi')
        tgt_kv = _apply_preprocessing(tgt_kv, preproc_method, label='English' )
    # Seed dictionary
    with _Timer("Loading and filtering seed dictionary"):
        all_pairs = _load_pairs(
            cfg['seed_dictionary'], src_kv, tgt_kv,
            use_translit=cfg.get('use_transliteration', False),
        )
        if not all_pairs and method != 'vecmap_unsupervised':
            raise ValueError("No valid seed pairs found.")
        train_pairs, test_pairs = _split_pairs(
            all_pairs,
            train_ratio=cfg.get('alignment_train_ratio', 0.8),
            seed=cfg.get('alignment_seed', 42),
        )
        _step(f"Train pairs  : {len(train_pairs):,}")
        _step(f"Test pairs   : {len(test_pairs):,}  ← never seen during alignment")
    #Pre-alignment isomorphism measurement
    if cfg.get('measure_isomorphism', True):
        iso_results = _isomorphism_report(src_kv, tgt_kv, all_pairs)
    else:
        iso_results = {}
    #Run ze alignment
    if method == 'procrustes':
        src_mapped, tgt_mapped = _align_procrustes(
            src_kv, tgt_kv, train_pairs,
            mode=cfg.get('procrustes_mode', 'orthogonal'),
        )
    else:
        vecmap_mode = {
            'vecmap_supervised':   'supervised',
            'vecmap_semi':         'semi',
            'vecmap_unsupervised': 'unsupervised',
        }[method]
        src_mapped, tgt_mapped = _align_vecmap(
            src_kv, tgt_kv, train_pairs,
            mode       = vecmap_mode,
            vecmap_dir = cfg.get('vecmap_dir', 'vecmap'),
            use_cuda   = cfg.get('alignment_use_cuda', False),
        )
    # evaluate
    eval_results, en_mat_n, r_t = _evaluate_held_out(
        src_mapped, tgt_mapped, test_pairs, topn_list=(1, 5, 10)
    )
    _print_examples(src_mapped, tgt_mapped, test_pairs, n=10,
                    en_mat_n=en_mat_n, r_t=r_t)
    # saves the eval log
    # Use alignment_output_prefix if provided, else fall back to alignment_matrix
    output_prefix = cfg.get('alignment_output_prefix', cfg.get('alignment_matrix', 'alignment'))
    eval_path = output_prefix.replace('.npy', '') + f'_{method}_eval.txt'
    os.makedirs (os.path.dirname(eval_path) or '.', exist_ok=True)
    with open (eval_path, 'w', encoding='utf-8') as fh:
        fh.write (f"Alignment method : {method}\n")
        fh.write(f"Source embedding : {cfg['source_embedding']}\n")
        fh.write (f"Target embedding : {cfg['target_embedding']}\n")
        fh.write( f"Embedding preprocessing : {preproc_method}\n")
        fh.write (f"Train pairs : {len(train_pairs)}\n")
        fh.write (f"Test  pairs : {len(test_pairs)}\n")
        fh.write("=" * 50 + "\n")
        if iso_results:
            fh.write("ISOMORPHISM (pre-alignment):\n")
            for k, v in iso_results.items():
                fh.write(f"  {k}: {v}\n")
            fh.write("=" * 50 + "\n")
        fh.write("ALIGNMENT RESULTS:\n")
        for k, v in eval_results.items():
            fh.write(f"  {k}: {v}\n")
    _step(f"Eval log -> {eval_path}")
    #saved the mapped + combined
    with _Timer("Saving mapped vector files"):
        mapped_dir = os.path.dirname(output_prefix) or '.'
        src_save = os.path.join(mapped_dir, f'dhivehi_{method}.kv')
        tgt_save = os.path.join(mapped_dir, f'english_{method}.kv')
        src_mapped.save(src_save); _step(f"Mapped src -> {src_save}")
        tgt_mapped.save(tgt_save); _step(f"Mapped tgt -> {tgt_save}")
    _create_aligned_space(src_mapped, tgt_mapped,cfg['aligned_space_output'],train_pairs=train_pairs, test_pairs=test_pairs)
    total = time.time() - overall_start
    print(f"\n ---------------------------------------------------------------------------------")
    print(f"  ALIGNMENT COMPLETE in {total:.1f}s ({total/60:.1f} min)")
    print(f"------------------------------------------------------------------------------------")
