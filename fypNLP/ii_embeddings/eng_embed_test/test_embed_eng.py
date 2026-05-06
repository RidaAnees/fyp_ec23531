
"""
pipeline/ii_embeddings/test/test_embed_eng.py
Interactive or evaluation-style testing of the trained English embedding.
Modes (set 'test_mode_eng' in config)
"""
import os
import csv
import gensim
import numpy as np
from datetime import datetime
from gensim.models import FastText, Word2Vec

# model load
def _load_model(model_path: str, embedding_type: str):
    etype = embedding_type.lower()
    if etype == 'fasttext':
        return FastText.load(model_path)
    elif etype == 'word2vec':
        return Word2Vec.load(model_path)
    raise ValueError(f"Unknown embedding_type: '{embedding_type}'. Use 'fasttext' or 'word2vec'.")

#  Gensim bundled file paths 
def _gensim_test_path(filename: str) -> str:
    path = os.path.join(os.path.dirname(gensim.__file__),
                        'test', 'test_data', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gensim bundled test file not found: {path}\n"
            "Try reinstalling gensim: pip install gensim"
        )
    return path

#  Interactive mode
def _explore(wv, word: str, topn: int = 10) -> None:
    in_vocab = word in wv
    tag      = 'IN VOCAB' if in_vocab else 'OOV (fasttext subword)'
    print(f"\n  Word: '{word}'  [{tag}]")
    try:
        vec  = wv[word]
        norm = float(np.linalg.norm(vec))
        print(f"  Vector norm: {norm:.4f}")
        similar = wv.most_similar(word, topn=topn)
        print(f"  Top {topn} neighbours:")
        for rank, (w, score) in enumerate(similar, 1):
            print(f"    {rank:2d}. {w:<30s} {score:.4f}")
    except KeyError as exc:
        print(f"  Cannot find vector: {exc}")

def _interactive_mode(wv) -> None:
    print("\n" + "─" * 60)
    print("  INTERACTIVE MODE  |  type a word, or 'quit' to exit")
    print("─" * 60)
    while True:
        word = input("\n  > ").strip()
        if not word:
            continue
        if word.lower() in {'quit', 'exit', 'q'}:
            break
        _explore(wv, word)

#  Task 1: Similarity (WordSim-353 via gensim) 
def task_similarity(wv) -> dict:
    wordsim_path = _gensim_test_path('wordsim353.tsv')
    pearson, spearman, oov = wv.evaluate_word_pairs(wordsim_path)
    return {
        'pearson_r':   round(pearson[0],   4),
        'pearson_p':   round(pearson[1],   4),
        'spearman_r':  round(spearman[0],  4),
        'spearman_p':  round(spearman[1],  4),
        'oov_ratio':   round(oov,          4),
    }

#  Task 2: Analogy (Google Analogy via gensim) 
def task_analogy(wv) -> dict:
    questions_path = _gensim_test_path('questions-words.txt')
    score, sections = wv.evaluate_word_analogies(questions_path)
    section_scores = []
    for s in sections:
        correct = len(s['correct'])
        total   = correct + len(s['incorrect'])
        section_scores.append({
            'section':  s['section'],
            'correct':  correct,
            'total':    total,
            'accuracy': round(correct / total, 4) if total else None,
        })
    return {
        'overall_accuracy': round(score, 4),
        'sections':         section_scores,
    }

# task 3:coherence (probe words, centroid avg cosine sim )
def _coherence_score(wv, word: str, topn: int = 10) -> float | None:
    try:
        neighbours = wv.most_similar(word, topn=topn)
    except KeyError:
        return None
    vecs = []
    for w, _ in neighbours:
        try:
            vecs.append(wv[w])
        except KeyError:
            pass
    if len(vecs) < 2:
        return None
    vecs     = np.array(vecs)
    centroid = vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-10
    sims = []
    for v in vecs:
        norm = np.linalg.norm(v)
        if norm > 0:
            sims.append(float(np.dot(v / norm, centroid)))
    return float(np.mean(sims)) if sims else None

def task_coherence(wv, probe_words: list[str], topn: int = 10) -> dict:
    scores, skipped = [], []
    for word in probe_words:
        s = _coherence_score(wv, word, topn=topn)
        if s is None:
            skipped.append(word)
        else:
            scores.append((word, round(s, 4)))
    mean_coherence = round(float(np.mean([s for _, s in scores])), 4) if scores else None
    return {
        'scored':         len(scores),
        'skipped':        len(skipped),
        'mean_coherence': mean_coherence,
        'details':        scores,
        'skipped_words':  skipped,
    }

# Output helpers 
def _print_results(vocab: int, dims: int,
                   sim: dict, ana: dict, coh: dict) -> None:
    print(f"\n  {'='*55}")
    print(f"  Model    : trained_eng")
    print(f"  Vocab    : {vocab:,}   Dims: {dims}")
    print(f"  {'─'*55}")
    print(f"  [1] Similarity — WordSim-353")
    print(f"      Pearson r  : {sim['pearson_r']}  (p={sim['pearson_p']})")
    print(f"      Spearman r : {sim['spearman_r']}  (p={sim['spearman_p']})")
    print(f"      OOV ratio  : {sim['oov_ratio']}%")
    print(f"  [2] Analogy — Google Analogy")
    acc = f"{ana['overall_accuracy']*100:.2f}%"
    print(f"      Overall accuracy : {acc}")
    for s in ana['sections']:
        if s['total']:
            sec_acc = f"{s['accuracy']*100:.1f}%" if s['accuracy'] is not None else 'n/a'
            print(f"      {s['section']:<45s} {s['correct']}/{s['total']} ({sec_acc})")
    print(f"  [3] Coherence — probe words")
    print(f"      scored={coh['scored']}  skipped={coh['skipped']}")
    print(f"      mean coherence : {coh['mean_coherence']}")
    if coh['skipped_words']:
        print(f"      skipped        : {coh['skipped_words']}")
    print(f"  {'='*55}\n")

def _write_csv(output_path: str, vocab: int, dims: int,
               sim: dict, ana: dict, coh: dict) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    file_exists = os.path.exists(output_path)
    with open(output_path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow([
                'model', 'vocab_size', 'dims',
                'wordsim353_pearson_r', 'wordsim353_spearman_r', 'oov_ratio',
                'analogy_accuracy',
                'coherence_scored', 'coherence_skipped', 'mean_coherence',
                'eval_date',
            ])
        writer.writerow([
            'trained_eng', vocab, dims,
            sim['pearson_r'], sim['spearman_r'], sim['oov_ratio'],
            ana['overall_accuracy'],
            coh['scored'], coh['skipped'], coh['mean_coherence'],
            datetime.now().isoformat(timespec='seconds'),
        ])
    print(f"  CSV saved : {output_path}")

def _load_probes(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Coherence probe file not found: {path}")
    with open(path, encoding='utf-8') as fh:
        return [l.strip() for l in fh if l.strip()]

def _file_mode(wv, cfg: dict) -> None:
    topn        = cfg.get('eval_topn', 10)
    probe_words = _load_probes(cfg['eval_coherence_file'])
    print(f"  Coherence probes : {len(probe_words)}")
    print(f"  Top-N            : {topn}")
    print("\n  [1/3] Similarity (WordSim-353) …")
    sim = task_similarity(wv)
    print("  [2/3] Analogy (Google Analogy) …")
    ana = task_analogy(wv)
    print("  [3/3] Coherence …")
    coh = task_coherence(wv, probe_words, topn=topn)
    _print_results(len(wv), wv.vector_size, sim, ana, coh)
    etype       = cfg.get('embedding_type_eng', 'unknown').lower()
    default     = f'ii_embeddings/eval/e_trained_{etype}_eval.csv'
    _write_csv(cfg.get('eval_output_trained_e', default), len(wv), wv.vector_size, sim, ana, coh)


#  Pipeline step wrapper
def run_test_embed_eng(cfg: dict) -> None:
    model_path = cfg['test_input_file_eng']
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Embedding not found: {model_path}\n"
            "Run the 'e.embed' step first."
        )
    print(f"  Loading {cfg['embedding_type_eng'].upper()} model: {model_path}")
    model = _load_model(model_path, cfg['embedding_type_eng'])
    wv    = model.wv
    print(f"  Vocabulary : {len(wv):,} words")
    print(f"  Dimensions : {model.vector_size}")
    mode = cfg.get('test_mode_eng', 'interactive').lower()
    if mode == 'interactive':
        _interactive_mode(wv)
    elif mode == 'file':
        _file_mode(wv, cfg)
    else:
        raise ValueError(f"Unknown test_mode_eng: '{mode}'. Use 'interactive' or 'file'.")
