
"""
Identify "lexical gaps" aka English concepts that have no close Dhivehi equivalent in the aligned embedding space
For each English word in the top-N vocabulary:
  - Project into the aligned space (en_WORD)
  - find its nearest Dhivehi neighbour (dv_WORD)
  - If the best cosine similarity < gap_threshold -> flag as a gap
Seed word exclusion
Words that appear in the seed dictionary are excluded from gap evaluation.
The alignment matrix was trained on these pairs, so their cosine similarities are inflated so reporting them as "matches" would be data leakage
The seed word list is written automatically by align.py as a .txt file alongside the aligned .kv file.
The output CSV includes a semantic domain tag derived from a small hand-coded domain keyword list.
"""
import os
import csv
from collections import defaultdict
import numpy as np
from gensim.models import KeyedVectors

#Lightweight semantic domain tagger 
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    'technology':  ['internet', 'computer', 'software', 'app', 'phone', 'digital',
                    'wifi', 'email', 'server', 'database', 'algorithm', 'pixel',
                    'smartphone', 'laptop', 'tablet', 'cloud', 'bluetooth'],
    'finance':     ['bank', 'loan', 'credit', 'debit', 'budget', 'inflation',
                    'mortgage', 'dividend', 'equity', 'tax', 'currency', 'invest'],
    'medicine':    ['hospital', 'vaccine', 'diagnosis', 'therapy', 'surgery',
                    'antibiotic', 'pandemic', 'virus', 'symptom', 'clinic'],
    'politics':    ['democracy', 'parliament', 'constitution', 'election', 'senator',
                    'coalition', 'referendum', 'minister', 'diplomat', 'sanction'],
    'sports':      ['football', 'cricket', 'basketball', 'tennis', 'marathon',
                    'tournament', 'championship', 'athlete', 'referee', 'penalty'],
    'food':        ['restaurant', 'pizza', 'burger', 'coffee', 'chocolate',
                    'pasta', 'sushi', 'buffet', 'calorie', 'protein'],
    'education':   ['university', 'curriculum', 'thesis', 'semester', 'scholarship',
                    'bachelor', 'master', 'degree', 'exam', 'lecture'],
    'environment': ['climate', 'pollution', 'sustainability', 'ecosystem',
                    'biodiversity', 'carbon', 'recycle', 'coral', 'tsunami'],
}
#build reverse lookup: keyword -> domain
_KEYWORD_TO_DOMAIN: dict[str, str] = {}
for _domain, _keywords in _DOMAIN_KEYWORDS.items():
    for _kw in _keywords:
        _KEYWORD_TO_DOMAIN[_kw]= _domain

def _tag_domain(en_word: str) -> str:
    return _KEYWORD_TO_DOMAIN.get(en_word.lower(), 'other')

#Gap detector
class LexicalGapDetector:
    def __init__(self, aligned_kv_path: str,
                 gap_threshold: float = 0.60,
                 top_n_english: int   = 10_000):
        print(f"  Loading aligned embeddings: {aligned_kv_path}")
        self.kv  = KeyedVectors.load(aligned_kv_path, mmap='r')
        self.kv_path = aligned_kv_path  
        self.gap_threshold = gap_threshold
        self.top_n_english = top_n_english
        self.dv_keys= [w for w in self.kv.index_to_key if w.startswith('dv_')]
        self.en_keys= [w for w in self.kv.index_to_key if w.startswith('en_')]
        #pre-build Dhivehi matrix for fast nearest-neighbour search
        print(f"  Building Dhivehi vector matrix ({len(self.dv_keys):,} words) …")
        self.dv_mat = np.array(
            [self.kv[w] for w in self.dv_keys], dtype=np.float32)
        
        # Row-normalise one time
        norms= np.linalg.norm(self.dv_mat, axis=1, keepdims=True) + 1e-9
        self.dv_mat_normed = self.dv_mat / norms
        print(f"English words to scan  : min({len(self.en_keys):,}, {top_n_english:,})")
    def _nearest_dhivehi(self, en_vec: np.ndarray) -> tuple[str, float]:
        """Return (word, cosine_sim) of the nearest Dhivehi word."""
        en_normed = en_vec / (np.linalg.norm(en_vec) + 1e-9)
        sims= self.dv_mat_normed @ en_normed
        idx = int(np.argmax(sims))
        return self.dv_keys[idx][3:], float(sims[idx])# strip 'dv_' prefix
    def _load_seed_words(self) -> set[str]:
        seed_path = self.kv_path.replace('.kv', '_seed_words.txt')
        seed_words = set()
        if os.path.exists(seed_path):
            with open(seed_path, encoding='utf-8') as f:
                seed_words = {line.strip() for line in f if line.strip()}
            print(f"Excluding {len(seed_words):,} seed words from gap evaluation")
            print(f"(these were used to train the alignment matrix)")
        else:
            print(f" Warning: no seed word list found at {seed_path}")
            print(f"All English words will be scanned")
                  
        return seed_words
    def find_gaps(self) -> list[dict]:
        """scan top-N English words (excluding seed words) and flag those with no close Dhivehi match
        """
        #exclude seed words to prevent data leakage
        seed_words= self._load_seed_words()
        scan_keys= [k for k in self.en_keys[:self.top_n_english]
                      if k[3:] not in seed_words]   # k[3:] strips 'en_' prefix
        print(f"  Scanning {len(scan_keys):,} English words "
              f"(after excluding {len(seed_words):,} seed words) …")
        results = []
        n_gaps  = 0
        for i, en_key in enumerate(scan_keys, 1):
            en_word = en_key[3:]   # strip 'en_' prefix
            en_vec  = self.kv[en_key]
            nearest_dv, sim = self._nearest_dhivehi(en_vec)
            is_gap = sim < self.gap_threshold
            if is_gap:
                n_gaps += 1
            results.append({
                'english_word':en_word,
                'nearest_dhivehi':nearest_dv,
                'cosine_sim': round(sim, 4),
                'is_gap': is_gap,
                'domain':  _tag_domain(en_word),
            })
            if i % 2000 == 0:
                print(f" … {i:,}/{len(scan_keys):,}  (gaps so far: {n_gaps:,})")
        print(f"Total gaps detected : {n_gaps:,} / {len(scan_keys):,} "
              f"({100*n_gaps/max(len(scan_keys),1):.1f}%)")
        #Domain breakdown of gaps
        domain_counts: defaultdict[str, int] = defaultdict(int)
        for r in results:
            if r['is_gap']:
                domain_counts[r['domain']] += 1
        print("  Gap breakdown by domain:")
        for dom, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"    {dom:<20s}: {cnt:,}")
        return results

#Pipeline step wrapper
def run_gap_detection(cfg: dict) -> None:
    aligned_path = cfg.get('aligned_space_output', '')
    if not aligned_path or not os.path.exists(aligned_path):
        raise FileNotFoundError(
            f"Aligned space not found: {aligned_path}\n"
            "Run the 'align' step first."
        )
    detector = LexicalGapDetector(
        aligned_kv_path = aligned_path,
        gap_threshold = cfg.get('gap_threshold',  0.60),
        top_n_english = cfg.get('top_n_english',  10_000),
    )
    results  = detector.find_gaps()
    out_path = cfg.get('gap_output', 'v_results/lexical_gaps.csv')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'english_word', 'nearest_dhivehi', 'cosine_sim', 'is_gap', 'domain'])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved -> {out_path}")
