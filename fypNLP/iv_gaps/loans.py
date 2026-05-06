
"""
Classify Dhivehi vocabulary tokens as native_dhivehi/ loanword/named_entity 
For each Dhivehi word:
  1. Romanise it (Thaana -> Latin)
  2. Named-entity check against word lists
  3. Get top-K nearest English neighbours from the aligned embedding space
  4. For each neighbour, compute phonetic similarity (normalised Levenshtein with borrowing-adaptation normalisation applied to both sides)
  5. If best phonetic score >= phonetic_threshold AND embedding similarity >= embed_threshold -> loanword
Uses Transliterator.py if available (import tried first) Falls back to a built-in rule-based Thaana -> Latin table if not found
"""
import os
import re
import csv
from collections import defaultdict
import numpy as np
import editdistance
from gensim.models import KeyedVectors

#fallback transliterator
_THAANA_MAP = {
    # Consonants
    'ހ': 'h',  'ށ': 'sh', 'ނ': 'n',  'ރ': 'r',  'ބ': 'b',
    'ޅ': 'lh', 'ކ': 'k',  'އ': 'a',  'ވ': 'v',  'މ': 'm',
    'ފ': 'f',  'ދ': 'dh', 'ތ': 'th', 'ލ': 'l',  'ގ': 'g',
    'ޏ': 'gn', 'ސ': 's',  'ޑ': 'd',  'ޒ': 'z',  'ޓ': 't',
    'ޔ': 'y',  'ޕ': 'p',  'ޖ': 'j',  'ޗ': 'ch', 'ޘ': 'ss',
    'ޙ': 'h',  'ޚ': 'kh', 'ޛ': 'z',  'ޜ': 'z',  'ޝ': 'sh',
    'ޞ': 's',  'ޟ': 'd',  'ޠ': 't',  'ޡ': 'z',  'ޢ': 'a',
    'ޣ': 'gh', 'ޤ': 'q',  'ޥ': 'w',
    # Vowel diacritics
    'ަ': 'a',  'ާ': 'aa', 'ި': 'i',  'ީ': 'ee', 'ު': 'u',
    'ޫ': 'oo', 'ެ': 'e',  'ޭ': 'ey', 'ޮ': 'o',  'ޯ': 'oa',
    'ް': '', 
}

# romaniser
def _thaana_to_latin(word: str) -> str:
    try:
        from Transliterator import transliterate_word
        return transliterate_word(word)
    except ImportError:
        return ''.join(_THAANA_MAP.get(ch, ch) for ch in word)

#phonetic similarity
def _normalise_for_comparison(word: str) -> str:
    """Collapse common Dhivehi borrowing adaptations before Levenshtein comparison.Dhivehi adds long vowels and final helper vowels when borrowing English words.
    Eg
      beynku -> benk   (closer to 'bank')
      intanet-> intant (closer to 'internet')
    """
    word = word.lower()
    word = re.sub(r'(.)\1+', r'\1', word) # collapse repeated chars (vowel lengthening)
    word = re.sub(r'[uiea]$', '', word)  # strip word final helper vowel
    return word

def phonetic_similarity(dv_romanised: str, en_word: str) -> float:
    """
    Normalised Levenshtein similarity in [0, 1] after borrowing normalisation.1.0 = identical, 0.0 = completely different.
    """
    dv_norm = _normalise_for_comparison(dv_romanised)
    en_norm= _normalise_for_comparison(en_word)
    max_len = max(len(dv_norm), len(en_norm), 1)
    dist= editdistance.eval(dv_norm, en_norm)
    return 1.0 - dist / max_len

#Named entity loader
def _load_ne_lists(ne_dir: str) -> set:
    """
    Load plain-text proper noun lists from a directory. each file should have one word per line (lowercase). e.g. person_names.txt, place_names.txt, brand_names.txt
    """
    ne_set = set()
    if not ne_dir or not os.path.isdir(ne_dir):
        return ne_set
    for fname in os.listdir(ne_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(ne_dir, fname), encoding='utf-8') as fh:
                for line in fh:
                    word = line.strip().lower()
                    if word:
                        ne_set.add(word)
    print(f"  Loaded {len(ne_set):,} named-entity entries from {ne_dir}")
    return ne_set

# Classifier
class LoanWordDetector:
    def __init__(self, aligned_kv_path: str, ne_lists_dir: str = '', phonetic_threshold: float = 0.65,embed_threshold: float= 0.55,top_k: int= 20):
        print(f"  Loading aligned embeddings: {aligned_kv_path}")
        self.kv= KeyedVectors.load(aligned_kv_path, mmap='r')
        self.ne_set = _load_ne_lists(ne_lists_dir)
        self.phonetic_threshold  = phonetic_threshold
        self.embed_threshold = embed_threshold
        self.top_k = top_k
        self.dv_words = [w[3:] for w in self.kv.index_to_key if w.startswith('dv_')]
        self.en_words = [w[3:] for w in self.kv.index_to_key if w.startswith('en_')]
        print(f"Dhivehi vocab : {len(self.dv_words):,}" )
        print(f"English vocab : {len(self.en_words):,}" )
        print(f"Phonetic threshold : {phonetic_threshold}" )
        print(f"Embed threshold : {embed_threshold}")
        print(f" Top-K neighbours : {top_k}")
        try:
            from Transliterator import transliterate_word  # noqa: F401
            print(f"  Transliterator: Transliterator.py (imported)")
        except ImportError:
            print(f"  Transliterator: built-in rule table (Transliterator.py not found)")
    #internal helpers
    def _result(self, word, romanised, category, best_en, phon, embed) -> dict:
        return {
            'word':  word,
            'romanised': romanised,
            'category':  category,
            'best_english_match': best_en,
            'phonetic_score':  round(phon,  4),
            'embed_score': round(embed, 4),
        }
    #core classification
    def classify_word(self, dv_word: str) -> dict:
        romanised = _thaana_to_latin(dv_word)
        if romanised.lower() in self.ne_set or dv_word.lower() in self.ne_set:
            return self._result(dv_word, romanised, 'named_entity', '', 0.0, 0.0)
        try:
            neighbours= self.kv.most_similar(f'dv_{dv_word}', topn=self.top_k * 3)
            en_neighbours = [(w[3:], s) for w, s in neighbours if w.startswith('en_')]
            en_neighbours = en_neighbours[:self.top_k]
        except KeyError:
            return self._result(dv_word, romanised, 'native_dhivehi', '', 0.0, 0.0)
        best_en= ''
        best_phon = 0.0
        best_embed = 0.0
        for en_word,embed_sim in en_neighbours:
            phon = phonetic_similarity(romanised, en_word)
            if phon > best_phon:
                best_phon= phon
                best_en= en_word
                best_embed = embed_sim
        # phonetically similar AND semantically close
        is_loan  = (best_phon  >= self.phonetic_threshold and best_embed >= self.embed_threshold)
        category = 'loanword' if is_loan else 'native_dhivehi'
        return self._result(dv_word, romanised, category, best_en, best_phon, best_embed)
    def classify_all(self, max_words: int = 0) -> list:
        words = self.dv_words[:max_words] if max_words else self.dv_words
        print(f"\n  Classifying {len(words):,} Dhivehi words … " )
        results= []
        for i, w in enumerate(words, 1):
            results.append(self.classify_word(w))
            if i % 5000 == 0:
                print(f"    … {i:,}/{len(words):,}")
        # Summary
        counts = defaultdict(int)
        for r in results:
            counts[r['category']] += 1
        total = len(results)
        print("\n Classification summary")
        for cat, n in sorted(counts.items()):
            pct = 100 * n / max(total, 1)
            bar = '-' * int(pct / 2)
            print(f" {cat:<20s}: {n:>6,}  ({pct:5.1f}%)  {bar}")
        print(f"  {'TOTAL':<20s}: {total:>6,}")
        #show examples
        loans = [r for r in results if r['category'] == 'loanword']
        if loans:
            print(f"\n  Sample loanwords detected:")
            for r in loans[:10]:
                print(f"    {r['word']:<20s} -> {r['romanised']:<15s} "
                      f"≈ {r['best_english_match']:<15s} "
                      f"(phon={r['phonetic_score']:.2f}, "
                      f"embed={r['embed_score']:.2f})")
        return results

# Pipeline step wrapper
def run_loan_detection(cfg: dict) -> None:
    aligned_path = cfg.get('aligned_space_output', '')
    if not aligned_path or not os.path.exists(aligned_path):
        raise FileNotFoundError(
            f"Aligned space not found: {aligned_path}\n"
            "Run the 'align' step first."
        )
    detector = LoanWordDetector(
        aligned_kv_path = aligned_path,
        ne_lists_dir= cfg.get('ne_lists_dir', ''),
        phonetic_threshold = cfg.get('phonetic_threshold', 0.65),
        embed_threshold = cfg.get('embed_threshold',0.55),
        top_k = cfg.get('loan_top_k',20),
    )
    results  = detector.classify_all()
    out_path = cfg.get('loan_output', 'v_results/loan_words.csv')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            'word', 'romanised', 'category',
            'best_english_match', 'phonetic_score', 'embed_score'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to -> {out_path}")
