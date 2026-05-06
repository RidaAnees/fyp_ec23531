
"""
Cleans raw English text:
"""
import re
import os
import unicodedata
from collections import Counter

#Character ranges
_LATIN   = re.compile(r'[a-zA-Z]+')
_FOREIGN = re.compile(r'[^\x00-\x7F]+') #non-ASCII tokens
_KEEP    = re.compile(r'[^a-zA-Z\s]')#chars to strip

#helpers 
def normalise_unicode(text: str) -> str:
    """NFKC normalisation + zero-width character removal."""
    text = text.replace('\u200c', '').replace('\u200d', '')
    return unicodedata.normalize('NFKC', text)

def normalise_spelling(text: str) -> str:
    """
    Normalise common English spelling variants and ligatures.
    """
    _LIGATURES = str.maketrans({
        '\u00e6': 'ae',   
        '\u00f6': 'oe',   
        '\u0153': 'oe',   
        '\ufb01': 'fi',   
        '\ufb02': 'fl',   
        '\u2019': "'",    #right single quotation mark
        '\u2018': "'",    #left single quotation mark
        '\u201c': '"',    # left double quotation mark
        '\u201d': '"',    #right double quotation mark
        '\u2014': ' ',    #em dash ->  space
        '\u2013': ' ',    # en dash -> space
    })
    return text.translate(_LIGATURES)

def strip_non_latin(text: str) -> str:
    """Remove everything that is not ASCII alpha or whitespace"""
    return _KEEP.sub('', text)

def extract_foreign_tokens(text: str) -> list:
    """
    Return non-ASCII tokens found in an English line, with plus or minus 3 word context.
    """
    words= text.split()
    results= []
    for i, word in enumerate(words):
        if _FOREIGN.search(word):
            results.append({
                'word': word,
                'left_context':  words[max(0, i - 3):i],
                'right_context': words[i + 1:i + 4],
            })
    return results

def detect_script(text: str) -> str:
    """Return 'latin', 'foreign', or 'mixed'."""
    has_latin   = bool(_LATIN.search(text))
    has_foreign = bool(_FOREIGN.search(text))
    if has_latin and not has_foreign:
        return 'latin'
    if has_foreign and not has_latin:
        return 'foreign'
    return 'mixed'

def clean_line(line: str) -> str:
    """Full cleaning pipeline for a single English line."""
    line = normalise_unicode(line)
    line = normalise_spelling(line)
    line = line.lower()             
    line = strip_non_latin(line)
    return line.strip()

#  Single file 
def clean_file(input_path: str, output_path: str, log_foreign: bool = True) -> dict:
    """Read raw English text from one file, clean it, write cleaned output then returns a stats dict """
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}")
    with open(input_path, 'r', encoding='utf-8') as fh:
        raw_lines = fh.readlines()
    cleaned = []
    foreign_tokens = Counter()
    n_dropped = 0
    for raw in raw_lines:
        if log_foreign:
            for ft in extract_foreign_tokens(raw):
                foreign_tokens[ft['word'].lower()] += 1
        line= clean_line(raw)
        if line:
            cleaned.append(line)
        else:
            n_dropped += 1
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(cleaned))
    stats = {
        'raw_lines': len(raw_lines),
        'clean_lines': len(cleaned),
        'dropped_lines': n_dropped,
        'unique_foreign_tokens': len(foreign_tokens),
        'top_foreign_tokens': foreign_tokens.most_common(20),
    }
    print(f"Raw lines : {stats['raw_lines']:,}")
    print(f"Clean lines: {stats['clean_lines']:,}")
    print(f"Dropped: {stats['dropped_lines']:,}")
    print(f" Foreign tokens (unique): {stats['unique_foreign_tokens']:,}")
    if stats['top_foreign_tokens']:
        print(f"  Top 10 foreign: "
              f"{[w for w, _ in stats['top_foreign_tokens'][:10]]}")
    return stats

#  Main function 
def run_preprocessing_eng(cfg: dict) -> None:
    """
    Clean and merge multiple raw English files into a single clean file
    """
    raw= cfg['raw_eng']
    clean = cfg['clean_eng']
    if isinstance(raw, list) and isinstance(clean, str):
        print(f"  Merging {len(raw)} raw file(s) into single output:")
        all_cleaned_lines = []
        total_foreign = Counter()
        total_dropped = 0
        total_raw_lines= 0
        for i, raw_path in enumerate(raw, 1):
            print(f"\n [{i}/{len(raw)}] Processing: {os.path.basename(raw_path)}")
            with open(raw_path, 'r', encoding='utf-8') as fh:
                raw_lines = fh.readlines()
            total_raw_lines += len(raw_lines)
            file_cleaned= []
            for line in raw_lines:
                for ft in extract_foreign_tokens(line ) :
                    total_foreign[ft['word'].lower()] += 1
                cleaned_line= clean_line(line)
                if cleaned_line:
                    file_cleaned.append(cleaned_line)
                    all_cleaned_lines.append(cleaned_line )
                else:
                    total_dropped+= 1
            print(f"      -> {len(file_cleaned):,} clean lines")
        os.makedirs(os.path.dirname(clean) or '.', exist_ok=True )
        with open(clean, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(all_cleaned_lines))
        print(f"\n  {'='*50}")
        print(f"MERGED OUTPUT: {clean}")
        print(f"Total raw files: {len(raw)}" )
        print(f"Total raw lines: {total_raw_lines:,}" )
        print(f"Total clean lines: {len(all_cleaned_lines):,}")
        print(f"Total dropped: {total_dropped:,}")
        print(f"Unique foreign tokens: {len(total_foreign):,}" )
        if total_foreign:
            print(f"  Top 10 foreign: {[w for w, _ in total_foreign.most_common(10)]}")
        return
    # single file
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(clean, str):
        clean = [clean]
    if len(raw) != len(clean):
        raise ValueError(
            f"raw_eng has {len(raw)} file(s) but clean_eng has {len(clean)} -counts must match."
        )
    for i, (raw_path, clean_path) in enumerate(zip(raw, clean), 1 ):
        print(f"\nFile {i}/{len(raw)}: {raw_path}" )
        clean_file(raw_path, clean_path)
