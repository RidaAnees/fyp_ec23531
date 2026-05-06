
"""
Cleans raw Dhivehi 
"""
import re
import os
import unicodedata
from collections import Counter

#Thaana /Latin character ranges 
_THAANA= re.compile(r'[\u0780-\u07BF]+')
_LATIN  = re.compile(r'[a-zA-Z]+' )
_KEEP   =re.compile(r'[^\u0780-\u07BF\s]' )   #chars to strip

#Spelling normalisation tables
#Thaana letters to modern equivalents
_CLASSICAL_TO_MODERN = str.maketrans({
    '\u07A4': '\u078E',   # ޤ qaafu     ގ gaafu
    '\u07A2': '\u0787',   # ޢ ain       އ alifu
    '\u0799': '\u0780',   # ޙ hhaa       ހ haa
    '\u079B': '\u0792',   # ޛ zha        ޒ zaviyani
    '\u079C': '\u0792',   # ޜ zainu variant  ޒ zaviyani
    '\u07A1': '\u0792',   # ޡ zaa          ޒ zaviyani
    '\u07A3': '\u078E',   # ޣ gainu        ގ gaafu
    '\u07A0': '\u078C',   # ޠ to          ތ thaalu
    '\u079E': '\u0790',   # ޞ saadhu     ސ seenu
    '\u079D': '\u0781',   # ޝ shainu     ށ shaviyani
    '\u0798': '\u0790',   # ޘ ttaa        ސ seenu
})

# helpers
def normalise_unicode(text: str)-> str:
    """NFKC normalisation + zero-width character removal."""
    text= text.replace('\u200c', '').replace('\u200d', '')
    return unicodedata.normalize('NFKC', text )

def normalise_spelling(text: str) -> str:
    """
    normalises classical Thaana letters to modern equivalents.
    """
    return text.translate(_CLASSICAL_TO_MODERN)

def strip_non_thaana(text: str) -> str:
    """ Remove all that is not Thaana or whitespace"""
    return _KEEP.sub('', text)

def extract_code_switches(text: str) -> list:
    """Return foreign tokens found inside a Dhivehi line, with plus or minus 3 word context"""
    words= text.split()
    results= []
    for i,word in enumerate(words):
        if _LATIN.fullmatch(word):
            results.append ({
                'word': word,
                'left_context': words[max(0, i - 3):i],
                'right_context': words[i + 1:i + 4],
            } )
    return results

def detect_script(text: str) -> str:
    """Return 'thaana', 'latin', or 'mixed'"""
    t= len(_THAANA.findall(text))
    l = len(_LATIN.findall(text))
    if t> 0 and l == 0:
        return 'thaana'
    if l>0 and t == 0:
        return 'latin'
    return 'mixed'

def clean_line (line: str) -> str:
    line = normalise_unicode(line)
    line = normalise_spelling(line)
    line = strip_non_thaana(line)
    return line.strip()

def clean_file(input_path: str, output_path: str,
               log_code_switches: bool = True) -> dict:
    """ Read raw Dhivehi text from ONE file, clean it, write cleaned output. Returns a stats dict """
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    with open(input_path, 'r' , encoding='utf-8') as fh:
        raw_lines = fh.readlines()
    cleaned=[]
    code_switches = Counter()
    n_dropped = 0
    for raw in raw_lines:
        if log_code_switches:
            for cs in extract_code_switches(raw) :
                code_switches[cs['word'].lower()] += 1
        line= clean_line(raw)
        if line:
            cleaned.append(line)
        else:
            n_dropped += 1
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(cleaned))
    stats = {
        'raw_lines':len(raw_lines),
        'clean_lines':  len(cleaned),
        'dropped_lines' :  n_dropped,
        'unique_cs_tokens': len(code_switches),
        'top_cs_tokens' :  code_switches.most_common(20),
    }
    print (f"  Raw lines    : {stats['raw_lines']:,}")
    print( f"  Clean lines  :{stats['clean_lines']:,}" )
    print (f"  Dropped : {stats['dropped_lines']:,}" )
    print(f"  Code-switched tokens (unique) : {stats['unique_cs_tokens']:,} ")
    if stats['top_cs_tokens']:
        print(f"Top 10 code-switched: "
              f"{[w for w, _ in stats['top_cs_tokens'][:10]]}")
    return stats

# Main function
def run_preprocessing(cfg: dict) -> None:
    """Clean and merge multiple raw Dhivehi files into a single clean file """
    raw= cfg['raw_dhiv']
    clean = cfg['clean_dhiv']
    
    # multiple raw files to 1 clean file 
    if isinstance(raw, list) and isinstance(clean, str):
        print(f"  Merging {len(raw)} raw file(s) into single output:")
        
        all_cleaned_lines= []
        total_code_switches = Counter()
        total_dropped= 0
        total_raw_lines =  0
        
        for i, raw_path in enumerate(raw, 1):
            print(f"\n [{i}/{len(raw)}] Processing : {os.path.basename(raw_path)}" )
            
            with open(raw_path, 'r', encoding='utf-8') as fh:
                raw_lines = fh.readlines()
            
            total_raw_lines += len(raw_lines)
            file_cleaned= []
            
            for line in raw_lines:
                # Extract code-switches before cleaning
                for cs in extract_code_switches(line):
                    total_code_switches[cs['word'].lower()] += 1
                
                cleaned_line = clean_line(line )
                if cleaned_line:
                    file_cleaned.append(cleaned_line)
                    all_cleaned_lines.append(cleaned_line )
                else:
                    total_dropped += 1
            
            print(f"      -> {len(file_cleaned):,} clean lines " )
        
        #write the merged output
        os.makedirs(os.path.dirname(clean) or '.', exist_ok=True )
        with open(clean, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(all_cleaned_lines) )
        
        # Print summary
        print(f"\n  {'='*50}")
        print(f" MERGED OUTPUT: {clean}" )
        print(f"Total raw files    : {len(raw)}" )
        print(f"Total raw lines : {total_raw_lines:,}" )
        print(f"Total clean lines  : {len(all_cleaned_lines):,}")
        print(f"Total dropped    : {total_dropped:,}")
        print(f"  Unique code-switched tokens: {len(total_code_switches):,}" )
        if total_code_switches:
            print(f"  Top 10 code-switched:{[w for w, _ in total_code_switches.most_common(10)]} " )
        return
    
    #Single file or 1-to-1 mapping
    if isinstance(raw, str):
        raw = [raw]
    
    if isinstance(clean, str):
        clean = [clean]
    
    if len(raw) != len(clean):
        raise ValueError(
            f" raw_dhiv has {len(raw)} file(s) but clean_dhiv has {len(clean)} - counts must match.\n"
            " For merging multiple raw files into one clean file, use:\n"
            "  'raw_dhiv': ['file1.txt', 'file2.txt'],\n"
            "  'clean_dhiv': 'merged_output.txt'"
        )
    
    for i,(raw_path, clean_path) in enumerate(zip(raw, clean), 1):
        print(f"\n  File {i}/{len(raw)}: {raw_path} ")
        clean_file(raw_path, clean_path)
