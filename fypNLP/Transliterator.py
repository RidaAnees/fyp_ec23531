# """
# 1. Rule-based  (default, no training data needed)
# 2. Passthrough  (if run_transliteration is False in config)
# ──────
# One transliterated token per line (same line count as clean Thaana input).
# """
# import re
# import os
# import unicodedata

# # Based on the ALA-LC romanisation scheme adapted for Dhivehi.
# # Reference: Gnanadesikan (2017) and common digital Dhivehi practice.
# _CONSONANTS = {
#     '\u0780': 'h',   # ހ  haa
#     '\u0781': 'sh',  # ށ  shaviyani
#     '\u0782': 'n',   # ނ  noonu
#     '\u0783': 'r',   # ރ  raa
#     '\u0784': 'b',   # ބ  baa
#     '\u0785': 'lh',  # ޅ  lhaviyani
#     '\u0786': 'k',   # ކ  kafu
#     '\u0787': 'a',   # އ  alifu (glottal — rendered as bare vowel carrier)
#     '\u0788': 'v',   # ވ  vaavu
#     '\u0789': 'm',   # މ  meemu
#     '\u078A': 'f',   # ފ  faa
#     '\u078B': 'dh',  # ދ  dhaalu
#     '\u078C': 'th',  # ތ  thaalu
#     '\u078D': 'l',   # ލ  laamu
#     '\u078E': 'g',   # ގ  gaafu
#     '\u078F': 'gn',  # ޏ  gnaviyani
#     '\u0790': 's',   # ސ  seenu
#     '\u0791': 'd',   # ޑ  daalu
#     '\u0792': 'z',   # ޒ  zainu
#     '\u0793': 't',   # ޓ  taviyani
#     '\u0794': 'y',   # ޔ  yaa
#     '\u0795': 'p',   # ޕ  paviyani
#     '\u0796': 'j',   # ޖ  jeem
#     '\u0797': 'ch',  # ޗ  chaviyani
#     '\u0798': 'tt',  # ޘ  ttaa  (retroflex, rare)
#     '\u0799': 'hh',  # ޙ  hhaa  (pharyngeal, rare)
#     '\u079A': 'kh',  # ޚ  khaa
#     '\u079B': 'th',  # ޛ  thaalu variant
#     '\u079C': 'z',   # ޜ  zainu variant
#     '\u079D': 'sh',  # ޝ  shainu
#     '\u079E': 'ss',  # ޞ  saadhu
#     '\u079F': 'dh',  # ޟ  dhaadhu
#     '\u07A0': 't',   # ޠ  to
#     '\u07A1': 'z',   # ޡ  zaa
#     '\u07A2': 'a',   # ޢ  ain
#     '\u07A3': 'gh',  # ޣ  gainu
#     '\u07A4': 'q',   # ޤ  qaafu
#     '\u07A5': 'w',   # ޥ  waavu
# }
# _VOWEL_MARKS = {
#     '\u07A6': 'a',   # ަ  abafili  (short a)
#     '\u07A7': 'aa',  # ާ  aabaafili (long aa)
#     '\u07A8': 'i',   # ި  ibifili  (short i)
#     '\u07A9': 'ee',  # ީ  eebee    (long ii)
#     '\u07AA': 'u',   # ު  ubufili  (short u)
#     '\u07AB': 'oo',  # ޫ  ooboofili (long uu)
#     '\u07AC': 'e',   # ެ  ebefili  (short e)
#     '\u07AD': 'ey',  # ޭ  eybeyfili (long ee)
#     '\u07AE': 'o',   # ޮ  obofili  (short o)
#     '\u07AF': 'oa',  # ޯ  oaboafili (long oo)
#     '\u07B0': '',    # ް  sukun    ( no vowel, silent)
# }
# _THAANA_RANGE = re.compile(r'[\u0780-\u07BF]')

# def transliterate_word(word: str) -> str:
#     """Convert a single Thaana word to its Latin romanisation."""
#     result = []
#     i = 0
#     chars = list(word)
#     while i < len(chars):
#         ch = chars[i]
#         if ch in _CONSONANTS:
#             result.append(_CONSONANTS[ch])
#             # peek at next char for vowel mark
#             if i + 1 < len(chars) and chars[i + 1] in _VOWEL_MARKS:
#                 result.append(_VOWEL_MARKS[chars[i + 1]])
#                 i += 2
#             else:
#                 # bare consonant (usually gemination or word-final)
#                 i += 1
#         elif ch in _VOWEL_MARKS:
#             # standalone diacritic (shouldn't normally appear alone)
#             result.append(_VOWEL_MARKS[ch])
#             i += 1
#         else:
#             # non-Thaana character — keep as is
#             result.append(ch)
#             i += 1
#     return ''.join(result)

# def transliterate_line(line: str) -> str:
#     """Transliterate all Thaana tokens in a line; leave Latin tokens untouched."""
#     tokens = line.split()
#     out = []
#     for tok in tokens:
#         if _THAANA_RANGE.search(tok):
#             out.append(transliterate_word(tok))
#         else:
#             out.append(tok.lower()) # already Latin (code-switch or punctuation)
#     return ' '.join(out)

# def transliterate_file(input_path: str, output_path: str) -> dict:
#     """
#     Transliterate an entire cleaned Thaana file to Latin.
#     Returns a stats dict.
#     """
#     print(f"  Input  : {input_path}")
#     print(f"  Output : {output_path}")
#     with open(input_path, 'r', encoding='utf-8') as fh:
#         lines = [l.rstrip('\n') for l in fh]
#     transliterated = [transliterate_line(l) for l in lines]
#     os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
#     with open(output_path, 'w', encoding='utf-8') as fh:
#         fh.write('\n'.join(transliterated))
#     stats = {'lines': len(lines)}
#     print(f"  Lines processed: {stats['lines']:,}")
#     return stats

# def run_transliteration(cfg: dict) -> None:
#     if not cfg.get('run_transliteration', False):
#         print("  Transliteration disabled in config — skipping.")
#         return
#     transliterate_file(cfg['clean_dhiv'], cfg['transliterate_output'])
