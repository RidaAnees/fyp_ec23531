
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃                  !!!!                        ┃
┃    Edit config.py to control behaviour       ┃
┃        DO NOT MODIFY THIS FILE               ┃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dhivehi NLP Pipeline — main entry point
"""
from config import config
from i_data.preprocess                  import run_preprocessing    
from i_data.preprocess_eng              import run_preprocessing_eng
from Transliterator                     import run_transliteration
from ii_embeddings.embed                import run_embedding        
from ii_embeddings.embed_eng            import run_embedding_eng
from ii_embeddings.dhiv_embed_test.test_embed_dhiv    import run_test_embed_dhiv      
from ii_embeddings.eng_embed_test.test_embed_eng      import run_test_embed_eng
from ii_embeddings.eng_embed_test.test_pretrained    import run_test_pretrained_eng
from iii_alignment.align      import run_alignment
from iv_gaps.loans            import run_loan_detection
from iv_gaps.gaps             import run_gap_detection
#from vi_visuals.visualise     import run_visualisation
# from i_data.stemming          import run_stemming
# Step registry 
STEPS = {
    'd': {
        'preprocess':     run_preprocessing,
        'transliterate':  run_transliteration,
        # 'stem':           run_stemming,
        'embed':          run_embedding,
        'test_embed':     run_test_embed_dhiv,
        # no test_pretrained for Dhivehi
    },
    'e': {
        'preprocess':      run_preprocessing_eng,
        'embed':           run_embedding_eng,
        'test_embed':      run_test_embed_eng,
        'test_pretrained': run_test_pretrained_eng
    },
    'shared': {
        'align':        run_alignment,
        'detect_loans': run_loan_detection,
        'find_gaps':    run_gap_detection,
        #'visualise':    run_visualisation,
    },
}
ALL_STEPS_D      = list(STEPS['d'].keys())
ALL_STEPS_E      = list(STEPS['e'].keys())
ALL_STEPS_SHARED = list(STEPS['shared'].keys())

#  Helpers
def _resolve_steps(raw) -> list[tuple[str, str]]:
    if isinstance(raw, str):
        raw = [s.strip() for s in raw.split(',')]
    if raw == ['all']:
        return (
            [('d', s) for s in ALL_STEPS_D] +
            [('e', s) for s in ALL_STEPS_E] +
            [('shared', s) for s in ALL_STEPS_SHARED]
        )
    resolved = []
    errors   = []
    for entry in raw:
        if '.' in entry:
            prefix, _, step = entry.partition('.')
            if prefix not in ('d', 'e'):
                errors.append(f"'{entry}': unknown prefix '{prefix}' (use 'd' or 'e')")
                continue
            if step == 'all':
                bucket = ALL_STEPS_D if prefix == 'd' else ALL_STEPS_E
                resolved.extend((prefix, s) for s in bucket)
            elif step in STEPS[prefix]:
                resolved.append((prefix, step))
            else:
                errors.append(
                    f"'{entry}': unknown step '{step}' for lang '{prefix}'. "
                    f"Available: {list(STEPS[prefix].keys())}"
                )
        else:
            # No prefix — must be a shared step
            if entry in STEPS['shared']:
                resolved.append(('shared', entry))
            else:
                errors.append(
                    f"'{entry}': no prefix given and not a shared step. "
                    f"Shared steps: {ALL_STEPS_SHARED}. "
                    f"Prefix with 'd.' or 'e.' for language-specific steps."
                )
    if errors:
        raise ValueError("Invalid steps in config:\n  " + "\n  ".join(errors))
    return resolved

def _banner(msg):
    width = 60
    print("\n" + "─" * width)
    print(f"  {msg}")
    print("─" * width)

#pipeline runner
def run_pipeline(cfg=None):
    if cfg is None:
        cfg = config
    steps = _resolve_steps(cfg.get('steps', 'all'))
    step_labels = [f"{lang}.{step}" if lang != 'shared' else step for lang, step in steps]
    _banner(f"Pipeline starting  |  steps: {step_labels}")
    for lang, step in steps:
        label = f"{lang}.{step}" if lang != 'shared' else step
        _banner(f"STEP: {label.upper()}")
        STEPS[lang][step](cfg)
        print(f"✓  {label} complete")
    _banner("Pipeline finished successfully! ")

if __name__ == "__main__": 
    run_pipeline()
