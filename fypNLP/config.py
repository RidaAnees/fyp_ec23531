
""" 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃             CRITICAL !!!!              ┃
┃           MAIN APPLICATION FILE        ┃
┃        DO NOT MODIFY UNLESS SURE       ┃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Central configuration for the Dhivehi NLP pipeline. Edit ONLY this file to control all pipeline behaviour.
After modifying this file, run main.py
ENSURE ALL PATHS ARE CORRECT, COMMENT OUT WHAT IS NOT NEEDED BEFORE RUNS 
"""
config = { """
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ┃             PIPELINE STEPS                                                      ┃
        ┃                                                                                 ┃          
        ┃   Language-prefixed (d. = Dhivehi, e. = English):                               ┃
        ┃           d.preprocess, d.transliterate, d.stem, d.embed, d.test_embed          ┃
        ┃            e.preprocess, e.stem, e.embed, e.test_embed, e.test_pretrained       ┃
        ┃                                                                                 ┃
        ┃   Shared (no prefix):                                                           ┃
        ┃          align, detect_loans, find_gaps, visualise                              ┃
        ┃                                                                                 ┃
        ┃   Shortcuts:                                                                    ┃
        ┃       'd.all' -> all Dhivehi steps                                              ┃
        ┃       'e.all' -> all English steps                                              ┃
        ┃       'all'   -> everything in order: d.all, e.all, shared                      ┃
        ┃   you can combine multiple steps individually aswell                            ┃
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ """
    # WRITE BELOW TO CONTROL EXECUTION FLOW: 
    'steps': ['find_gaps'],


############################################################################################################################################3
    # DHIVEHI PREPROCESSING
    # --------------------------------
    'raw_dhiv':   [
        'i_data/raw/div_news_2019_30K-sentences.txt',
        'i_data/raw/div_newscrawl_2015_300K-sentences.txt',
        'i_data/raw/div_wikipedia_2016_30K-sentences.txt',
        'i_data/raw/div_wikipedia_2021_30K-sentences.txt',
        'i_data/raw/div-mv_web_2015_1M-sentences.txt',
        'i_data/raw/div-mv_web_2016_1M-sentences.txt',
        'i_data/raw/dhiv_mix.txt'
        ],
    'clean_dhiv': "i_data/processed/dhiv_clean.txt", 
    'stem_input': "i_data/processed/dhiv_clean.txt",
    'stem_output': "i_data/processed/dhiv_stemmed.txt",
    # DHIVEHI TRANSLITERATION  (Thaana to Latin)
    'run_transliteration':    False,     # Set to True to enable
    'transliterate_output':   'i_data/processed/dhiv_latin.txt',
    # -----------------------------------
    # DHIVEHI EMBEDDINGS
    # ------------------------------------
    'embedding_input':  "i_data/processed/dhiv_clean.txt",  # ← SINGLE file
    'embedding_output_dhiv':"ii_embeddings/trained/dhiv_word2vec_sg_150d.model",
    'embedding_type_dhiv': 'word2vec', #or 'word2vec' (default: 'fasttext')
    
    'merge_sentences':  5,
    'vector_size':      150,
    'window':           5,
    'min_count':        3,
    'workers':          8,
    'epochs':           50,
    'sg':               0,
    
    'min_n': 2,
    'max_n': 8,
    'bucket': 2000000,
    #--------------------------------------------- 
    # DHIVEHI EMBEDDING TESTING
    # --------------------------------------------
    'test_mode':        'file',       # 'interactive' or 'file'
    'dhiv_test_embedding': 'ii_embeddings/trained/dhiv_word2vec_sg_150d.model',  

##########################################################################################################################
    # ------------------------------------
    # ENGLISH PREPROCESSING
    # ------------------------------------
    'raw_eng':   [
        "i_data/raw_eng/brown_corpus.txt",  
        "i_data/raw_eng/eng_newscrawl_2018_1M-sentences.txt",  
        "i_data/raw_eng/eng_wikipedia_2016_1M-sentences.txt",  
        "i_data/raw_eng/eng-uk_web_2002_1M-sentences.txt",  
        "i_data/raw_eng/eng_ccnews_25m.txt",    # comment out for small English corpus 
    ],
    'clean_eng':      "i_data/processed/eng_clean.txt",
   
    # -------------------------------------------
    # ENGLISH — EMBEDDINGS
    # --------------------------------------------
    'embedding_input_eng': "i_data/processed/eng_clean.txt",
    'embedding_type_eng':  'fasttext',  # mirrors Dhivehi default
    'merge_sentences_eng': 5,
    'vector_size_eng':     300,
    'window_eng':          5,
    'min_count_eng':       3,
    'workers_eng':         8,
    'epochs_eng':          50,
    'sg_eng':              0,
    'eng_model_output':    'ii_embeddings/trained/eng_large/eng_large_fasttext_cbow_300d.model',
    'eng_vec_output':      'ii_embeddings/trained/eng_large/eng_large_fasttext_cbow_300d.vec',  
    # -----------------------------------------------
    # ENGLISH — EMBEDDING TESTING(trained model)
    # ---------------------------------------------
    'test_mode_eng':        'file',
    'test_input_file_eng':  'ii_embeddings/trained/eng_fasttext_cbow_150d.model',
    'eval_coherence_file':  'ii_embeddings/eval/coherence_probes.txt',
    'eval_topn':            10,
    'eval_output_trained_e':  'ii_embeddings/eval/e_trained_eval_fasttext.csv',
    # # ENGLISH — PRETRAINED MODEL TESTING
    # 'eng_pretrained_model': 'ii_embeddings/pretrained/wiki-news-300d-1M.kv', 
    # 'eval_output_pretrained_e':'ii_embeddings/eval/e_pretrained_eval.csv',  
    
####################################################################################################################
    # ALIGNMENT
    # source_embedding: trained Dhivehi model
    # target_embedding:  English vectors

    'alignment_matrix': 'ii_embeddings/alignment/vecmap_semi.npy',  #vecmap_sup or vecmap_semi or vecmap_unsup or procrustes_W.npy
    'alignment_method':      'vecmap_semi',  # procrustes / vecmap_supervised / vecmap_semi / vecmap_unsupervised
    #'procrustes_mode':   'orthogonal',  # 'orthogonal' or 'linear'
    'embedding_preprocessing': 'center',  # Options: 'all', 'none', 'norm_only', 'center', 'center_whiten', 'center_whiten_renorm'
    'source_embedding':      'ii_embeddings/trained/dhiv_fasttext_sg/dhiv_fasttext_sg.model',
    'target_embedding':      'ii_embeddings/trained/eng_large/eng_large_fasttext_sg_300d.model',
    'seed_dictionary':       'i_data/bilingual/seed_dict.csv',
    'aligned_space_output':  'iii_alignment/aligned_final.kv',
    'use_transliteration':   False,
    'alignment_train_ratio': 0.8,
    'alignment_seed':        42,
    'vecmap_dir':            'vecmap', # path to cloned VecMap repo
    'alignment_use_cuda':    True,  # set True if you have CUDA and want GPU

#########################################################################################################################3
    #  ---------------------------------------
    # LOAN WORD DETECTION
    # phonetic_threshold : 0–1, lower = stricter (fewer detected)
    # ne_lists_dir  : folder containing proper noun word lists
    # -----------------------------------------
    'phonetic_threshold':  0.65,
    'ne_lists_dir':        'i_data/bilingual/ne_lists/',
    'loan_output':         'v_results/loan_words.csv',
    # -------------------------------------------------
    # GAP DETECTION
    # gap_threshold  : max cosine similarity to count as a gap
    # top_n_english  : how many English words to scan
    # --------------------------------------------------
    'gap_threshold':   0.35,
    'top_n_english':   10_000,
    'gap_output':      'v_results/lexical_gaps.csv',
    # ----------------------------------------------------
    # VISUALISATION
    # vis_words  : English words to highlight in the PCA plot
    # vis_output : folder to save plots
    'vis_words':  ['hello', 'water', 'house', 'internet', 'bank', 'phone'],
    'vis_output': 'vi_visuals/output/2',
}
