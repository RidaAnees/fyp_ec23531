
"""Train FastText or Word2Vec embeddings on an English corpus
Mirrors embed.py (Dhivehi) exactly; The only behavioural difference is that this step saves both a .model file
(for gensim-based downstream steps) and a .vec file (plain text word2vec format), since alignment tools like VecMap expect the latter.
"""
import os
import logging
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
logging.basicConfig (
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.WARNING,
)

#  Training callback
class _ProgressCallback(CallbackAny2Vec):
    def __init__(self, total_epochs: int):
        self.epoch = 0
        self.total = total_epochs
    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch {self.epoch}/{self.total}")

#  Corpus loader 
def _load_corpus(path: str, merge_sentences: int) -> list[list[str]]:
    """
    Identical to the Dhivehi loader — merge N consecutive lines into one
    pseudo-sentence for longer training contexts.
    """
    print(f"  Loading corpus: {path}")
    with open(path, 'r', encoding='utf-8') as fh:
        lines = [l.strip() for l in fh if l.strip()]
    sentences = []
    for i in range(0, len(lines), merge_sentences):
        block= lines[i: i + merge_sentences]
        tokens = ' '.join(block).split()
        if tokens:
            sentences.append(tokens)
    n_tokens = sum(len(s) for s in sentences)
    print(f"  Sentences (after merge×{merge_sentences}): {len(sentences):,}")
    print(f"  Total tokens : {n_tokens:,}")
    return sentences

#Trainer class
class EmbeddingTrainerEng:
    def __init__(self, vector_size=100, window=5, min_count=2,
                 workers=4, epochs=10, sg=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
    def train(self, corpus_path: str, model_output: str, vec_output: str,embedding_type: str, merge_sentences: int = 1 ):
        os.makedirs(os.path.dirname(model_output) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(vec_output)   or '.', exist_ok=True)
        sentences = _load_corpus(corpus_path, merge_sentences)
        callback= _ProgressCallback(self.epochs)
        common_kwargs = dict(
            sentences= sentences,
            vector_size = self.vector_size,
            window = self.window,
            min_count= self.min_count,
            workers = self.workers,
            sg = self.sg,
            epochs = self.epochs,
            callbacks = [callback],
        )
        etype = embedding_type.lower()
        if etype == 'fasttext':
            print("  Training FastText (subword-aware) …")
            model = FastText(**common_kwargs)
        elif etype == 'word2vec':
            print("  Training Word2Vec …")
            model = Word2Vec(**common_kwargs)
        else:
            raise ValueError(
                f"Unknown embedding_type: '{embedding_type}'. "
                "Use 'fasttext' or 'word2vec'."
            )
        print(f"  Vocabulary size : {len(model.wv):,}")
        # .model 
        model.save(model_output)
        print(f"  Saved .model    : {model_output}")
        # .vec 
        model.wv.save_word2vec_format(vec_output)
        print(f"  Saved .vec: {vec_output}")
        return model

# Pipeline step wrapper
def run_embedding_eng(cfg: dict) -> None:
    trainer = EmbeddingTrainerEng(
        vector_size = cfg['vector_size_eng'],
        window      = cfg['window_eng'],
        min_count   = cfg['min_count_eng'],
        workers     = cfg['workers_eng'],
        epochs      = cfg['epochs_eng'],
        sg          = cfg['sg_eng'],
    )
    trainer.train(
        corpus_path    = cfg['embedding_input_eng'],
        model_output   = cfg['eng_model_output'],
        vec_output     = cfg['eng_vec_output'],
        embedding_type = cfg['embedding_type_eng'],
        merge_sentences= cfg['merge_sentences_eng'],
    )
