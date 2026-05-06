
"""
Train FastText or Word2Vec embeddings on a Dhivehi corpus
"""
import os
import logging
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.WARNING,   # suppress gensim's verbose INFO
)
from config import config 
print("\n ##################################################################" )
print(f"workers: {config.get('workers')}")
print(f"CPU available: {os.cpu_count()}")
# Training callback
class _ProgressCallback(CallbackAny2Vec):
    def __init__(self, total_epochs: int):
        self.epoch       = 0
        self.total       = total_epochs
    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch {self.epoch}/{self.total}")

#Corpus loader 
def _load_corpus(path: str, merge_sentences: int) -> list[list[str]]:
    print(f"  Loading corpus: {path}")
    with open(path, 'r', encoding='utf-8') as fh:
        lines = [l.strip() for l in fh if l.strip()]
    sentences = []
    for i in range(0, len(lines), merge_sentences):
        block  = lines[i: i + merge_sentences]
        tokens = ' '.join(block).split()
        if tokens:
            sentences.append(tokens)
    n_tokens = sum(len(s) for s in sentences)
    print(f"  Sentences (after merge×{merge_sentences}): {len(sentences):,}")
    print(f"  Total tokens : {n_tokens:,}")
    return sentences

#Trainer class
class EmbeddingTrainer:
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4, epochs=10, sg=1, min_n=3, max_n=6, bucket=2000000):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.min_n= min_n
        self.max_n = max_n
        self.bucket = bucket
    def train(self, corpus_path: str, output_path: str, embedding_type: str, merge_sentences: int = 1):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        sentences = _load_corpus(corpus_path, merge_sentences)
        callback  = _ProgressCallback(self.epochs)
        common_kwargs = dict(
            sentences = sentences,
            vector_size = self.vector_size,
            window = self.window,
            min_count= self.min_count,
            workers= self.workers,
            sg = self.sg,
            epochs = self.epochs,
            callbacks = [callback],
        )
        etype = embedding_type.lower()
        if etype == 'fasttext':
            print("  Training FastText (subword-aware) …")
            print(f"    min_n={self.min_n}, max_n={self.max_n}, bucket={self.bucket}")
            model = FastText(
                **common_kwargs,
                min_n=self.min_n,
                max_n=self.max_n,
                bucket=self.bucket,
            )
        elif etype == 'word2vec':
            print ("  Training Word2Vec …")
            model = Word2Vec(**common_kwargs)
        else:
            raise ValueError(
                f"Unknown embedding_type: '{embedding_type}'. "
                "Use 'fasttext' or 'word2vec'."
            )
        print (f"  Vocabulary size : {len(model.wv):,}")
        model.save(output_path)
        print (f"  Saved to : {output_path}")
        return model

# Pipeline step 
def run_embedding(cfg: dict) -> None:
    trainer = EmbeddingTrainer(
        vector_size = cfg['vector_size'],
        window  = cfg['window'],
        min_count = cfg['min_count'],
        workers= cfg['workers'],
        epochs = cfg['epochs'],
        sg = cfg['sg'],
        min_n = cfg.get('min_n'),
        max_n  = cfg.get('max_n'),
        bucket = cfg.get('bucket'),
    )
    trainer.train(
        corpus_path = cfg['embedding_input'],
        output_path =  cfg['embedding_output_dhiv'],
        embedding_type = cfg['embedding_type_dhiv'],
        merge_sentences= cfg['merge_sentences'],
    )
