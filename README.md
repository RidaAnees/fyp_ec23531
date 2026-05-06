# fyp_ec23531

This is a pipeline for training, aligning, and analysing Dhivehi and English word embeddings, with a focus on lexical gap detection between the two languages

---------------------------------------------------------------------------------------------------
## Setup
```bash
git clone https://github.com/RidaAnees/fyp_ec23531.git
cd fypNLP
conda create -n fypNLP python=3.9
conda activate fypNLP
pip install -r requirements.txt
git clone https://github.com/artetxem/vecmap vecmap
```

---------------------------------------------------------------------------------------------------
## Execution
All pipeline behaviour is controlled from `config.py`. Set the `steps` field to
control which steps run:

```python
'steps': ['d.preprocess', 'd.embed', 'e.preprocess', 'e.embed', 'align']
```

Then run:

```bash
python main.py
```
---------------------------------------------------------------------------------------------------
## Data
Due to difficulties uploading large files to GitHub, I have uploaded:
- All code
- All raw files needed
- Some trained models
- Best aligned model only
- Some results

- Before running the pipeline, create different directories and file names to not overwrite these sample files
- These sample files ensure testing without running whole pipeline yourself 

---------------------------------------------------------------------------------------------------
## Dependencies
- Python 3.9+
- gensim
- numpy
- pandas
- scipy
- scikit-learn
- datasets (HuggingFace)
- VecMap (included as submodule)

---------------------------------------------------------------------------------------------------
## Notes
- CUDA is supported for VecMap alignment — set `alignment_use_cuda: True` in config
- All pipeline behaviour controlled from `config.py` — do not edit individual scripts
- Seed dictionary contains ~536 valid Dhivehi–English pairs after vocab filtering


