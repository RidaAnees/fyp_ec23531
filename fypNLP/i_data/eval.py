
import os
os.makedirs('ii_embeddings/eval', exist_ok=True)
probes = [
    'family', 'money', 'government', 'science', 'technology',
    'food', 'animal', 'sport', 'music', 'religion',
    'education', 'health', 'politics', 'nature', 'business',
    'war', 'peace', 'love', 'death', 'time',
    'water', 'fire', 'earth', 'air', 'light',
]
with open('ii_embeddings/eval/coherence_probes.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(probes))
print("Done.")
