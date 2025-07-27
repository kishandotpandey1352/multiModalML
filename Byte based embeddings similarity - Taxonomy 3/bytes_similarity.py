import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from bytes_chunks import vectorize_file_bytes

# --- SETTINGS ---
MODALITY_FOLDERS = {
    "text": "../dataset/text",
    "tabular": "../dataset/table",
    "audio": "../dataset/audio",
    "image": "../dataset/image",
}
FILES_PER_MODALITY = 20 # Number of files to sample from each modality
VECTOR_DIM = 128

# --- HELPERS ---
def get_random_files(path, n=5):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if len(files) < n:
        print(f"Only found {len(files)} files in {path}")
        n = len(files)
    return [os.path.join(path, f) for f in random.sample(files, n)]

# --- MAIN: Compute average embedding per modality ---
modality_embeddings = {}

for modality, folder in MODALITY_FOLDERS.items():
    file_paths = get_random_files(folder, FILES_PER_MODALITY)
    vectors = []
    for path in file_paths:
        vec = vectorize_file_bytes(path, dim=VECTOR_DIM)
        vectors.append(vec)
    modality_embedding = np.mean(vectors, axis=0)
    modality_embeddings[modality] = modality_embedding
    print(f"✓ {modality}: Averaged over {len(vectors)} files")

# --- Cosine Similarity between modalities ---
df = pd.DataFrame(modality_embeddings).T
df.to_csv("modality_embeddings.csv")

sim_matrix = pd.DataFrame(
    cosine_similarity(df),
    index=df.index,
    columns=df.index
)
sim_matrix.to_csv("modality_similarity_matrix.csv")

# --- HEATMAP ---
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, annot=True, cmap='coolwarm', linewidths=0.5, square=True, fmt=".2f")
plt.title("Modality Embedding Similarity Heatmap")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("modality_embedding_heatmap.png")
plt.close()

# --- DENDROGRAM ---
distance_matrix = 1 - sim_matrix
linkage_matrix = linkage(distance_matrix, method='average')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=sim_matrix.index, leaf_rotation=45)
plt.title("Modality Clustering Dendrogram (Avg Embedding)")
plt.ylabel("Distance (1 - Cosine Similarity)")
plt.tight_layout()
plt.savefig("modality_embedding_dendrogram.png")
plt.close()
