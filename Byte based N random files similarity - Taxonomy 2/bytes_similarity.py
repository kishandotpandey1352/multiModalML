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
N_RUNS = 20
VECTOR_DIM = 128

# --- HELPER: Pick a random file from folder ---
def get_random_file(path):
    candidates = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return os.path.join(path, random.choice(candidates))

# --- MAIN LOOP ---
similarity_matrices = []

for run in range(N_RUNS):
    print(f"Run {run + 1}/{N_RUNS}...")

    vectors = {}
    for modality, folder in MODALITY_FOLDERS.items():
        file_path = get_random_file(folder)
        vectors[modality] = vectorize_file_bytes(file_path, dim=VECTOR_DIM)

    df = pd.DataFrame(vectors).T

    sim_matrix = pd.DataFrame(
        cosine_similarity(df),
        index=df.index,
        columns=df.index
    )
    similarity_matrices.append(sim_matrix)

# --- AVERAGE SIMILARITY MATRIX ---
avg_sim_matrix = sum(similarity_matrices) / N_RUNS
avg_sim_matrix.to_csv("byte_similarity_matrix_avg.csv")

# --- HEATMAP ---
plt.figure(figsize=(10, 8))
sns.heatmap(avg_sim_matrix, annot=True, cmap='coolwarm', linewidths=0.5, square=True, fmt=".2f")
plt.title(f"Byte-Based Similarity Heatmap (Average of {N_RUNS} Runs)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("byte_similarity_heatmap_avg.png")
plt.close()

# --- DENDROGRAM ---
distance_matrix = 1 - avg_sim_matrix
linkage_matrix = linkage(distance_matrix, method='average')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=avg_sim_matrix.index, leaf_rotation=45)
plt.title(f"Byte-Based Dendrogram (Average of {N_RUNS} Runs)")
plt.ylabel("Distance (1 - Cosine Similarity)")
plt.tight_layout()
plt.savefig("byte_similarity_dendrogram_avg.png")
plt.close()
