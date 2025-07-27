import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from bytes_chunks import vectorize_file_bytes 

# Define modality files
file_dict = {
    "text_file": "../dataset/text/sample_1.txt",
    "tabular_file": "../dataset/table/C7-1.csv",
    "audio_file": "../dataset/audio/2_iO_0.0n_4m_.wav",
    "image_file": "../dataset/image/0ae4f8a60.jpg"
    # "video_file": "datasets/video/sample.mp4",
    # "simulation_file": "datasets/simulation/sample.tar.gz",
    # "graph_file": "datasets/graph/edges.txt"
}

# Generate vectors from raw bytes
vectors = {
    name: vectorize_file_bytes(path, dim=128)
    for name, path in file_dict.items()
}

df = pd.DataFrame(vectors).T
df.to_csv("byte_vectors.csv")

# Cosine similarity
similarity_matrix = pd.DataFrame(
    cosine_similarity(df),
    index=df.index,
    columns=df.index
)
similarity_matrix.to_csv("byte_similarity_matrix.csv")

# --- HEATMAP ---
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5, square=True, fmt=".2f")
plt.title("Byte-Based Similarity Heatmap (Single File)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("byte_similarity_heatmap.png")
plt.close()

# --- DENDROGRAM ---
distance_matrix = 1 - similarity_matrix
linkage_matrix = linkage(distance_matrix, method='average')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=similarity_matrix.index, leaf_rotation=45)
plt.title("Byte-Based Dendrogram (Single File)")
plt.ylabel("Distance (1 - Cosine Similarity)")
plt.tight_layout()
plt.savefig("byte_similarity_dendrogram.png")
plt.close()
