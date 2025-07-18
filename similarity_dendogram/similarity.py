import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

# Load the modality vectors
file_path = "modality_vectors.csv"
df = pd.read_csv(file_path, index_col=0)

# Compute cosine similarity matrix
similarity_matrix = pd.DataFrame(
    cosine_similarity(df),
    index=df.index,
    columns=df.index
)

# Save similarity matrix
similarity_matrix.to_csv("modality_similarity_matrix.csv")

# --- HEATMAP ---
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', linewidths=0.5, square=True, fmt=".2f")
plt.title("Modality Similarity Heatmap (Cosine Similarity)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("modality_similarity_heatmap.png")
plt.close()

# --- DENDROGRAM ---
distance_matrix = 1 - similarity_matrix
linkage_matrix = linkage(distance_matrix, method='average')

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=similarity_matrix.index, leaf_rotation=45)
plt.title("Modality Clustering Dendrogram (Based on Cosine Distance)")
plt.ylabel("Distance (1 - Cosine Similarity)")
plt.tight_layout()
plt.savefig("modality_similarity_dendrogram.png") 
plt.close()
