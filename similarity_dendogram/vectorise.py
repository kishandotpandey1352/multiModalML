import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

import librosa
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import networkx as nx
from node2vec import Node2Vec
import tarfile
import tempfile

# ---------- Utility ----------
def normalize_vector_length(vec, length=128):
    if len(vec) > length:
        return vec[:length]
    elif len(vec) < length:
        return np.pad(vec, (0, length - len(vec)))
    return vec

# ---------- Text ----------
def vectorize_text(text_list, dim=128):
    vectorizer = TfidfVectorizer(max_features=dim)
    X = vectorizer.fit_transform(text_list)
    return normalize_vector_length(np.mean(X.toarray(), axis=0), dim)

# ---------- Tabular ----------
def vectorize_tabular(df, dim=128):
    df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    X = StandardScaler().fit_transform(df)
    return normalize_vector_length(np.mean(X, axis=0), dim)

# ---------- Genomic ----------
def vectorize_genomic(df, dim=128):
    df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    pca = PCA(n_components=min(dim, df.shape[1]))
    return normalize_vector_length(np.mean(pca.fit_transform(df), axis=0), dim)

# ---------- Time Series ----------
def vectorize_time_series(df, dim=128):
    # Convert 'datetime' to a numeric time index (if needed)
    df = df.copy()
    df["time"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**9
    df.drop(columns=["datetime"], inplace=True)

    # Melt the dataframe into long format
    df = df.melt(id_vars=["time"], var_name="id", value_name="value")

    # Drop non-numeric or NaN
    df = df[pd.to_numeric(df["value"], errors="coerce").notna()]
    df["value"] = df["value"].astype(float)

    if df.empty:
        print("Time-series input is empty after filtering. Returning zero vector.")
        return np.zeros(dim)

    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True
    )

    if features.empty:
        print("TSFRESH returned empty features. Returning zero vector.")
        return np.zeros(dim)

    impute(features)
    vec = features.iloc[0].values
    return normalize_vector_length(vec, dim)

# ---------- Audio ----------
def vectorize_audio(file_path, dim=128):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim)
    return normalize_vector_length(np.mean(mfcc, axis=1), dim)

# ---------- Image ----------
resnet = resnet18(pretrained=True)
resnet.eval()
resnet_layer = torch.nn.Sequential(*list(resnet.children())[:-1])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
def vectorize_image(image_path, dim=128):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        vec = resnet_layer(tensor).squeeze().numpy()
    return normalize_vector_length(vec, dim)

# ---------- Video ----------
def vectorize_video(video_path, dim=128):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < 5:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = resnet_layer(tensor).squeeze().numpy()
        frames.append(vec)
    cap.release()
    return normalize_vector_length(np.mean(frames, axis=0), dim)

# ---------- Graph ----------
def vectorize_graph(edge_list_path, dim=128):
    G = nx.read_edgelist(edge_list_path)
    node2vec = Node2Vec(G, dimensions=dim, quiet=True)
    model = node2vec.fit()
    vectors = [model.wv[n] for n in G.nodes if n in model.wv]
    return normalize_vector_length(np.mean(vectors, axis=0), dim)

def vectorize_simulation(tar_path, dim=128):
    with tempfile.TemporaryDirectory() as tempdir:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=tempdir)

        all_arrays = []

        for root, _, files in os.walk(tempdir):
            for file in files:
                path = os.path.join(root, file)
                try:
                    if file.endswith(".npy"):
                        arr = np.load(path)
                    elif file.endswith(".npz"):
                        npz = np.load(path)
                        arr = np.concatenate([npz[k] for k in npz.files if npz[k].ndim == 1])
                    elif file.endswith(".bin"):
                        arr = np.fromfile(path, dtype=np.float32)
                    elif file.endswith(".txt"):
                        arr = np.loadtxt(path)
                    else:
                        continue

                    if arr.ndim == 1:
                        all_arrays.append(arr)
                    elif arr.ndim == 2:
                        all_arrays.extend([arr[:, i] for i in range(arr.shape[1])])
                except Exception as e:
                    print(f"⚠️ Skipped {file}: {e}")

        if not all_arrays:
            print("⚠️ No usable numeric data found. Returning zero vector.")
            return np.zeros(dim)

        combined = np.vstack([a[:dim] if len(a) >= dim else np.pad(a, (0, dim - len(a))) for a in all_arrays])
        avg_vector = np.mean(combined, axis=0)
        return avg_vector

# ---------- Main Driver ----------
def process_modalities(data_dict, dim=128, output_csv="modality_vectors.csv"):
    vectors = {}
    for name, (data, modality) in data_dict.items():
        if modality == 'text':
            vec = vectorize_text(data, dim)
        elif modality == 'tabular':
            vec = vectorize_tabular(data, dim)
        elif modality == 'genomic':
            vec = vectorize_genomic(data, dim)
        elif modality == 'time_series':
            vec = vectorize_time_series(data, dim=dim)
        elif modality == 'audio':
            vec = vectorize_audio(data, dim)
        elif modality == 'image':
            vec = vectorize_image(data, dim)
        elif modality == 'video':
            vec = vectorize_video(data, dim)
        elif modality == 'simulation':
            vec = vectorize_simulation(data, dim)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        vectors[name] = vec
        print(f"{name}: Vector shape = {vec.shape}")

    df = pd.DataFrame(vectors).T
    df.to_csv(output_csv)
    return df

# Example usage
if __name__ == "__main__":
    with open("datasets/Text/Ammonia removal and disinfection in fish tank water - real/20191023-5mM NaCl 0mgL-1 NH3-N E coli/Gamry/20191023-1-EIS.DTA", "r") as f:
        text_data = [f.read()]

    data_dict = {
        "text_dataset": (text_data, "text"),
        "tabular_dataset": (pd.read_csv("datasets/Tabular/1.csv"), "tabular"),
        "sensor_series": (pd.read_csv("datasets/time-series/Skoltech Anomaly Benchmark/anomaly-free.csv", sep=";"), "time_series"),
        "audio_clip": ("datasets/audio/file_example_WAV_10MG.wav", "audio"),
        "image_sample": ("datasets/images/Lemon_dataset/0037_G_I_120_A.jpg", "image"),
        "simulation": ("datasets/simulation/T20-X-V7-5-W70.tar.gz", "simulation"),
        "video": ("datasets/video/sample-30s.mp4", "video"),
    }

    process_modalities(data_dict, dim=128)
