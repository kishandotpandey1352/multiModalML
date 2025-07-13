import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

# ---------- Utility: Pad/Trim Vector ----------
def normalize_vector_length(vec, length=128):
    if len(vec) > length:
        return vec[:length]
    elif len(vec) < length:
        return np.pad(vec, (0, length - len(vec)))
    return vec

# ---------- Text Dataset ----------
def vectorize_text(text_list, dim=128):
    vectorizer = TfidfVectorizer(max_features=dim)
    X = vectorizer.fit_transform(text_list)
    return normalize_vector_length(np.mean(X.toarray(), axis=0), dim)

# ---------- Tabular Dataset ----------
def vectorize_tabular(df, dim=128):
    df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    X = StandardScaler().fit_transform(df)
    return normalize_vector_length(np.mean(X, axis=0), dim)

# ---------- Genomics Dataset ----------
def vectorize_genomic(df, dim=128):
    df = df.select_dtypes(include=[np.number]).dropna(axis=1)
    dim = min(dim, df.shape[0], df.shape[1])
    pca = PCA(n_components=dim)
    return normalize_vector_length(np.mean(pca.fit_transform(df), axis=0), dim)

# ---------- Time Series Dataset ----------
def vectorize_time_series(df, id_col="id", time_col="time", value_col="value", dim=128):
    """
    Extracts time series features and returns a fixed-length vector of specified `dim`.
    Assumes `df` has columns: id, time, value
    """
    # Extract features
    features = extract_features(
        df,
        column_id=id_col,
        column_sort=time_col,
        column_value=value_col,
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True
    )

    # Impute missing values
    impute(features)

    # Convert to 1D vector
    vec = features.iloc[0].values  # Single time series expected per call
    return normalize_vector_length(vec, dim)

# ---------- Driver Function ----------
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
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        vectors[name] = normalize_vector_length(vec, dim)  # enforce consistency

    # Debugging step (optional)
    for k, v in vectors.items():
        print(f"{k}: {len(v)}")

    df = pd.DataFrame(vectors).T
    df.to_csv(output_csv)
    return df


data_dict = {
    "ehr_data": (pd.read_csv("datasets\\ehr.csv"), "tabular"),
    "clinical_notes": (["Patient stable.", "No fever.", "BP normal."], "text"),
    "genomics": (pd.read_csv("datasets\\snp_matrix.csv"), "genomic"),
    "heart_rate_series": (pd.read_csv("datasets\\time_series.csv"), "time_series")
}

if __name__ == "__main__":
    process_modalities(data_dict, dim=128, output_csv="modality_vectors.csv")
