"""
Artemisia annua Unsupervised Deep Representation Learning Pipeline.
Author: [Your Name/Lab]
Date: 2026

This script executes the data harmonization, clustering, differential expression, 
and deep autoencoder training detailed in the manuscript.
"""

import os
import random
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Suppress TensorFlow logging clutter for a clean console output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================================
# 0. REPRODUCIBILITY SETUP
# ==========================================
def set_global_seeds(seed=42):
    """Sets random seeds across all libraries to ensure strict reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[INFO] Global random seed set to {seed}")

# ==========================================
# 1. DATA HARMONIZATION & PREPROCESSING
# ==========================================
def load_and_harmonize(file_paths):
    """
    Loads multiple tissue TPM files, transposes them (samples as rows), 
    and returns the strict intersection of their feature spaces (genes).
    """
    print("[INFO] Harmonizing multi-tissue datasets...")
    dataframes = []
    labels = []
    
    for tissue, path in file_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', index_col=0).T
            dataframes.append(df)
            labels.extend([tissue] * len(df))
            print(f"  -> Loaded {tissue}: {df.shape[0]} samples, {df.shape[1]} genes")
        else:
            print(f"  -> WARNING: {path} not found. Skipping.")
            
    # Find strict mathematical intersection of all gene identifiers
    common_genes = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_genes.intersection_update(df.columns)
        
    common_genes = sorted(list(common_genes))
    print(f"[INFO] Global harmonized feature space: {len(common_genes)} conserved loci.")
    
    # Concatenate and filter to only common genes
    X_global = pd.concat([df[common_genes] for df in dataframes], axis=0)
    return X_global, np.array(labels)

def preprocess_transcriptomes(X_tpm, threshold=1.0):
    """
    Applies log2(TPM+1) transformation and removes low-variance noise loci.
    """
    print("[INFO] Preprocessing transcriptomic matrix...")
    X_log = np.log2(X_tpm + 1)
    
    # Low-variance filter: gene must exceed threshold in at least one sample
    mask = X_log.max(axis=0) > threshold
    X_filtered = X_log.loc[:, mask]
    
    print(f"[INFO] Filtered matrix shape: {X_filtered.shape} (Removed {X_log.shape[1] - X_filtered.shape[1]} noise loci)")
    return X_filtered

# ==========================================
# 2. UNSUPERVISED CLUSTERING
# ==========================================
def discover_subpopulations(X_leaf, n_clusters=3):
    """
    Performs PCA and K-Means clustering to autonomously isolate chemotypes.
    """
    print("[INFO] Executing PCA and K-Means Clustering on Leaf cohort...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_leaf)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  -> PC1 Variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    
    return cluster_labels, X_pca

# ==========================================
# 3. DIFFERENTIAL EXPRESSION (WITH FDR)
# ==========================================
def run_differential_expression(X_leaf, labels, target_cluster=0):
    """
    Calculates Log2FC, performs Welch's t-test, and applies Benjamini-Hochberg FDR correction.
    """
    print(f"[INFO] Running Differential Expression profiling for Cluster {target_cluster}...")
    
    island_1 = X_leaf[labels == target_cluster]
    island_2 = X_leaf[labels != target_cluster]
    
    log2_fc = island_1.mean() - island_2.mean()
    
    p_values = []
    genes = island_1.columns
    
    for gene in genes:
        # Welch's independent t-test (equal_var=False)
        stat, p = stats.ttest_ind(island_1[gene], island_2[gene], equal_var=False)
        p_values.append(p)
        
    p_values = np.array(p_values)
    p_values[np.isnan(p_values)] = 1.0 # Handle perfectly zero variance cases
    
    # Apply Benjamini-Hochberg False Discovery Rate (FDR) correction
    rejected, p_val_adj = fdrcorrection(p_values, alpha=0.05, method='indep')
    
    de_results = pd.DataFrame({
        'Gene_ID': genes,
        'Log2FC': log2_fc.values,
        'P_Value': p_values,
        'FDR_Adj_P': p_val_adj,
        'Significant': rejected
    }).sort_values(by='Log2FC', ascending=False)
    
    de_results.to_csv('differential_expression_results.csv', index=False)
    print("[INFO] DE results saved to 'differential_expression_results.csv'")
    return de_results

# ==========================================
# 4. DEEP AUTOENCODER ARCHITECTURE
# ==========================================
def build_and_train_autoencoder(X_filtered):
    """
    Constructs and trains the Universal Multi-Tissue Deep Autoencoder.
    Returns the trained encoder model and the training history.
    """
    print("[INFO] Architecting Universal Multi-Tissue Deep Autoencoder...")
    
    # Scale strictly to [0, 1] for the Sigmoid output layer
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_filtered)
    
    input_dim = X_scaled.shape[1]
    latent_dim = 128
    
    # --- ENCODER ---
    input_layer = Input(shape=(input_dim,))
    
    x = Dense(512, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # --- BOTTLENECK ---
    latent_layer = Dense(latent_dim, activation='relu', name='latent_bottleneck')(x)
    
    # --- DECODER ---
    x = Dense(256, activation='relu')(latent_layer)
    x = BatchNormalization()(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Sigmoid activation to map predictions back to [0, 1] scale
    output_layer = Dense(input_dim, activation='sigmoid')(x)
    
    # Compile Model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=latent_layer) # Detached encoder
    
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    print("[INFO] Commencing neural network backpropagation (40 epochs)...")
    history = autoencoder.fit(
        X_scaled, X_scaled,
        epochs=40,
        batch_size=16,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )
    
    return encoder, history, X_scaled

# ==========================================
# 5. LATENT MANIFOLD PROJECTION
# ==========================================
def extract_and_project_latent_space(encoder, X_scaled):
    """
    Extracts the 128D latent representation and projects it to 2D using t-SNE.
    """
    print("[INFO] Extracting latent manifold and computing t-SNE projection...")
    
    # Extract 128D deep features
    X_latent = encoder.predict(X_scaled, verbose=0)
    
    # Project to 2D space
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_latent)
    
    # Save latent coordinates
    pd.DataFrame(X_tsne, columns=['tSNE_1', 'tSNE_2']).to_csv('tsne_coordinates.csv', index=False)
    print("[INFO] Latent coordinates saved to 'tsne_coordinates.csv'")
    
    return X_tsne

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    set_global_seeds(42)
    
    # Define datasets (Update paths if your files are in a specific folder)
    file_paths = {
        'Leaf': 'Leaf_TPM_v2.txt',
        'Root': 'Root_TPM_v2.txt',
        'Flower': 'Flower_TPM_v2.txt',
        'Stem': 'Stem_TPM_v2.txt',
        'Shoot': 'Shoot_TPM_v2.txt',
        'Petiole': 'Petiole_TPM_v2.txt'
    }
    
    # 1. Harmonization & Preprocessing
    # Note: If you only have Leaf data right now, the code will just process Leaf.
    X_global_tpm, tissue_labels = load_and_harmonize(file_paths)
    X_global_filtered = preprocess_transcriptomes(X_global_tpm, threshold=1.0)
    
    # 2. Isolate Leaf cohort for unsupervised discovery
    leaf_mask = (tissue_labels == 'Leaf')
    X_leaf = X_global_filtered[leaf_mask]
    
    if len(X_leaf) > 0:
        # 3. PCA & K-Means
        cluster_labels, _ = discover_subpopulations(X_leaf, n_clusters=3)
        
        # 4. Differential Expression 
        # (Assuming Cluster 0 is the anomaly based on previous analysis)
        de_results = run_differential_expression(X_leaf, cluster_labels, target_cluster=0)
        
    # 5. Deep Autoencoder Global Representation Learning
    encoder, history, X_scaled = build_and_train_autoencoder(X_global_filtered)
    
    # 6. Manifold Projection
    X_tsne = extract_and_project_latent_space(encoder, X_scaled)
    
    print("\n[SUCCESS] Unsupervised Pipeline Execution Complete.")
