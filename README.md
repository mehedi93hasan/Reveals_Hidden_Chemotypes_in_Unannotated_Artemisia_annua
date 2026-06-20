# Artemisia annua Unsupervised Transcriptomic Pipeline

This repository contains the official code implementation for the manuscript:
**"Unsupervised Deep Learning Reveals Hidden Chemotypes in Unannotated Artemisia annua Transcriptomes"**

## Overview
This pipeline performs fully unsupervised deep representation learning and clustering on bulk RNA-Seq datasets to bypass the metadata bottleneck. It autonomously discovers hidden chemotypes and extracts tissue-specific non-linear gene networks using K-Means clustering and a Universal Multi-Tissue Deep Autoencoder.

## Features
1. **Multi-Tissue Harmonization:** Automatically intersects disjointed transcriptomic datasets to create a unified feature space.
2. **Unsupervised Sub-Population Clustering:** PCA and K-Means isolation of high-variance chemotypes.
3. **Differential Expression (DE):** Welch's t-test with Benjamini-Hochberg FDR correction.
4. **Deep Autoencoder:** 39,975 -> 128 dimensional bottleneck with strict Batch Normalization and Dropout regularizers.
5. **Latent Manifold Projection:** t-SNE dimensionality reduction of the neural network's latent space.

## Usage
1. Place your raw TPM expression files (e.g., `Leaf_TPM_v2.txt`, `Root_TPM_v2.txt`) in the `data/` directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the pipeline: `python artemisia_pipeline.py`
