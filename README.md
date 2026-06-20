# Unsupervised Deep Representation Learning in *Artemisia annua*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Official code repository for the manuscript:  
> **"Unsupervised Deep Learning Reveals Hidden Chemotypes in Unannotated *Artemisia annua* Transcriptomes"**

## 📌 Overview
Traditional transcriptomic analyses rely heavily on supervised models requiring exhaustive metadata, creating a major bottleneck for utilizing massive public RNA-Seq repositories. This repository provides a fully unsupervised deep learning and clustering pipeline designed to autonomously map global physiological trajectories and discover hidden chemotypes directly from raw, disjointed agricultural data.

Applied to the antimalarial plant *Artemisia annua*, this pipeline successfully compressed a harmonized 39,975-dimensional feature space (n=317) into a 128-dimensional non-linear manifold, isolating a hyper-active Chromosome 4 gene cluster (anchored by mikado.chr4G1338) without human-provided annotations.

---

## 🧠 Model Architecture

*(Upload your Figure 1 image to the repository and link it here)*

![Universal Multi-Tissue Deep Autoencoder Architecture](figures/Figure1_Architecture.png)
> **Fig 1.** Schematic representation of the Universal Multi-Tissue Deep Autoencoder. The framework ingests a harmonized 39,975-dimensional transcriptomic vector, compressing it through sequentially regularized dense layers (512 → 256) into a 128-dimensional latent bottleneck. Batch Normalization and Dropout regularizers ensure robust, non-linear feature extraction independent of human metadata.

---

## ⚙️ Core Pipeline Features

1. **Multi-Tissue Harmonization:** Automatically ingests disjointed TPM expression matrices from distinct developmental tissues and calculates the exact mathematical intersection of loci to create a unified global feature space.
2. **Transcriptomic Preprocessing:** Applies log2(TPM + 1) transformations, threshold-based low-variance filtering, and Min-Max scaling to stabilize gradient descent optimization.
3. **Unsupervised Chemotype Discovery:** Utilizes Principal Component Analysis (PCA) and K-Means clustering (k=3) to autonomously isolate high-variance sub-populations.
4. **Rigorous Differential Expression:** Computes Log2 Fold Change and evaluates significance using Welch's independent t-tests, corrected via the Benjamini-Hochberg False Discovery Rate (FDR) protocol.
5. **Universal Deep Autoencoder:** A highly regularized, symmetric neural network (39k -> 512 -> 256 -> 128 bottleneck) optimized with Adam to extract fundamental biological grammar.
6. **Latent Manifold Projection:** Employs t-SNE to project the 128-dimensional deep features into a 2D interpretable visualization of the plant's physiological architecture.

---

## 📂 Repository Structure

├── data/                                 # Place your raw TPM expression files here
│   ├── Leaf_TPM_v2.txt
│   ├── Root_TPM_v2.txt
│   └── ...
├── figures/                              # Directory for architecture and generated plots
│   └── Figure1_Architecture.png          
├── results/                              # Output directory for generated CSVs
│   ├── differential_expression_results.csv
│   └── tsne_coordinates.csv
├── artemisia_pipeline.py                 # Core execution script
├── requirements.txt                      # Python dependencies
└── README.md


---

## 🚀 Installation & Setup

We recommend running this pipeline within a dedicated virtual environment (e.g., `conda` or `venv`) to prevent dependency conflicts.

**1. Clone the repository:**
git clone https://github.com/YourUsername/Artemisia-Deep-Learning.git
cd Artemisia-Deep-Learning

**2. Create and activate a virtual environment (Optional but recommended):**
python -m venv artemisia_env
source artemisia_env/bin/activate  # On Windows use: artemisia_env\Scripts\activate

**3. Install the required dependencies:**
pip install -r requirements.txt


---

## 💻 Usage & Execution

**1. Data Preparation:**
Ensure your raw, transposed transcriptomic datasets (where columns are genes and rows are samples) are placed in the root directory or the `data/` folder. The script expects tab-separated `.txt` files containing TPM values.

**2. Run the Pipeline:**
Execute the primary pipeline script. The script is designed with strict global random seed setting (seed=42) to guarantee exact mathematical reproducibility.

python artemisia_pipeline.py


**3. Expected Output:**
The terminal will output the harmonization logging, PCA variance captures, FDR adjustments, and the epoch-by-epoch loss metrics of the autoencoder. Upon completion, the pipeline generates two primary analytical files:
* `differential_expression_results.csv`: Contains Log2FC, raw p-values, and FDR-adjusted p-values for the isolated chemotypes.
* `tsne_coordinates.csv`: Contains the 2D projected coordinates derived from the deep autoencoder's 128-dimensional latent bottleneck.

---

## 📝 Citation
If you utilize this pipeline or the findings presented in our manuscript, please cite our work:

@article{YourLastName2026Artemisia,
  title={Unsupervised Deep Learning Reveals Hidden Chemotypes in Unannotated Artemisia annua Transcriptomes},
  author={Your Name and Co-authors},
  journal={Journal Name (e.g., Bioinformatics)},
  year={2026},
  volume={xx},
  number={xx},
  pages={xx-xx}
}

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
