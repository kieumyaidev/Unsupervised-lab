# Lab 2: Unsupervised Learning - Fashion-MNIST Clustering

## Project Overview

This project implements unsupervised clustering of Fashion-MNIST images using autoencoders. The goal is to discover natural groupings of similar fashion items **without using any labels** during training.

## Problem Statement

The task involves clustering 70,000 fashion images into meaningful groups using unsupervised learning approaches:
- **No labels during training** - completely unsupervised
- **10 fashion categories**: T-shirts, trousers, dresses, coats, sandals, shirts, sneakers, bags, ankle boots, pullovers
- **Goal**: Discover visual similarities and group similar items together

This problem has real-world applications in e-commerce product categorization, inventory management, recommendation systems, and trend analysis.

## Dataset

**Fashion-MNIST Dataset** from Zalando Research
- **Source**: Available through PyTorch's `torchvision.datasets.FashionMNIST`
- **Images**: 70,000 grayscale images (28×28 pixels)
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Classes**: 10 fashion categories (perfectly balanced)
- **Format**: Automatically downloaded via PyTorch

## Methodology

### Approaches Implemented
1. **Autoencoder (AE) + K-Means Clustering**
   - Standard autoencoder with encoder-decoder architecture
   - Latent dimension: 16, 32, or 64
   - K-Means clustering on latent representations

2. **Variational Autoencoder (VAE) + Gaussian Mixture Model (GMM)**
   - Probabilistic autoencoder with reparameterization trick
   - Gaussian Mixture Model for clustering
   - Better-structured latent space for clustering

### Evaluation Metrics
- **Normalized Mutual Information (NMI)**: Measures alignment between clusters and true labels (0-1 scale)
- **Adjusted Rand Index (ARI)**: Evaluates clustering similarity, adjusted for chance (0-1 scale)
- **Silhouette Score**: Assesses cluster separation and cohesion (-1 to 1 scale)

### Key Features
- Comprehensive EDA with visualizations
- Hyperparameter tuning (latent dimensions, learning rates)
- Latent space visualization
- Reconstruction quality analysis

## Results

| Approach | NMI | ARI | Silhouette Score |
|----------|-----|-----|------------------|
| **VAE + GMM** | **0.5713** | **0.4065** | 0.1990 |
| AE + KMeans | 0.5403 | 0.3567 | 0.2388 |

**Key Findings:**
- VAE + GMM performed best, demonstrating that probabilistic latent spaces work well with probabilistic clustering methods
- NMI of 0.57 means clusters capture 57% of information in true labels - good for completely unsupervised learning
- Results are competitive with published unsupervised clustering results on Fashion-MNIST (typically 0.50-0.65 NMI)

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- PyTorch

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Dataset Setup
- Fashion-MNIST is automatically downloaded when you run the notebook
- No manual download required

### Running the Notebook
1. Open `lab_2.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The notebook includes:
   - Data loading and preprocessing
   - Exploratory Data Analysis (EDA)
   - Autoencoder and VAE implementation
   - Clustering algorithms
   - Results visualization and analysis
   - Discussion and future improvements

## Project Structure

```
Lab_2/
├── README.md
├── lab_2.ipynb          # Main notebook
└── requirements.txt      # Python dependencies
```

## Key Concepts

- **Autoencoder (AE)**: Neural network for learning compressed representations
- **Variational Autoencoder (VAE)**: Probabilistic version with latent distributions
- **K-Means Clustering**: Partition-based clustering algorithm
- **Gaussian Mixture Model (GMM)**: Probabilistic clustering model
- **Latent Space**: Lower-dimensional representation learned by autoencoders
- **Reparameterization Trick**: VAE technique for backpropagation through random sampling

## References

- **Dataset**: Fashion-MNIST - Available through PyTorch's `torchvision.datasets.FashionMNIST`
- **Key Concepts**: CU Boulder's Unsupervised Learning Course Note

## Author

Completed as part of the Unsupervised Learning Final Project.

