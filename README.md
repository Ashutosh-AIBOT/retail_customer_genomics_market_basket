# Retail Customer Genomics & Market Basket (Project 12)

Unsupervised Customer Discovery Engine with clustering intelligence and market basket recommendations.

## What Is Implemented
- Dimensionality reduction:
  - PCA (2D/3D)
  - t-SNE
  - UMAP
- Clustering comparison:
  - K-Means
  - Agglomerative Clustering
  - Gaussian Mixture Models (GMM)
  - Spectral Clustering
- Recommendation engine:
  - Apriori
  - FP-Growth
- Evaluation:
  - Silhouette score
  - Calinski-Harabasz index
  - K-selection validation curves
- Business persona report generation
- Streamlit enterprise dashboard: `app.py`

## Folder Structure
- `src/data_loader.py`: Instacart table load and synthetic fallback
- `src/feature_engineering.py`: customer matrix + basket matrix
- `src/dimensionality.py`: PCA/t-SNE/UMAP embeddings
- `src/clustering.py`: clustering model suite
- `src/evaluation.py`: metrics + cluster curves
- `src/recommendation.py`: association-rule mining
- `src/persona.py`: persona report generator
- `src/train_pipeline.py`: full artifact generation flow

## Dataset Placement
Put Instacart CSV files in `data/raw/`:
- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `departments.csv`
- `aisles.csv`

If missing, synthetic fallback data is used.

## Run (Conda `ml-env`)
```bash
conda activate ml-env
cd 12_retail_customer_genomics_market_basket
pip install -r requirements.txt
python -m src.train_pipeline
streamlit run app.py
```

## Deployment
`render.yaml` is included for direct Render deployment.

# retail_customer_genomics_market_basket
