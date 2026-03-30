import os

from dashboard_core import ProjectConfig, run_project_app


CONFIG = ProjectConfig(
    key="Project 12",
    title="Retail Customer Genomics & Market Basket",
    subtitle="Unsupervised customer discovery and recommendation intelligence",
    icon="🛒",
    domain="Retail analytics and personalization",
    objective=(
        "Discover behavioral customer segments and generate product affinity recommendations "
        "using Instacart transaction patterns."
    ),
    business_value=(
        "Improves campaign targeting, basket size, and personalization outcomes through robust "
        "cluster validation and association-rule recommendations."
    ),
    prediction_label="Segment / Recommendation",
    highlights=[
        "Dimensionality-reduction suite: PCA, t-SNE, and UMAP for customer pattern discovery.",
        "Clustering comparison across K-Means, Agglomerative, GMM, and Spectral methods.",
        "Association-rule engine with Apriori and FP-Growth for market basket intelligence.",
        "Evaluation with Silhouette and Calinski-Harabasz metrics plus model selection curves.",
        "Business persona reporting layer for strategy-ready customer segment narratives.",
    ],
)


run_project_app(CONFIG, os.path.dirname(__file__))
