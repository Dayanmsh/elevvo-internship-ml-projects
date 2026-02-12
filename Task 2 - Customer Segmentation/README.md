# Task 2: Customer Segmentation

This project segments mall customers into groups based on their annual income and spending score using clustering algorithms.

## Dataset
- **Mall_Customers.csv** (not included in repo, download from Kaggle)
- Features: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

## Steps
1. **Exploratory Data Analysis (EDA):**
   - Visualize distributions and relationships
   - Check for missing values
2. **Preprocessing:**
   - Encode categorical variables
   - Scale features
3. **Clustering:**
   - KMeans (with optimal k selection using Elbow Method)
   - DBSCAN (bonus)
4. **Analysis:**
   - Visualize clusters
   - Analyze average spending per cluster
5. **Export:**
   - Save cluster assignments and summaries to CSV

## Bonus
- DBSCAN clustering
- Average spending analysis per cluster

## How to Run
1. Install requirements: `pip install -r ../../requirements.txt`
2. Place `Mall_Customers.csv` in this folder
3. Run `customer_segmentation.py` or open `Customer Segmentation.ipynb`

## Outputs
- `customer_segmentation_results.csv`: Clustered data
- `kmeans_cluster_summary.csv`, `dbscan_cluster_summary.csv`: Cluster summaries

## Author
[Your Name]

---
*For more details, see the notebook and report.*