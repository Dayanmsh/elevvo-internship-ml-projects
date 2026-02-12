# Task 2: Customer Segmentation - Report

## Objective
Segment mall customers into distinct groups based on annual income and spending score using unsupervised learning (clustering).

## Dataset
- **Mall_Customers.csv** (Kaggle)
- Features: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

## Methodology
1. **EDA:**
   - Visualized income vs. spending score, age, and gender distributions
   - Checked for missing values (none found)
2. **Preprocessing:**
   - Encoded Gender
   - Scaled features (Age, income, spending_score)
3. **Clustering:**
   - **KMeans:**
     - Used Elbow Method to select optimal k (k=5)
     - Visualized clusters
   - **DBSCAN (Bonus):**
     - Explored different eps values
     - Detected noise/outliers
4. **Analysis:**
   - Summarized cluster characteristics (mean age, income, spending score)
   - Analyzed average spending per cluster
5. **Export:**
   - Saved results and summaries to CSV

## Results
- KMeans identified 5 customer segments with distinct spending and income patterns
- DBSCAN found clusters and outliers, but with more noise
- Business insights: High-income, high-spending clusters are key targets; low-spending clusters may need different strategies

## Bonus
- DBSCAN clustering and comparison
- Spending analysis per cluster

## Files
- `Customer Segmentation.ipynb`: Full analysis and visualizations
- `customer_segmentation.py`: Script version
- `customer_segmentation_results.csv`: Clustered data
- `kmeans_cluster_summary.csv`, `dbscan_cluster_summary.csv`: Cluster summaries

## Recommendations
- Use cluster insights for targeted marketing
- Consider additional features for deeper segmentation

---
*Prepared for Elevvo Internship, February 2026*