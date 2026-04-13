import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
print("Generating fashion e-commerce dataset...")
n = 5000
df = pd.DataFrame({
    'CustomerID': np.random.randint(1000, 6000, n),
    'InvoiceNo': [f'INV{i:05d}' for i in range(n)],
    'InvoiceDate': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, n), unit='D'),
    'Quantity': np.random.randint(1, 10, n),
    'UnitPrice': np.round(np.random.uniform(5, 200, n), 2)
})
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
print(f"Dataset ready: {df.shape[0]:,} rows, {df['CustomerID'].nunique()} customers")

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()
print(f"RFM table: {rfm.shape[0]:,} customers")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes, ['Recency','Frequency','Monetary'], ['#E91E8C','#9C27B0','#3F51B5']):
    ax.hist(rfm[col], bins=40, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f'{col} Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('rfm_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: rfm_distributions.png")

rfm_log = np.log1p(rfm[['Recency','Frequency','Monetary']])
rfm_scaled = StandardScaler().fit_transform(rfm_log)

inertias, silhouettes = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, labels))
    print(f"K={k} | Silhouette={silhouettes[-1]:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(2,9), inertias, 'o-', color='#E91E8C', linewidth=2)
ax1.set_title('Elbow Method', fontweight='bold')
ax1.set_xlabel('K'); ax1.set_ylabel('Inertia')
ax2.plot(range(2,9), silhouettes, 's-', color='#9C27B0', linewidth=2)
ax2.set_title('Silhouette Score', fontweight='bold')
ax2.set_xlabel('K')
plt.tight_layout()
plt.savefig('optimal_k.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: optimal_k.png")

km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = km_final.fit_predict(rfm_scaled)

cluster_summary = rfm.groupby('Cluster').agg(
    Customers=('CustomerID','count'),
    Avg_Recency=('Recency','mean'),
    Avg_Frequency=('Frequency','mean'),
    Avg_Monetary=('Monetary','mean')
).round(1)

segment_map = {
    cluster_summary['Avg_Monetary'].idxmax(): 'Loyal Champions',
    cluster_summary['Avg_Recency'].idxmin(): 'New / Promising',
    cluster_summary['Avg_Recency'].idxmax(): 'Lost / At-Risk',
}
for c in range(4):
    if c not in segment_map:
        segment_map[c] = 'Occasional Buyers'

cluster_summary['Segment'] = cluster_summary.index.map(segment_map)
rfm['Segment'] = rfm['Cluster'].map(segment_map)
print("\nCluster Profiles:")
print(cluster_summary[['Segment','Customers','Avg_Recency','Avg_Frequency','Avg_Monetary']])

colors_list = ['#E91E8C','#9C27B0','#3F51B5','#00BCD4']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
seg_counts = rfm['Segment'].value_counts()
axes[0].pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%', colors=colors_list, startangle=140)
axes[0].set_title('Customer Segments', fontweight='bold')
seg_monetary = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
axes[1].bar(range(len(seg_monetary)), seg_monetary.values, color=colors_list, edgecolor='white')
axes[1].set_xticks(range(len(seg_monetary)))
axes[1].set_xticklabels(seg_monetary.index, rotation=15, ha='right', fontsize=9)
axes[1].set_title('Avg Revenue per Segment', fontweight='bold')
plt.tight_layout()
plt.savefig('segment_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: segment_analysis.png")

heat_data = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(heat_norm, annot=heat_data.round(1), fmt='g', cmap='RdPu', linewidths=0.5, ax=ax)
ax.set_title('RFM Heatmap by Segment', fontweight='bold')
plt.tight_layout()
plt.savefig('rfm_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: rfm_heatmap.png")

rfm.to_excel('customer_segments.xlsx', index=False)
cluster_summary.to_excel('segment_summary.xlsx')
print("Saved Excel files")
print("\nProject Complete!")
