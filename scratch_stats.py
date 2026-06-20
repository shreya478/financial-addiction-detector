import warnings
warnings.filterwarnings('ignore')

with open('clustering_analysis.py', 'r') as f:
    code = f.read()

# Prevent plotting from blocking execution
code = code.replace('plt.show()', 'pass')

ns = {}
# Execute the clustering analysis script
exec(code, ns)

user_features = ns['user_features']
labels = ns['labels']

# Merge the clusters with the original labels
merged = user_features.merge(labels.set_index('user_id')[['total_score']], left_index=True, right_index=True)

# Calculate stats
stats = merged.groupby('dbscan_cluster')['total_score'].agg(['mean', 'median', 'std', 'count'])
print('\n--- DSM-5 SCORE BY DBSCAN CLUSTER ---')
print(stats)
