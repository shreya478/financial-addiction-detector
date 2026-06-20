import pandas as pd
from sklearn.model_selection import train_test_split

labels = pd.read_csv('data/labels.csv')

def map_target(score):
    if score <= 2: return 0 # Casual
    if score <= 5: return 1 # Frequent
    if score <= 7: return 2 # Compulsive
    return 3 # Crisis

labels['target'] = labels['total_score'].apply(map_target)

y = labels['target'].values
# Dummy X just for the split
X = labels[['user_id']] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tiers = {0: 'Casual (0-2)', 1: 'Frequent (3-5)', 2: 'Compulsive (6-7)', 3: 'Crisis (8-9)'}

print("--- TRAINING SET DISTRIBUTION ---")
train_counts = pd.Series(y_train).value_counts().sort_index()
for tier_idx in range(4):
    count = train_counts.get(tier_idx, 0)
    print(f"{tiers[tier_idx]:<18} : {count}")

print("\n--- TEST SET DISTRIBUTION ---")
test_counts = pd.Series(y_test).value_counts().sort_index()
for tier_idx in range(4):
    count = test_counts.get(tier_idx, 0)
    print(f"{tiers[tier_idx]:<18} : {count}")
