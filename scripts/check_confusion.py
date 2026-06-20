import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from opacus import PrivacyEngine
import warnings
warnings.filterwarnings('ignore')

class RiskClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_classes=4):
        super(RiskClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def load_data_and_features():
    labels = pd.read_csv('data/labels.csv')
    def map_target(score):
        if score <= 2: return 0 # Casual
        if score <= 5: return 1 # Frequent
        if score <= 7: return 2 # Compulsive
        return 3 # Crisis
    labels['target'] = labels['total_score'].apply(map_target)
    
    transactions = pd.read_csv('data/transactions.csv', parse_dates=['timestamp'])
    
    tx_agg = transactions.groupby('user_id').agg(
        txn_count=('transaction_id', 'count'),
        amount_var=('amount', 'var'),
        weekend_sum=('timestamp', lambda x: x.dt.dayofweek.isin([5, 6]).sum()),
        night_sum=('timestamp', lambda x: x.dt.hour.between(0, 5).sum()),
        is_deposit_sum=('transaction_type', lambda x: (x == 'deposit').sum()),
        is_bet_sum=('transaction_type', lambda x: (x == 'bet').sum())
    )
    
    tx_agg['weekend_skew'] = tx_agg['weekend_sum'] / tx_agg['txn_count']
    tx_agg['night_ratio'] = tx_agg['night_sum'] / tx_agg['txn_count']
    tx_agg['income_spend'] = tx_agg['is_deposit_sum'] / (tx_agg['is_bet_sum'] + 1e-5)
    tx_agg['amount_var'] = tx_agg['amount_var'].fillna(0)
    
    df = tx_agg.merge(labels[['user_id', 'target']], on='user_id')
    
    features = ['txn_count', 'amount_var', 'weekend_skew', 'night_ratio', 'income_spend']
    X = df[features].values
    y = df['target'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def train_eval(X_train, y_train, X_test, y_test, target_epsilon):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    
    dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = RiskClassifier(input_dim=X_train.shape[1], hidden_dim=32, num_classes=4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model, optimizer=optimizer, data_loader=train_loader,
        epochs=20, target_epsilon=target_epsilon, target_delta=1e-5, max_grad_norm=1.0,
    )
    
    model.train()
    for _ in range(20):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_te), dim=1).numpy()
        
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2, 3])
    return cm

X, y = load_data_and_features()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cm = train_eval(X_train, y_train, X_test, y_test, 10.0)

tiers = ['Casual', 'Frequent', 'Compulsive', 'Crisis']

print("--- CONFUSION MATRIX (Epsilon = 10.0) ---")
print(f"{'True / Pred':<15} | {'Casual':<10} | {'Frequent':<10} | {'Compulsive':<10} | {'Crisis':<10}")
print("-" * 65)
for i in range(4):
    print(f"{tiers[i]:<15} | {cm[i][0]:<10} | {cm[i][1]:<10} | {cm[i][2]:<10} | {cm[i][3]:<10}")
