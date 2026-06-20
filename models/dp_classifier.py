import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import os
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
        if score <= 2: return 0
        if score <= 5: return 1
        if score <= 7: return 2
        return 3
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
    
    return X, y, df['user_id'].values, features

from imblearn.over_sampling import SMOTE

def train_dp_model(X_train, y_train, X_test, y_test, target_epsilon, delta=1e-5):
    # Apply SMOTE to training set
    counts = np.bincount(y_train)
    freq_count = counts[1]
    
    smote_dict = {
        0: counts[0],          
        1: counts[1],          
        2: counts[1],  # Oversample Compulsive to match Frequent
        3: counts[3]           
    }
    smote = SMOTE(sampling_strategy=smote_dict, random_state=42)
    X_tr_resampled, y_tr_resampled = smote.fit_resample(X_train, y_train)

    X_tr = torch.tensor(X_tr_resampled, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_resampled, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)
    
    dataset = TensorDataset(X_tr, y_tr)
    batch_size = 256
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = RiskClassifier(input_dim=X_train.shape[1], hidden_dim=32, num_classes=4)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # Class weights calculation on resampled data
    resampled_counts = np.bincount(y_tr_resampled)
    weights = 1.0 / resampled_counts
    weights = weights / np.sum(weights) * len(resampled_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    privacy_engine = PrivacyEngine()
    
    epochs = 20
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=1.0,
    )
    
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    epsilon = privacy_engine.get_epsilon(delta)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_te)
        preds = torch.argmax(test_outputs, dim=1).numpy()
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
    return model, acc, f1, epsilon

def run_experiment():
    print("Loading features...")
    X, y, user_ids, feature_names = load_data_and_features()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    targets = [1.0, 5.0, 10.0, 8.0]
    results = []
    
    for eps in targets:
        print(f"Training DP-SGD with target epsilon = {eps}...")
        model, acc, f1, actual_eps = train_dp_model(X_train, y_train, X_test, y_test, target_epsilon=eps)
        print(f"Result -> Actual Eps: {actual_eps:.2f}, Acc: {acc:.3f}, F1: {f1:.3f}")
        results.append({'target_eps': eps, 'actual_eps': actual_eps, 'accuracy': acc, 'f1_score': f1})
        
        if eps == 8.0:
            torch.save({
                'state_dict': model._module.state_dict(),
                'X_train': X_train,
                'X_test': X_test,
                'feature_names': feature_names
            }, 'models/final_dp_model.pth')
            
    df_res = pd.DataFrame(results)
    df_plot = df_res[df_res['target_eps'].isin([1.0, 5.0, 10.0])].sort_values('actual_eps')
    
    plt.figure(figsize=(8, 5))
    plt.plot(df_plot['actual_eps'], df_plot['accuracy'], marker='o', label='Accuracy')
    plt.plot(df_plot['actual_eps'], df_plot['f1_score'], marker='s', label='F1 Score')
    plt.title('Privacy-Utility Tradeoff (DP-SGD at \u03b4=1e-5)')
    plt.xlabel('Privacy Budget (Epsilon)')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig('data/privacy_tradeoff.png')
    print("Tradeoff plot saved to data/privacy_tradeoff.png")

if __name__ == '__main__':
    run_experiment()
