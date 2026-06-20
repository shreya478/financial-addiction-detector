import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from slowapi import Limiter
from slowapi.util import get_remote_address
from models.state_machine import BehavioralStateMachine
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

limiter = Limiter(key_func=get_remote_address)

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

class MLModelCache:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModelCache, cls).__new__(cls)
        return cls._instance
        
    def initialize(self):
        if self._initialized: return
        print("Initializing ML Model Cache...")
        self.labels = pd.read_csv('data/labels.csv').set_index('user_id')
        self.transactions = pd.read_csv('data/transactions.csv', parse_dates=['timestamp'])
        
        checkpoint = torch.load('models/final_dp_model.pth', weights_only=False)
        self.feature_names = checkpoint['feature_names']
        
        self.model = RiskClassifier(input_dim=len(self.feature_names))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        X_train = checkpoint['X_train']
        background = torch.tensor(X_train[np.random.choice(X_train.shape[0], 100, replace=False)], dtype=torch.float32)
        self.explainer = shap.GradientExplainer(self.model, background)
        
        tx_agg = self.transactions.groupby('user_id').agg(
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
        
        self.tx_agg = tx_agg
        self.scaler = StandardScaler()
        self.scaler.fit(self.tx_agg[self.feature_names].values)
        
        self.state_machine = BehavioralStateMachine()
        self._initialized = True

ml_cache = MLModelCache()

def get_ml_cache():
    return ml_cache
