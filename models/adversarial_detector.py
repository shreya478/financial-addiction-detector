import pandas as pd
import numpy as np

class AdversarialEvasionDetector:
    """
    Rule-based detector to identify potentially anomalous or 'bot-like' 
    behavior designed to evade the addiction detection models.
    """
    
    def __init__(self, min_txns=10, variance_threshold=1e-2, timing_std_threshold=1.0):
        self.min_txns = min_txns
        self.variance_threshold = variance_threshold
        self.timing_std_threshold = timing_std_threshold
        
    def flag_users(self, transactions_df):
        """
        Takes raw transactions dataframe and returns a Series of user_ids mapped to boolean flags.
        """
        flags = {}
        
        # Sort and group
        tx_sorted = transactions_df.sort_values(['user_id', 'timestamp'])
        
        for user_id, user_tx in tx_sorted.groupby('user_id'):
            if len(user_tx) < self.min_txns:
                flags[user_id] = False
                continue
                
            # Amount variance
            amt_var = user_tx['amount'].var()
            
            # Timing standard deviation (in minutes)
            time_diffs = user_tx['timestamp'].diff().dt.total_seconds() / 60.0
            time_std = time_diffs.std()
            
            # Flag if both are suspiciously perfect (near zero)
            is_suspicious = False
            if pd.notna(amt_var) and amt_var < self.variance_threshold:
                if pd.notna(time_std) and time_std < self.timing_std_threshold:
                    is_suspicious = True
                    
            flags[user_id] = is_suspicious
            
        return pd.Series(flags)

if __name__ == '__main__':
    print("Testing Adversarial Detector...")
    tx = pd.read_csv('data/transactions.csv', parse_dates=['timestamp'])
    detector = AdversarialEvasionDetector()
    flags = detector.flag_users(tx)
    print(f"Total Evaders Detected: {flags.sum()} out of {len(flags)}")
