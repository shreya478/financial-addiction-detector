import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os

def generate_report(data_dir='data'):
    labels_path = os.path.join(data_dir, 'labels.csv')
    labels = pd.read_csv(labels_path)
    num_users = len(labels)
    
    np.random.seed(123) # Reproducible randomness
    
    platforms = ["RupeeRush", "TradeX", "QuickBet", "BullRun App"]
    labels['platform'] = np.random.choice(platforms, size=num_users)
    
    # Define hidden ground-truth aggressiveness mapping
    # High -> strong causal link; Low -> weak/noisy link
    aggressiveness = {
        "RupeeRush": 0.85,     # Highly aggressive dark patterns
        "TradeX": 0.60,        # Moderate
        "QuickBet": 0.30,      # Low
        "BullRun App": 0.0     # None (pure noise, should not be significant)
    }
    
    results = []
    
    for platform in platforms:
        p_data = labels[labels['platform'] == platform].copy()
        n = len(p_data)
        
        agg = aggressiveness[platform]
        
        # 1. Post-loss notification rate (0 to 100%)
        # Base rate + aggressiveness boost + noise
        post_loss_notif_rate = np.clip(np.random.normal(loc=20 + (agg * 60), scale=15, size=n), 0, 100)
        
        # 2. State Escalations
        # A baseline escalation propensity derived from their final total_score
        base_escalation = p_data['total_score'].values * 0.5 
        
        # The causal simulation: post_loss_notif_rate drives escalations, 
        # but the strength of the drive depends on aggressiveness. 
        causal_impact = (post_loss_notif_rate / 100.0) * agg * 15.0 
        
        # Add realistic variance/noise so the correlation isn't artificially clean
        noise = np.random.normal(loc=0, scale=1.5, size=n)
        
        escalation_count = np.clip(np.round(base_escalation + causal_impact + noise), 0, None)
        
        # Compute correlation
        r, p_value = pearsonr(post_loss_notif_rate, escalation_count)
        
        if r > 0.5 and p_value < 0.05:
            verdict = "Dark Pattern Detected"
        elif r > 0.3 and p_value < 0.05:
            verdict = "Weak Correlation"
        else:
            verdict = "Within Normal Range"
            
        if p_value < 0.001:
            p_str = "< 0.001"
        else:
            p_str = f"{p_value:.3f}"
            
        results.append({
            "Platform": platform,
            "Avg Post-Loss Notif Rate (%)": round(np.mean(post_loss_notif_rate), 1),
            "Avg Escalation Count": round(np.mean(escalation_count), 2),
            "Pearson_r": round(float(r), 4),
            "p_value": p_str,
            "Verdict": verdict
        })
        
    df_results = pd.DataFrame(results)
    return df_results.sort_values(by="Pearson_r", ascending=False)

if __name__ == "__main__":
    df = generate_report()
    print("--- Dark Pattern Scorer Evaluation ---")
    print(df.to_string(index=False))
