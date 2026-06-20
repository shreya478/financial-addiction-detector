import pandas as pd
import numpy as np
import os

def load_data():
    print("Loading data...")
    users = pd.read_csv('data/users.csv')
    sessions = pd.read_csv('data/sessions.csv', parse_dates=['start_time', 'end_time'])
    transactions = pd.read_csv('data/transactions.csv', parse_dates=['timestamp'])
    return users, sessions, transactions

def apply_labels(users, sessions, transactions):
    print("Applying labels...")
    
    # 1. Preoccupation: > 50 sessions total
    session_counts = sessions.groupby('user_id').size()
    preoccupation = (session_counts > 50).astype(int)
    
    # Pre-sort transactions
    tx = transactions.sort_values(['user_id', 'timestamp'])
    bets = tx[tx['transaction_type'] == 'bet'].copy()
    
    # 2. Tolerance
    # Calculate 30-bet rolling avg per user
    bets['rolling_avg'] = bets.groupby('user_id')['amount'].transform(lambda x: x.rolling(window=30, min_periods=10).mean())
    bets['spike'] = bets['amount'] > (bets['rolling_avg'] * 1.5)
    
    # Group by user to get % of spikes
    spike_rates = bets.groupby('user_id')['spike'].mean()
    tolerance = (spike_rates > 0.10).astype(int)
    
    # 3. Withdrawal: > 20% late night sessions and > 5 sessions total
    sessions['late_night'] = sessions['start_time'].dt.hour.between(0, 5)
    late_night_rates = sessions.groupby('user_id')['late_night'].mean()
    withdrawal = ((late_night_rates > 0.20) & (session_counts > 5)).astype(int)
    
    # 4. Loss of control: > 2 hrs and > 15 bets in a session
    sessions['duration_hrs'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds() / 3600
    long_sessions = sessions[sessions['duration_hrs'] > 2]
    bet_counts_per_session = bets.groupby('session_id').size()
    long_sessions = long_sessions.merge(bet_counts_per_session.rename('bet_count'), left_on='session_id', right_index=True, how='left')
    loss_control_users = long_sessions[long_sessions['bet_count'] > 15]['user_id'].unique()
    
    # 5. Loss chasing: same session, within 30 min, amount > 1.5x prev
    bets['prev_amount'] = bets.groupby('user_id')['amount'].shift(1)
    bets['prev_timestamp'] = bets.groupby('user_id')['timestamp'].shift(1)
    bets['prev_session'] = bets.groupby('user_id')['session_id'].shift(1)
    
    chase_mask = (
        (bets['session_id'] == bets['prev_session']) &
        ((bets['timestamp'] - bets['prev_timestamp']).dt.total_seconds() <= 1800) &
        (bets['amount'] > bets['prev_amount'] * 1.5)
    )
    loss_chasing_users = bets[chase_mask]['user_id'].unique()
    
    # 6. Lying: > 3 payment methods
    payment_counts = tx.groupby('user_id')['payment_method'].nunique()
    lying = (payment_counts > 3).astype(int)
    
    # 7. Escapism: > 40% tx during 9-5 and > 20 tx total
    tx['work_hours'] = tx['timestamp'].dt.hour.between(9, 17)
    work_rates = tx.groupby('user_id')['work_hours'].mean()
    tx_counts = tx.groupby('user_id').size()
    escapism = ((work_rates > 0.4) & (tx_counts > 20)).astype(int)
    
    # 8. Jeopardizing relationships: max bet > 5x avg bet and max bet > 100
    bet_stats = bets.groupby('user_id')['amount'].agg(['max', 'mean'])
    jeopardizing = ((bet_stats['max'] > bet_stats['mean'] * 5) & (bet_stats['max'] > 100)).astype(int)
    
    # 9. Bailout-seeking: any deposit with external_transfer=True
    bailout_users = tx[tx['external_transfer'] == True]['user_id'].unique()
    
    # Combine everything
    labels_df = pd.DataFrame({'user_id': users['user_id']}).set_index('user_id')
    
    labels_df['preoccupation_score'] = labels_df.index.map(preoccupation).fillna(0).astype(int)
    labels_df['tolerance_score'] = labels_df.index.map(tolerance).fillna(0).astype(int)
    labels_df['withdrawal_score'] = labels_df.index.map(withdrawal).fillna(0).astype(int)
    labels_df['loss_of_control_score'] = labels_df.index.isin(loss_control_users).astype(int)
    labels_df['loss_chasing_score'] = labels_df.index.isin(loss_chasing_users).astype(int)
    labels_df['lying_score'] = labels_df.index.map(lying).fillna(0).astype(int)
    labels_df['escapism_score'] = labels_df.index.map(escapism).fillna(0).astype(int)
    labels_df['jeopardizing_score'] = labels_df.index.map(jeopardizing).fillna(0).astype(int)
    labels_df['bailout_score'] = labels_df.index.isin(bailout_users).astype(int)
    
    labels_df['total_score'] = labels_df[['preoccupation_score', 'tolerance_score', 'withdrawal_score', 
                                          'loss_of_control_score', 'loss_chasing_score', 'lying_score', 
                                          'escapism_score', 'jeopardizing_score', 'bailout_score']].sum(axis=1)
                                          
    def assign_tier(score):
        if score >= 6: return 'High'
        if score >= 3: return 'Moderate'
        return 'Low'
        
    labels_df['risk_tier'] = labels_df['total_score'].apply(assign_tier)
    labels_df = labels_df.reset_index()
    
    os.makedirs('data', exist_ok=True)
    labels_df.to_csv('data/labels.csv', index=False)
    print("Labeling complete. Total High Risk users:", len(labels_df[labels_df['risk_tier'] == 'High']))
    return labels_df

if __name__ == '__main__':
    u, s, t = load_data()
    apply_labels(u, s, t)
