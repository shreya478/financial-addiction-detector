import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()

def generate_data(num_users=5000, months=6):
    print("Generating users...")
    users = []
    archetypes = ['casual', 'frequent', 'compulsive']
    weights = [0.6, 0.3, 0.1]
    
    for i in range(1, num_users + 1):
        users.append({
            'user_id': i,
            'username': fake.unique.user_name(),
            'archetype': np.random.choice(archetypes, p=weights),
            'created_at': fake.date_time_between(start_date=f"-{months}M", end_date=f"now")
        })
    users_df = pd.DataFrame(users)
    
    print("Generating sessions and transactions...")
    sessions = []
    transactions = []
    session_id_counter = 1
    transaction_id_counter = 1
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)
    
    # Pre-generate random batches to speed up
    for _, user in users_df.iterrows():
        user_id = user['user_id']
        archetype = user['archetype']
        
        if archetype == 'casual':
            days_active = random.randint(2, 15)
        elif archetype == 'frequent':
            days_active = random.randint(30, 90)
        else: # compulsive
            days_active = random.randint(100, 180)
            
        # Select active days across the 6 month window
        active_dates = sorted([start_date + timedelta(days=random.randint(0, months*30)) for _ in range(days_active)])
        
        current_amount_base = 20.0
        
        for date in active_dates:
            session_start = date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
            
            if archetype == 'compulsive':
                session_duration = timedelta(minutes=random.randint(30, 300))
                num_tx = random.randint(10, 30)
                current_amount_base *= random.uniform(1.01, 1.05) # Escalating bets
            elif archetype == 'frequent':
                session_duration = timedelta(minutes=random.randint(10, 120))
                num_tx = random.randint(3, 10)
            else:
                session_duration = timedelta(minutes=random.randint(5, 30))
                num_tx = random.randint(1, 3)
                
            session_end = session_start + session_duration
            
            sessions.append({
                'session_id': session_id_counter,
                'user_id': user_id,
                'start_time': session_start,
                'end_time': session_end,
                'device_type': random.choice(['mobile', 'desktop', 'tablet'])
            })
            
            if num_tx > 0:
                tx_seconds = sorted([random.randint(0, max(1, int(session_duration.total_seconds()))) for _ in range(num_tx)])
                tx_times = [session_start + timedelta(seconds=sec) for sec in tx_seconds]
                
                for i, tx_time in enumerate(tx_times):
                    tx_type = random.choices(['bet', 'win', 'deposit', 'withdrawal'], weights=[0.6, 0.2, 0.15, 0.05])[0]
                    amount = round(random.uniform(current_amount_base * 0.5, current_amount_base * 1.5), 2)
                    
                    external_transfer = False
                    if tx_type == 'deposit':
                        if archetype == 'compulsive' and random.random() < 0.4:
                            external_transfer = True
                        elif random.random() < 0.05:
                            external_transfer = True
                            
                    if tx_type == 'bet' and archetype == 'compulsive':
                        if i > 0 and transactions[-1]['transaction_type'] == 'bet':
                            amount = round(transactions[-1]['amount'] * random.uniform(1.2, 2.0), 2)
                    
                    transactions.append({
                        'transaction_id': transaction_id_counter,
                        'user_id': user_id,
                        'session_id': session_id_counter,
                        'timestamp': tx_time,
                        'amount': amount,
                        'transaction_type': tx_type,
                        'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer', 'crypto']),
                        'external_transfer': external_transfer
                    })
                    transaction_id_counter += 1
            
            session_id_counter += 1
            
    sessions_df = pd.DataFrame(sessions)
    transactions_df = pd.DataFrame(transactions)
    
    print("Saving to CSV...")
    os.makedirs('data', exist_ok=True)
    users_df.to_csv('data/users.csv', index=False)
    sessions_df.to_csv('data/sessions.csv', index=False)
    transactions_df.to_csv('data/transactions.csv', index=False)
    print("Data generation complete.")
    
    return users_df, sessions_df, transactions_df

if __name__ == '__main__':
    generate_data()
