import pandas as pd
import numpy as np

class BehavioralStateMachine:
    """
    Evaluates behavioral escalation states based on DSM-5 labels, 
    DBSCAN clusters, and Isolation Forest weekly overrides.
    """
    
    STATES = {
        0: 'Casual',
        1: 'Frequent',
        2: 'Compulsive',
        3: 'Crisis',
        4: 'Review (Anomaly)'
    }
        
    @staticmethod
    def map_base_state(dbscan_cluster, proxy_score):
        """
        Maps users to an initial baseline state based on DBSCAN clustering 
        and DSM-5 proxy scores.
        
        - Cluster 1 (Compulsive/Crisis): Split based on proxy_score.
        - Cluster 0 (Casual/Frequent): Split based on proxy_score (0-2 vs 3-5).
        - Noise (-1): Sent to Isolation Forest Review.
        """
        if dbscan_cluster == -1:
            return 4 # Review (Anomaly)
        elif dbscan_cluster == 0:
            # Sub-threshold split for the large combined cluster
            return 0 if proxy_score <= 2 else 1
        elif dbscan_cluster == 1:
            # High risk cluster split
            return 2 if proxy_score <= 7 else 3
        return 4 # Fallback

    @staticmethod
    def evaluate_trajectory(user_id, base_state, weekly_features_df):
        """
        Evaluates a user's weekly features against their historical baseline 
        to determine if an intra-week transition is warranted.
        
        weekly_features_df: DataFrame of weekly features sorted chronologically.
        Must contain: 'amount_variance', 'loss_chase_ratio', 'weekly_anomaly'
        """
        # If they are placed in manual review at base, keep them there unless cleared
        state = base_state
        
        hist_variance = []
        hist_chase = []
        
        for _, row in weekly_features_df.iterrows():
            # OVERRIDE: Isolation Forest anomaly flagging
            # -1 indicates anomaly in IsolationForest
            if row.get('weekly_anomaly', 1) == -1:
                state = 4 
                continue
                
            # If we have baseline history, check for deltas
            if len(hist_variance) > 0:
                avg_var = np.mean(hist_variance)
                avg_chase = np.mean(hist_chase)
                
                # ESCALATION: Variance spikes > 50% AND loss-chase escalates
                if avg_var > 0 and row['amount_variance'] > avg_var * 1.5 and row['loss_chase_ratio'] > avg_chase + 0.1:
                    if state < 3: # Can't escalate beyond Crisis
                        state += 1
                        
                # DE-ESCALATION: Variance drops heavily and no loss chasing
                elif avg_var > 0 and row['amount_variance'] < avg_var * 0.5 and row['loss_chase_ratio'] == 0:
                    if state > 0 and state != 4: # Can't de-escalate below Casual or out of Review automatically here
                        state -= 1
                        
            # RECOVERY: If previously anomalous but this week is normal, revert to base_state
            if state == 4 and row.get('weekly_anomaly', 1) != -1:
                state = base_state
                
            hist_variance.append(row['amount_variance'])
            hist_chase.append(row['loss_chase_ratio'])
            
        return state

if __name__ == '__main__':
    print("Behavioral State Machine initialized. State map:")
    for k, v in BehavioralStateMachine.STATES.items():
        print(f"  {k} -> {v}")
