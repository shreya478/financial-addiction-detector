import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generator import generate_data
from data.labeling_engine import load_data, apply_labels

def main():
    print("Starting Financial Addiction Detection System - Data Pipeline")
    # Generate the synthetic data
    generate_data(num_users=5000, months=6)
    
    # Load and apply labels
    users, sessions, transactions = load_data()
    apply_labels(users, sessions, transactions)
    print("Pipeline finished successfully. Outputs are in the data/ directory.")

if __name__ == "__main__":
    main()
