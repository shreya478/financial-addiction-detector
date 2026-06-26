import os
import torch
import torch.nn as nn

class RiskClassifier(nn.Module):
    def __init__(self, input_dim=15, hidden_dim=32, num_classes=4):
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

_MODEL_INSTANCE = None

def get_model():
    """
    Singleton function to load the PyTorch model only once.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        model_path = os.path.join(os.path.dirname(__file__), 'final_dp_model.pth')
        # Initialize the model with 15 input features
        _MODEL_INSTANCE = RiskClassifier(input_dim=15)
        
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                # Handle cases where the saved file is a checkpoint dict vs. raw state_dict
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    _MODEL_INSTANCE.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    _MODEL_INSTANCE.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                print(f"Warning: Could not load weights from {model_path}. Using initialized weights. Error: {e}")
        
        _MODEL_INSTANCE.eval()
        
    return _MODEL_INSTANCE

def predict(feature_dict):
    """
    Takes a dictionary of features, runs inference, and returns predicted tier and probabilities.
    Missing keys in the dictionary will default to 0.0.
    """
    features_list = [
        "spending_velocity", 
        "session_frequency", 
        "loss_chasing_score",
        "avg_transaction_amount", 
        "night_session_ratio", 
        "weekend_ratio",
        "upi_ratio", 
        "trading_ratio", 
        "recovery_attempts", 
        "anomaly_score",
        "cluster_id", 
        "escalation_count", 
        "notification_response_rate",
        "days_since_first_transaction", 
        "total_transactions"
    ]
    
    # Extract features, defaulting to 0.0 if not present
    input_vector = []
    for feat in features_list:
        val = feature_dict.get(feat, 0.0)
        input_vector.append(float(val))
        
    tensor_input = torch.tensor([input_vector], dtype=torch.float32)
    
    model = get_model()
    
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
        predicted_tier = int(torch.argmax(logits, dim=1).item())
        
    labels_map = {
        0: "Casual",
        1: "Frequent",
        2: "Compulsive",
        3: "Crisis"
    }
    
    return {
        "tier": predicted_tier,
        "probabilities": {
            "Casual": probs[0],
            "Frequent": probs[1],
            "Compulsive": probs[2],
            "Crisis": probs[3]
        },
        "label": labels_map.get(predicted_tier, "Unknown")
    }
