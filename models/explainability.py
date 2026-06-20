import torch
import shap
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Need to redefine the model structure
import torch.nn as nn
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

def generate_explanations():
    print("Loading trained DP model...")
    checkpoint = torch.load('models/final_dp_model.pth', weights_only=False)
    
    X_train = checkpoint['X_train']
    X_test = checkpoint['X_test']
    feature_names = checkpoint['feature_names']
    
    model = RiskClassifier(input_dim=X_train.shape[1])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    print("Initializing SHAP GradientExplainer...")
    # GradientExplainer requires a background dataset
    background = torch.tensor(X_train[np.random.choice(X_train.shape[0], 100, replace=False)], dtype=torch.float32)
    explainer = shap.GradientExplainer(model, background)
    
    print("Computing SHAP values for test set samples...")
    samples = torch.tensor(X_test[:5], dtype=torch.float32)
    shap_values = explainer.shap_values(samples)
    
    with torch.no_grad():
        preds = torch.argmax(model(samples), dim=1).numpy()
    
    print("\n--- SHAP Top-3 Features Sample ---")
    output_str = "--- SHAP Top-3 Features Sample ---\n"
    for i in range(len(samples)):
        pred_class = preds[i]
        sv = shap_values[pred_class][i]
        
        top_idx = np.argsort(np.abs(sv))[::-1][:3]
        
        top_features = [feature_names[idx] for idx in top_idx]
        impacts = [sv[idx] for idx in top_idx]
        
        user_str = f"User Sample {i+1} | Predicted State: {pred_class}\n"
        print(user_str, end='')
        output_str += user_str
        for rank, (feat, imp) in enumerate(zip(top_features, impacts)):
            feat_str = f"  {rank+1}. {feat} (Impact: {imp:+.4f})\n"
            print(feat_str, end='')
            output_str += feat_str
            
    # Save the output to a text file for the walkthrough to reference
    with open('data/shap_sample.txt', 'w') as f:
        f.write(output_str)
            
if __name__ == "__main__":
    generate_explanations()
