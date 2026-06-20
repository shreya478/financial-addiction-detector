from opacus.accountants.utils import get_noise_multiplier
import pandas as pd

# The DP-SGD configuration from dp_classifier.py
dataset_size = 4000 # 5000 * 0.8
batch_size = 256
epochs = 20
max_grad_norm = 1.0
delta = 1e-5
sample_rate = batch_size / dataset_size

print("--- OPACUS DP-SGD PARAMETERS ---")
for target_eps in [1.0, 5.0, 10.0]:
    noise_mult = get_noise_multiplier(
        target_epsilon=target_eps,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs
    )
    print(f"Target Epsilon: {target_eps:>4.1f} | Max Grad Norm: {max_grad_norm} | Epochs: {epochs} | Calculated Noise Multiplier: {noise_mult:.4f}")
