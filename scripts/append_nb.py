import nbformat

with open('models/clustering_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

markdown_content = """## Appendix: Debugging the DP-SGD Class Imbalance Trap

During the implementation of the DP-SGD PyTorch model, we encountered a severe class imbalance collapse where the minority `Compulsive` class failed to learn (F1 = 0.0000). 

Here is the step-by-step debugging progression for the `Epsilon=10` model:

### 1. Baseline Model
All Compulsive test samples were absorbed into the adjacent Frequent and Crisis classes.
```text
True / Pred     | Casual     | Frequent   | Compulsive | Crisis    
-----------------------------------------------------------------
Casual          | 278        | 127        | 0          | 0         
Frequent        | 74         | 376        | 0          | 0         
Compulsive      | 0          | 42         | 0          | 6         
Crisis          | 0          | 0          | 0          | 97        
```

### 2. Weighted Cross-Entropy Loss
We applied a 10x class weight to Compulsive, but F1 remained 0.0000. **Why?** DP-SGD gradient clipping (`max_grad_norm=1.0`) neutralized the loss scaling *before* noise injection!
```text
True / Pred     | Casual     | Frequent   | Compulsive | Crisis    
-----------------------------------------------------------------
Casual          | 328        | 77         | 0          | 0         
Frequent        | 116        | 334        | 0          | 0         
Compulsive      | 0          | 42         | 0          | 6         
Crisis          | 1          | 0          | 0          | 96        
```

### 3. SMOTE Oversampling
By physically duplicating Compulsive vectors to match the Frequent class size before DP-SGD training, we forced enough clipped gradients into the engine to establish the decision boundary.
```text
True / Pred     | Casual     | Frequent   | Compulsive | Crisis    
-----------------------------------------------------------------
Casual          | 266        | 139        | 0          | 0         
Frequent        | 76         | 247        | 127        | 0         
Compulsive      | 0          | 3          | 39         | 6         
Crisis          | 1          | 0          | 0          | 96        
```
*(Note: SMOTE caused 127 false-positive "Frequent" samples to be flagged as "Compulsive", which is a standard precision/recall tradeoff).*
"""

new_cell = nbformat.v4.new_markdown_cell(markdown_content)
nb.cells.append(new_cell)

with open('models/clustering_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook updated with debugging markdown.")
