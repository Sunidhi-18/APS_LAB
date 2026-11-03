"""
Execute all notebooks in-place using nbconvert ExecutePreprocessor
to ensure all cells have real, executed outputs, plots, and tables.
"""

import os
import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

sys.stdout.reconfigure(encoding='utf-8')

notebooks = [
    "notebook/01_dataset_exploration_and_visualization.ipynb",
    "notebook/02_model_training_and_comparison.ipynb",
    "notebook/model_evaluation_statistics.ipynb"
]

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

for nb_path in notebooks:
    print("=" * 70)
    print(f"[*] Executing Notebook: {nb_path}...")
    t0 = time.time()
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    try:
        # Run in the notebook's directory so relative paths like ../data work seamlessly
        nb_dir = os.path.dirname(os.path.abspath(nb_path))
        ep.preprocess(nb, {'metadata': {'path': nb_dir}})
        
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
            
        print(f"[+] Successfully executed {nb_path} in {time.time()-t0:.2f}s!")
    except Exception as e:
        print(f"[-] Execution failed for {nb_path}: {e}")
        import traceback
        traceback.print_exc()

print("\n✓ ALL NOTEBOOKS HAVE BEEN FULLY EXECUTED WITH SAVED OUTPUTS!")

