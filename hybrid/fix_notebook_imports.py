#!/usr/bin/env python3
"""
Fix missing sklearn.metrics imports in hybrid_complete_notebook.ipynb
"""
import json
import sys

def fix_notebook():
    notebook_path = 'hybrid/hybrid_complete_notebook.ipynb'
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and fix the cell with imports
    fixed = False
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if 'Additional imports for comprehensive visualizations' in source:
                print(f"Found cell {i} with imports")
                
                # Check if already fixed
                if 'from sklearn.metrics import' in source:
                    print(" Already fixed!")
                    return
                
                # Fix the imports
                old_line = 'from sklearn.preprocessing import MinMaxScaler'
                new_lines = '''from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score'''
                
                # Replace in source
                new_source = source.replace(old_line, new_lines)
                
                # Update cell source (split by lines)
                cell['source'] = new_source.split('\n')
                # Add newlines back
                cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                                 for i, line in enumerate(cell['source'])]
                
                fixed = True
                print(" Fixed imports!")
                break
    
    if not fixed:
        print("❌ Could not find the cell to fix")
        return
    
    # Save the fixed notebook
    output_path = 'hybrid/hybrid_complete_notebook_FIXED.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\n Fixed notebook saved to: {output_path}")
    print("\nYou can now:")
    print("1. Rename it to replace the original, OR")
    print("2. Open it directly in Jupyter")

if __name__ == '__main__':
    fix_notebook()
