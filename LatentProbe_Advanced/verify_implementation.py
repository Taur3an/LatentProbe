import json
import sys

def check_function_implementations(notebook_path):
    """Check if the functions in the notebook are properly implemented"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Checking implementation in: {notebook_path}")
        print("=" * 50)
        
        # Track found functions
        found_functions = {}
        
        # Look through all code cells
        for cell_idx, cell in enumerate(data['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Check for each function
                functions_to_check = [
                    'def run_single_shot',
                    'def evaluate_single_shot', 
                    'def run_multi_turn',
                    'def evaluate_multi_turn'
                ]
                
                for func_name in functions_to_check:
                    if func_name in source:
                        # Check if it's a proper implementation
                        has_api_call = 'client.chat.completions.create' in source
                        has_return = 'return' in source and 'print(' not in source.split('return')[0][-20:]
                        has_exception_handling = 'except Exception' in source
                        
                        found_functions[func_name] = {
                            'cell_index': cell_idx,
                            'has_api_call': has_api_call,
                            'has_return': has_return,
                            'has_exception_handling': has_exception_handling,
                            'is_proper_implementation': has_api_call and has_return
                        }
        
        # Print results
        print("Functions found:")
        for func_name, details in found_functions.items():
            print(f"\n{func_name}:")
            print(f"  Cell: {details['cell_index']}")
            print(f"  Has API calls: {details['has_api_call']}")
            print(f"  Has return values: {details['has_return']}")
            print(f"  Has exception handling: {details['has_exception_handling']}")
            print(f"  Properly implemented: {details['is_proper_implementation']}")
        
        # Check if all required functions are properly implemented
        properly_implemented = [
            details['is_proper_implementation'] 
            for details in found_functions.values()
        ]
        
        if len(properly_implemented) >= 4 and all(properly_implemented):
            print("\n" + "=" * 50)
            print("✅ SUCCESS: All functions are properly implemented!")
            print("✅ The notebook has real functionality, not just print statements.")
            return True
        else:
            print("\n" + "=" * 50)
            print("❌ ISSUE: Some functions are not properly implemented.")
            print("❌ They may only contain print statements.")
            return False
            
    except Exception as e:
        print(f"Error checking implementation: {e}")
        return False

if __name__ == "__main__":
    notebook_path = "AdversarialPromptGenerator_FIXED.ipynb"
    result = check_function_implementations(notebook_path)
    sys.exit(0 if result else 1)