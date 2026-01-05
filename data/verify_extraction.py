#!/usr/bin/env python3
"""
Quick verification script to check the extraction results
"""

import pandas as pd

def verify_extraction(original_file: str, updated_file: str, num_rows: int = 10):
    """Compare original and updated files to verify extraction worked correctly"""
    
    try:
        # Read both files
        original_df = pd.read_excel(original_file)
        updated_df = pd.read_excel(updated_file)
        
        print("=== VERIFICATION RESULTS ===\n")
        print(f"Original file: {original_file}")
        print(f"Updated file: {updated_file}")
        print(f"Showing first {num_rows} rows with changes:\n")
        
        first_col = original_df.columns[0]
        changes_found = 0
        
        for index in range(min(len(original_df), len(updated_df))):
            original_content = str(original_df.iloc[index][first_col])
            updated_content = str(updated_df.iloc[index][first_col])
            
            # Check if content changed
            if original_content != updated_content and not pd.isna(original_df.iloc[index][first_col]):
                changes_found += 1
                if changes_found <= num_rows:
                    print(f"--- Row {index + 1} ---")
                    print(f"Original (length {len(original_content)}):")
                    # Show just the end part with the command text
                    if len(original_content) > 200:
                        print(f"...{original_content[-200:]}")
                    else:
                        print(original_content)
                    print(f"\nUpdated (length {len(updated_content)}):")
                    print(updated_content)
                    print("\n" + "="*50 + "\n")
        
        print(f"Total changes found: {changes_found}")
        
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    verify_extraction("promptpilot.xlsx", "updated_promptpilot.xlsx", 5)



