#!/usr/bin/env python3
"""
Script to update Excel file's first column (A) by extracting content 
from angle brackets after '命令文本：' and replacing the entire cell content
with just the extracted text.

Usage:
    python update_excel_command_text.py <excel_file_path>
"""

import pandas as pd
import re
import sys
import os
from typing import Optional

def extract_command_text(text: str) -> Optional[str]:
    """
    Extract content between angle brackets after '命令文本：'
    
    Args:
        text: The input text containing the command pattern
        
    Returns:
        Extracted command text or None if not found
    """
    if not isinstance(text, str):
        return None
    
    # Pattern to match '命令文本：<content>' and extract content
    pattern = r'命令文本：<([^>]+)>'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)
    return None

def update_excel_command_text(file_path: str, output_path: Optional[str] = None) -> bool:
    """
    Update Excel file's column A by extracting command text from angle brackets
    
    Args:
        file_path: Path to the input Excel file
        output_path: Path for output file (if None, overwrites input file)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return False
        
        print(f"Reading Excel file: {file_path}")
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        if df.empty:
            print("Warning: Excel file is empty.")
            return False
        
        # Get the first column (A)
        first_col = df.columns[0]
        
        print(f"Processing column '{first_col}'...")
        
        # Track changes
        changes_made = 0
        
        # Process each cell in the first column
        for index, cell_value in df[first_col].items():
            if pd.isna(cell_value):
                continue
                
            # Extract command text from angle brackets
            extracted_text = extract_command_text(str(cell_value))
            
            if extracted_text:
                print(f"Row {index + 1}: Extracted '{extracted_text}'")
                df.at[index, first_col] = extracted_text
                changes_made += 1
            else:
                print(f"Row {index + 1}: No command text pattern found, keeping original content")
        
        if changes_made == 0:
            print("No command text patterns found in column A.")
            return True
        
        # Determine output file path
        if output_path is None:
            output_path = file_path
        
        print(f"Saving updated file to: {output_path}")
        
        # Save the updated Excel file
        df.to_excel(output_path, index=False)
        
        print(f"Successfully updated {changes_made} cells in column A.")
        return True
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments and execute the update"""
    if len(sys.argv) < 2:
        print("Usage: python update_excel_command_text.py <excel_file_path> [output_file_path]")
        print("\nExample:")
        print("  python update_excel_command_text.py promptpilot.xlsx")
        print("  python update_excel_command_text.py promptpilot.xlsx updated_promptpilot.xlsx")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Convert to absolute path if relative
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.getcwd(), input_file)
    
    if output_file and not os.path.isabs(output_file):
        output_file = os.path.join(os.getcwd(), output_file)
    
    success = update_excel_command_text(input_file, output_file)
    
    if success:
        print("\n✓ Excel file updated successfully!")
    else:
        print("\n✗ Failed to update Excel file.")
        sys.exit(1)

if __name__ == "__main__":
    main()



