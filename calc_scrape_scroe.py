#!/usr/bin/env python3
"""
Script to read filter_data sheet from Excel file, classify content using LLM, and save results.
"""

import pandas as pd
from pathlib import Path
import sys
import os
import csv
from openai import OpenAI
from typing import Optional, Tuple, Dict
import time
import re
import traceback

# LangChain imports
try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain not installed. Please install with: pip install langchain langchain-openai")
    LANGCHAIN_AVAILABLE = False


def classify_content_with_langchain(title: str, content: str, client: OpenAI) -> str:
    """
    Use LangChain agent to classify if content is about 'not_thought' or 'thoughts'.
    
    Args:
        title: The title of the content
        content: The main content text
        client: OpenAI client instance (used for API key and base_url)
        
    Returns:
        Classification result: 'not_thought' or 'thought'
    """
    
    try:
        # Initialize LangChain ChatOpenAI with the same settings as the original client
        llm = ChatOpenAI(
            model="gpt-4.1",
            openai_api_key=client.api_key,
            openai_api_base=str(client.base_url),
            temperature=0,
            max_tokens=4000
        )
        
        # Create a cognitive evaluation tool
        def classify_tool(query: str) -> str:
            """Tool to evaluate content for heuristic insight using 6-dimension cognitive framework."""
            prompt = f"""
            You are an expert in cognitive science and frontier technology. Your task is to evaluate whether content contains *heuristic insight* — that is, whether it contributes valuable, thought-provoking cognitive content.

            Please evaluate the content across the following six dimensions. Assign a score from 1 to 5 for each, and explain your reasoning for each score:

            ---

            【1. Novelty】
            - 5: Introduces a new concept, technology, research finding, or trend
            - 3: Common topic but offers a fresh take or interpretation
            - 1: Repeats known information or offers no new knowledge

            【2. Complexity & Depth】
            - 5: Involves logical reasoning, theoretical structure, or layered insight
            - 3: Some structural thinking, but not deeply explored
            - 1: Superficial or descriptive without deeper reasoning

            【3. Heuristic Value】
            - 5: Provides an "aha!" moment, usable model/framework/analogy, or thinking tool
            - 3: Somewhat thought-provoking, but vague or limited
            - 1: No cognitive trigger, purely declarative

            【4. Cognitive Rarity】
            - 5: Contrarian, counter-intuitive, or uniquely framed
            - 3: Somewhat unconventional but still within common discourse
            - 1: Mainstream or trivial perspective

            【5. Actionability】
            - 5: Directly applicable to work, learning, decision-making, or creation
            - 3: Indirectly useful, requires adaptation
            - 1: Not usable or applicable

            【6. Clarity】
            - 5: Clear, concise, and logically structured
            - 3: Understandable but requires effort
            - 1: Confusing, rambling, or poorly worded

            ---

            【Output Format】  
            Please return the following structured evaluation:

            Original Content:
            {query}

            Dimension Scores:
            1. Novelty: x/5 – (brief explanation)
            2. Complexity & Depth: x/5 – (brief explanation)
            3. Heuristic Value: x/5 – (brief explanation)
            4. Cognitive Rarity: x/5 – (brief explanation)
            5. Actionability: x/5 – (brief explanation)
            6. Clarity: x/5 – (brief explanation)

            Total Score (out of 30):
            <sum of scores>

            Conclusion:
            - ≥ 25: Highly Insightful
            - 16–24: Moderately Insightful
            - ≤ 15: Weak or Not Insightful

            Please ensure your analysis reflects careful reasoning, not just surface-level interpretation.
            """
            
            messages = [
                SystemMessage(content="You are an expert in cognitive science and frontier technology evaluation."),
                HumanMessage(content=prompt)
            ]
            
            response = llm.invoke(messages)
            return response.content

        # Create the tool
        classification_tool = Tool(
            name="evaluate_heuristic_insight",
            description="Evaluate content for heuristic insight using 6-dimension cognitive science framework",
            func=classify_tool
        )

        # Create prompt template for the agent
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in cognitive science and frontier technology. Use the evaluate_heuristic_insight tool to analyze the given title and content for cognitive value."),
            ("human", "Please evaluate this content for heuristic insight:\nTitle: {title}\nContent: {content}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create agent
        agent = create_openai_tools_agent(
            llm=llm,
            tools=[classification_tool],
            prompt=prompt_template
        )

        # Create executor
        executor = AgentExecutor(
            agent=agent,
            tools=[classification_tool],
            verbose=False,
            handle_parsing_errors=True
        )

        # Execute the agent
        input_query = f"{title} {content}"
        result = executor.invoke({
            "title": title,
            "content": content
        })
        
        # Parse the result
        agent_output = result.get("output", "")
        print(f"LangChain Agent Output: {agent_output}")
        
        # Try to extract score and conclusion from the output
        try:
            # Look for Total Score
            total_score = None
            if "Total Score" in agent_output:
                score_match = re.search(r'Total Score.*?(\d+)', agent_output)
                if score_match:
                    total_score = int(score_match.group(1))
            
            # Determine classification based on score or conclusion
            if total_score is not None:
                return total_score
            else:
                # Fallback: look for numerical scores in dimension scores
                scores = re.findall(r'(\d+)/5', agent_output)
                if len(scores) >= 6:
                    total = sum(int(score) for score in scores[:6])
                    return total
                else:
                    # Final fallback
                    return ''
                    
        except Exception as e:
            print(f"Error parsing LangChain output: {e}")
            # Fallback parsing
            if any(keyword in agent_output.lower() for keyword in ['highly insightful', 'moderately insightful']):
                return '1'
            else:
                return '0'
                
    except Exception as e:
        print(f"LangChain classification error: {e}")
        print(traceback.format_exc())

def read_and_classify_data(file_path: str, api_key: Optional[str] = None, max_rows: int = 100, classify_method: str = "langchain") -> None:
    """
    Read the Filter_Data sheet, classify content using LLM, and save results.
    
    Args:
        file_path: Path to the Excel file
        api_key: OpenAI API key (if not provided, will look for env variable)
        max_rows: Maximum number of rows to process (default: 100)
        classify_method: Classification method to use ('original', 'langchain', 'multi_agents')
    """
    print(f"Starting processing with max_rows: {max_rows}, classify_method: {classify_method}")
    try:
        # Check if file exists
        if not Path(file_path).exists():
            print(f"Error: File '{file_path}' not found.")
            return
        
        # Initialize OpenAI client
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY_MIRACLE')
        
        if not api_key:
            print("Error: OpenAI API key not found. Please set OPENAI_API_KEY_MIRACLE environment variable or pass it as argument.")
            return
            
        client = OpenAI(api_key=api_key, base_url='http://openai-proxy.miracleplus.com/v1')
        
        # Read the specific sheet
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        
        # Check if the sheet has any data
        if df.empty:
            print("Warning: The Test_Data sheet is empty.")
            return
        
        # Check if required columns exist
        required_columns = ['Title', 'Content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Limit the number of rows to process
        total_rows = len(df)
        rows_to_process = min(max_rows, total_rows)
        
        print(f"Reading Filter_Data sheet from: {file_path}")
        print(f"Total rows in sheet: {total_rows}")
        print(f"Processing rows: {rows_to_process} (limit: {max_rows})")
        print("Starting LLM classification...")
        print("-" * 50)
        
        # Initialize the ai_result column
        df['ai_result'] = ''
        
        # Create CSV file for index, title, and content
        csv_file_path = file_path.replace('.xlsx', f'_content_export_{rows_to_process}rows.csv')
        
        # Process limited number of rows and write to CSV
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            processed_count = 0
            for index, row in df.iterrows():
                if index < 200:
                    continue
                # Check if we've reached the limit
                if processed_count >= max_rows:
                    print(f"Reached maximum row limit ({max_rows}). Stopping processing.")
                    break
                
                title = str(row['Title']) if pd.notna(row['Title']) else ""
                content = str(row['Content']) if pd.notna(row['Content']) else ""
                
                # Skip empty rows for LLM classification
                if not title and not content:
                    df.at[index, 'ai_result'] = 'empty'
                    processed_count += 1
                    continue
                
                print(f"Processing row {processed_count + 1}/{rows_to_process}: {title[:50]}...")
                
                # Classify using selected method
                classification = classify_content_with_langchain(title, content, client)
                    
                
                df.at[index, 'ai_result'] = classification
                
                processed_count += 1
                
                # Print progress every 10 rows
                if processed_count % 10 == 0:
                    print(f"Completed {processed_count}/{rows_to_process} rows...")
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
        
        print("-" * 50)
        print(f"Finished processing {processed_count} rows out of {total_rows} total rows.")
        
        # Save the updated DataFrame back to Excel
        output_file = file_path.replace('.xlsx', f'_classified_{processed_count}rows.xlsx')
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Copy original sheets first
            original_xl = pd.ExcelFile(file_path)
            for sheet_name in original_xl.sheet_names:
                if sheet_name == 'Sheet1':
                    # Write the updated Filter_Data sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Copy other sheets as-is
                    original_df = pd.read_excel(file_path, sheet_name=sheet_name)
                    original_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Excel results saved to: {output_file}")
        
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except ValueError as e:
        if "Worksheet named 'Filter_Data' not found" in str(e):
            print("Error: Sheet 'Filter_Data' not found in the Excel file.")
            try:
                xl_file = pd.ExcelFile(file_path)
                print(f"Available sheets: {xl_file.sheet_names}")
            except Exception:
                pass
        else:
            print(f"Error reading Excel file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """Main function to run the script."""
    # Default values
    excel_file = "20250630.xlsx"
    max_rows = 100
    classify_method = "langchain"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            max_rows = int(sys.argv[2])
            if max_rows <= 0:
                print("Error: max_rows must be a positive integer.")
                return
        except ValueError:
            print("Error: max_rows must be a valid integer.")
            return
    
    if len(sys.argv) > 3:
        classify_method = sys.argv[3]
        valid_methods = ["original", "llm", "langchain", "multi_agents"]
        if classify_method not in valid_methods:
            print(f"Error: classify_method must be one of {valid_methods}")
            return
    
    # Check for API key in environment
    api_key = os.getenv('OPENAI_API_KEY_MIRACLE')
    if not api_key:
        print("Warning: OPENAI_API_KEY_MIRACLE environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY_MIRACLE='your-api-key-here'")
        print("Or add it to your .env file")
        return
    
    print(f"Starting processing with max_rows: {max_rows}, classify_method: {classify_method}")
    print("Usage: python read_excel_filter_data.py [excel_file] [max_rows] [classify_method]")
    print(f"Example: python read_excel_filter_data.py 20250611.xlsx 50 multi_agents")
    print("Classify methods: original/llm, langchain, multi_agents")
    print("-" * 50)
    
    read_and_classify_data(excel_file, api_key, max_rows, classify_method)


if __name__ == "__main__":
    main() 