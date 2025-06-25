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
import json
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


def classify_content_with_llm(title: str, content: str, client: OpenAI) -> str:
    """
    Use LLM to classify if content is about 'not_thought' or 'thoughts'.
    
    Args:
        title: The title of the content
        content: The main content text
        client: OpenAI client instance
        
    Returns:
        Classification result: 'not_thought' or 'thought'
    """
    try:
        # prompt = f"""
        # Please analyze the following title and content and classify it as either 'not_thought' or "thought".

        # - "not_thought": factual information, current events, announcements, product launches, updates, or anything else that is not a personal opinion, analysis, commentary, predictions, philosophical musings, etc.
        # - "thought": personal opinions, analysis, commentary, predictions, philosophical musings, etc.

        # Title: {title}
        # Content: {content}

        # Respond with exactly one word: either "not_thought" or "thought"
        # """

        prompt = f"""
        我正在构建一个具备深度认知与科研创新能力的 AI Scientist，它应具备以下核心能力：
            •	长期深度思考与推理能力
            •	科研问题识别与分析
            •	跨学科整合与创新
            •	系统化知识结构与认知学习能力
            •	专业科研场景中的智能协助

        为了训练这个系统，我希望构建一组高质量、具备深度思考轨迹的训练数据。我的想法是：

        从我与学生日常的学术讨论中提取过程性认知数据，尤其关注那些
            •	提出了有挑战性的问题
            •	我的回答过程中进行了细致分析与推理
            •	有助于揭示认知路径与科研思维方式

        你的任务：

        你将收到一段来自我们师生讨论的对话内容，内容包含一个具有科研价值的问题。

        请你执行以下操作：
            1.	判断问题是否具备科研/认知价值（即是否值得被 AI 模型学习和模仿）
            2.	提取该片段的结构化内容，包含：
                •	question: 问题的描述（可做重述以更清晰表达）
            3.  回答此问题（用简洁语言总结给出的解答）
            4.  thought：完整呈现从问题到解答的中间思考过程与推理路径


        注意：
            •	thought 是最关键部分，必须反映如何推理出答案，而不是直接陈述结论
            •	所有输出必须采用严格的 JSON 格式，格式如下：

        {{
            "question": "对问题的清晰描述",
            "answer": "对最终回答的简明总结",
            "thought": "详细的推理与分析过程，展示思考路径",
            "is_thought": "Yes 或 No，是否具备科研/认知价值"
        }}

        输入示例：

        问题：
        {title} {content}
        """
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                # {"role": "system", "content": "You are a content classifier. Respond with exactly one word: either 'not_thought' or 'thought'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0
        )
        
        # result = response.choices[0].message.content.strip().lower()

        llm_result = response.choices[0].message.content.strip()
        print(llm_result)
        exact_result = json.loads(llm_result)

        result = 'thought' if exact_result['is_thought'] == 'Yes' else 'not_thought'


        # Ensure we only return valid classifications
        if result in ['not_thought', 'thought']:
            return result
        else:
            # Default to 'unknown' if unclear
            return 'unknown'
            
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error classifying content: {e}")
        return 'unknown'


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
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available, falling back to direct OpenAI API")
        return classify_content_with_llm(title, content, client)
    
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
            - ≥ 27: Highly Insightful
            - 21–26: Moderately Insightful
            - ≤ 20: Weak or Not Insightful

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
                import re
                score_match = re.search(r'Total Score.*?(\d+)', agent_output)
                if score_match:
                    total_score = int(score_match.group(1))
            
            # Look for conclusion keywords
            conclusion = None
            if "Highly Insightful" in agent_output:
                conclusion = "Highly Insightful"
            elif "Moderately Insightful" in agent_output:
                conclusion = "Moderately Insightful"
            elif "Weak or Not Insightful" in agent_output or "Not Insightful" in agent_output:
                conclusion = "Weak or Not Insightful"
            
            # Determine classification based on score or conclusion
            if total_score is not None:
                # Use score-based classification
                if total_score >= 21:  # Moderately or Highly Insightful
                    return 'thought'
                else:  # <= 20: Weak or Not Insightful
                    return 'not_thought'
            elif conclusion:
                # Use conclusion-based classification
                if conclusion in ["Highly Insightful", "Moderately Insightful"]:
                    return 'thought'
                else:
                    return 'not_thought'
            else:
                # Fallback: look for numerical scores in dimension scores
                scores = re.findall(r'(\d+)/5', agent_output)
                if len(scores) >= 6:
                    total = sum(int(score) for score in scores[:6])
                    return 'thought' if total >= 21 else 'not_thought'
                else:
                    # Final fallback
                    return 'not_thought'
                    
        except Exception as e:
            print(f"Error parsing LangChain output: {e}")
            # Fallback parsing
            if any(keyword in agent_output.lower() for keyword in ['highly insightful', 'moderately insightful']):
                return 'thought'
            else:
                return 'not_thought'
                
    except Exception as e:
        print(f"LangChain classification error: {e}")
        print(traceback.format_exc())
        # Fallback to original method
        return classify_content_with_llm(title, content, client)


def classify_content_with_multi_agents(title: str, content: str, client: OpenAI) -> str:
    """
    Use multiple specialized LangChain agents to evaluate content across 6 dimensions.
    Each agent focuses on one specific dimension for more accurate evaluation.
    
    Args:
        title: The title of the content
        content: The main content text
        client: OpenAI client instance (used for API key and base_url)
        
    Returns:
        Classification result: 'not_thought' or 'thought'
    """
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available, falling back to direct OpenAI API")
        return classify_content_with_llm(title, content, client)
    
    try:
        # Initialize LangChain ChatOpenAI with the same settings as the original client
        llm = ChatOpenAI(
            model="gpt-4.1",
            openai_api_key=client.api_key,
            openai_api_base=str(client.base_url),
            temperature=0,
            max_tokens=500  # Shorter since each agent only returns a score
        )
        
        content_text = f"Title: {title}\nContent: {content}"
        
        # Define dimension-specific agents
        dimensions = {
            "novelty": {
                "prompt": """
                You are an expert in evaluating NOVELTY of content. Your task is to assess how novel and original the content is.

                Scoring criteria for Novelty:
                - 5: Introduces a completely new concept, technology, research finding, or trend
                - 4: Presents familiar concepts with significant new insights or connections
                - 3: Common topic but offers a fresh take or interpretation
                - 2: Some new elements but mostly familiar information
                - 1: Repeats known information or offers no new knowledge

                Content to evaluate:
                {content}

                Return ONLY a single number (1-5) representing the novelty score. No explanation needed.
                """,
                "description": "Evaluate content novelty and originality"
            },
            "complexity": {
                "prompt": """
                You are an expert in evaluating COMPLEXITY & DEPTH of content. Your task is to assess the depth of reasoning and structural thinking.

                Scoring criteria for Complexity & Depth:
                - 5: Involves sophisticated logical reasoning, theoretical structure, or multi-layered insight
                - 4: Good structural thinking with clear reasoning paths
                - 3: Some structural thinking, but not deeply explored
                - 2: Basic reasoning with limited depth
                - 1: Superficial or descriptive without deeper reasoning

                Content to evaluate:
                {content}

                Return ONLY a single number (1-5) representing the complexity score. No explanation needed.
                """,
                "description": "Evaluate content complexity and depth of reasoning"
            },
            "heuristic": {
                "prompt": """
                You are an expert in evaluating HEURISTIC VALUE of content. Your task is to assess how much the content provides cognitive insights or thinking tools.

                Scoring criteria for Heuristic Value:
                - 5: Provides clear "aha!" moments, usable models/frameworks/analogies, or powerful thinking tools
                - 4: Offers valuable cognitive insights that enhance understanding
                - 3: Somewhat thought-provoking, but insights are vague or limited
                - 2: Minor cognitive value, limited practical insights
                - 1: No cognitive trigger, purely declarative or informational

                Content to evaluate:
                {content}

                Return ONLY a single number (1-5) representing the heuristic value score. No explanation needed.
                """,
                "description": "Evaluate content's heuristic and cognitive value"
            },
            "rarity": {
                "prompt": """
                You are an expert in evaluating COGNITIVE RARITY of content. Your task is to assess how unconventional or uniquely framed the perspective is.

                Scoring criteria for Cognitive Rarity:
                - 5: Highly contrarian, counter-intuitive, or completely uniquely framed perspective
                - 4: Significantly unconventional viewpoint that challenges common thinking
                - 3: Somewhat unconventional but still within common discourse
                - 2: Slightly different perspective but mostly conventional
                - 1: Mainstream, trivial, or completely conventional perspective

                Content to evaluate:
                {content}

                Return ONLY a single number (1-5) representing the cognitive rarity score. No explanation needed.
                """,
                "description": "Evaluate content's cognitive rarity and uniqueness"
            },
            "actionability": {
                "prompt": """
                You are an expert in evaluating ACTIONABILITY of content. Your task is to assess how practically applicable the content is.

                Scoring criteria for Actionability:
                - 5: Directly and immediately applicable to work, learning, decision-making, or creation
                - 4: Highly applicable with minimal adaptation needed
                - 3: Indirectly useful, requires some adaptation or interpretation
                - 2: Limited practical application, mostly theoretical
                - 1: Not usable or applicable in practical contexts

                Content to evaluate:
                {content}

                Return ONLY a single number (1-5) representing the actionability score. No explanation needed.
                """,
                "description": "Evaluate content's practical actionability"
            },
            "clarity": {
                "prompt": """
                You are an expert in evaluating CLARITY of content. Your task is to assess how well-structured and understandable the content is.

                Scoring criteria for Clarity:
                - 5: Exceptionally clear, concise, and logically structured
                - 4: Well-organized and easy to understand
                - 3: Generally understandable but requires some effort
                - 2: Somewhat unclear or poorly organized
                - 1: Confusing, rambling, or very poorly worded

                Content to evaluate:
                {content}

                Return ONLY a single number (1-5) representing the clarity score. No explanation needed.
                """,
                "description": "Evaluate content's clarity and structure"
            }
        }
        
        # Create tools for each dimension
        tools = []
        for dimension_name, dimension_info in dimensions.items():
            def create_dimension_tool(name, prompt_template, desc):
                def evaluate_dimension(query: str) -> str:
                    formatted_prompt = prompt_template.format(content=query)
                    messages = [
                        SystemMessage(content=f"You are a specialized evaluator for {name}. Return only a number 1-5."),
                        HumanMessage(content=formatted_prompt)
                    ]
                    response = llm.invoke(messages)
                    return response.content.strip()
                
                return Tool(
                    name=f"evaluate_{name}",
                    description=desc,
                    func=evaluate_dimension
                )
            
            tool = create_dimension_tool(
                dimension_name, 
                dimension_info["prompt"], 
                dimension_info["description"]
            )
            tools.append(tool)
        
        # Create agent executor with all tools
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a multi-dimensional content evaluator. Use each specific evaluation tool to get scores for all 6 dimensions. Always use all 6 tools to get complete evaluation."),
            ("human", "Please evaluate this content across all 6 dimensions:\n{content}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_openai_tools_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_template
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=10  # Allow multiple tool calls
        )
        
        # Execute the multi-agent evaluation
        result = executor.invoke({"content": content_text})
        agent_output = result.get("output", "")
        
        print(f"Multi-Agent Output: {agent_output}")
        
        # Parse scores from agent output
        import re
        dimension_scores = {}
        
        # Extract scores for each dimension
        for dimension_name in dimensions.keys():
            # Look for pattern like "evaluate_novelty: 4" or just "4" in the context
            score_patterns = [
                rf"evaluate_{dimension_name}.*?(\d+)",
                rf"{dimension_name}.*?(\d+)",
                rf"(\d+)/5"  # fallback pattern
            ]
            
            score = None
            for pattern in score_patterns:
                matches = re.findall(pattern, agent_output, re.IGNORECASE)
                if matches:
                    try:
                        potential_score = int(matches[0])
                        if 1 <= potential_score <= 5:
                            score = potential_score
                            break
                    except ValueError:
                        continue
            
            if score is None:
                # Default fallback score if parsing fails
                score = 3
                print(f"Warning: Could not parse score for {dimension_name}, using default: {score}")
            
            dimension_scores[dimension_name] = score
        
        # Calculate total score and determine classification
        total_score = sum(dimension_scores.values())
        
        # Print detailed results
        print(f"\n" + "="*50)
        print("MULTI-AGENT EVALUATION RESULTS")
        print("="*50)
        for dim_name, score in dimension_scores.items():
            print(f"{dim_name.capitalize()}: {score}/5")
        print(f"Total Score: {total_score}/30")
        
        # Determine classification based on total score
        if total_score >= 21:  # Moderately or Highly Insightful
            classification = 'thought'
            conclusion = "Highly Insightful" if total_score >= 27 else "Moderately Insightful"
        else:  # <= 20: Weak or Not Insightful
            classification = 'not_thought'
            conclusion = "Weak or Not Insightful"
        
        print(f"Conclusion: {conclusion}")
        print(f"Classification: {classification}")
        print("="*50)
        
        return classification
        
    except Exception as e:
        print(f"Multi-agent classification error: {e}")
        print(traceback.format_exc())
        # Fallback to original method
        return classify_content_with_llm(title, content, client)


def calculate_metrics(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Calculate Precision, Recall, and F1 score by comparing AI results with ground truth.
    
    Args:
        df: DataFrame containing 'Thoughts' (ground truth) and 'ai_result' (predictions) columns
        
    Returns:
        Tuple of (metrics dictionary, mismatched_records dataframe)
    """
    # Filter out rows where either column is missing or has invalid values
    valid_rows = df[
        (df['Thoughts'].isin(['Y', 'N'])) & 
        (df['ai_result'].isin(['thought', 'not_thought']))
    ].copy()
    
    if len(valid_rows) == 0:
        print("Warning: No valid rows found for metric calculation.")
        empty_mismatches = pd.DataFrame(columns=['Original_Index', 'Actual_Value', 'Predicted_Value', 'Title'])
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'accuracy': 0.0, 'total_valid': 0}, empty_mismatches
    
    # Convert to binary format for easier calculation
    # Ground truth: Y = 1 (thought), N = 0 (not thought)
    # AI prediction: thought = 1, not_thought = 0
    valid_rows['gt_binary'] = (valid_rows['Thoughts'] == 'Y').astype(int)
    valid_rows['pred_binary'] = (valid_rows['ai_result'] == 'thought').astype(int)
    
    # Find mismatched records
    mismatched_rows = valid_rows[valid_rows['gt_binary'] != valid_rows['pred_binary']].copy()
    
    # Create mismatched records dataframe with original index
    mismatched_records = pd.DataFrame({
        'Original_Index': mismatched_rows.index + 1,  # +1 to make it 1-based for user readability
        'Actual_Value': mismatched_rows['Thoughts'],
        'Predicted_Value': mismatched_rows['ai_result'],
        'Title': mismatched_rows['Title'].str[:100] if 'Title' in mismatched_rows.columns else 'N/A'  # Truncate title for readability
    })
    
    # Calculate confusion matrix components
    tp = len(valid_rows[(valid_rows['gt_binary'] == 1) & (valid_rows['pred_binary'] == 1)])  # True Positive
    tn = len(valid_rows[(valid_rows['gt_binary'] == 0) & (valid_rows['pred_binary'] == 0)])  # True Negative
    fp = len(valid_rows[(valid_rows['gt_binary'] == 0) & (valid_rows['pred_binary'] == 1)])  # False Positive
    fn = len(valid_rows[(valid_rows['gt_binary'] == 1) & (valid_rows['pred_binary'] == 0)])  # False Negative
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'total_valid': len(valid_rows),
        'mismatched_count': len(mismatched_records),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return metrics, mismatched_records


def print_mismatched_records(mismatched_records: pd.DataFrame) -> None:
    """
    Print details of mismatched records.
    
    Args:
        mismatched_records: DataFrame containing mismatched predictions
    """
    if len(mismatched_records) == 0:
        print("\n🎉 Perfect! No mismatched records found.")
        return
    
    print(f"\n" + "=" * 80)
    print(f"MISMATCHED RECORDS ({len(mismatched_records)} total)")
    print("=" * 80)
    print(f"{'Index':<8} {'Actual':<8} {'Predicted':<12} {'Title (truncated)'}")
    print("-" * 80)
    
    for _, row in mismatched_records.iterrows():
        actual = row['Actual_Value']
        predicted = row['Predicted_Value']
        index = row['Original_Index']
        title = str(row['Title'])[:60] + "..." if len(str(row['Title'])) > 60 else str(row['Title'])
        
        print(f"{index:<8} {actual:<8} {predicted:<12} {title}")
    
    print("=" * 80)


def print_metrics_report(metrics: Dict[str, float], mismatched_records: pd.DataFrame) -> None:
    """
    Print a formatted metrics report.
    
    Args:
        metrics: Dictionary containing calculated metrics
        mismatched_records: DataFrame containing mismatched predictions
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Total valid comparisons: {metrics['total_valid']}")
    print(f"Correct predictions: {metrics['total_valid'] - metrics['mismatched_count']}")
    print(f"Mismatched predictions: {metrics['mismatched_count']}")
    print(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                 Thought  Not-Thought")
    print(f"Actual Thought      {metrics['tp']:3d}       {metrics['fn']:3d}")
    print(f"   Not-Thought      {metrics['fp']:3d}       {metrics['tn']:3d}")
    
    print("\nMetric Definitions:")
    print("• Precision: Of all items classified as 'thought', how many were actually thoughts?")
    print("• Recall: Of all actual thoughts, how many were correctly identified?")
    print("• F1 Score: Harmonic mean of precision and recall")
    print("=" * 60)
    
    # Print mismatched records
    print_mismatched_records(mismatched_records)


def read_and_classify_data(file_path: str, api_key: Optional[str] = None, max_rows: int = 100, classify_method: str = "multi_agents") -> None:
    """
    Read the Filter_Data sheet, classify content using LLM, and save results.
    
    Args:
        file_path: Path to the Excel file
        api_key: OpenAI API key (if not provided, will look for env variable)
        max_rows: Maximum number of rows to process (default: 100)
        classify_method: Classification method to use ('original', 'langchain', 'multi_agents')
    """
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
        df = pd.read_excel(file_path, sheet_name='Test_Data')
        
        # Check if the sheet has any data
        if df.empty:
            print("Warning: The Test_Data sheet is empty.")
            return
        
        # Check if required columns exist
        required_columns = ['Title', 'Content', 'Thoughts']
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
            csv_writer = csv.writer(csvfile)
            
            # Write CSV header
            csv_writer.writerow(['Index', 'Title', 'Content'])
            
            processed_count = 0
            for index, row in df.iterrows():
                # Check if we've reached the limit
                if processed_count >= max_rows:
                    print(f"Reached maximum row limit ({max_rows}). Stopping processing.")
                    break
                
                title = str(row['Title']) if pd.notna(row['Title']) else ""
                content = str(row['Content']) if pd.notna(row['Content']) else ""
                
                # Write to CSV file
                csv_writer.writerow([index + 1, title, content])
                
                # Skip empty rows for LLM classification
                if not title and not content:
                    df.at[index, 'ai_result'] = 'empty'
                    processed_count += 1
                    continue
                
                print(f"Processing row {processed_count + 1}/{rows_to_process}: {title[:50]}...")
                
                # Classify using selected method
                if classify_method == "original" or classify_method == "llm":
                    classification = classify_content_with_llm(title, content, client)
                elif classify_method == "langchain":
                    classification = classify_content_with_langchain(title, content, client)
                elif classify_method == "multi_agents":
                    classification = classify_content_with_multi_agents(title, content, client)
                else:
                    print(f"Warning: Unknown classify_method '{classify_method}', using default 'multi_agents'")
                    classification = classify_content_with_multi_agents(title, content, client)
                
                df.at[index, 'ai_result'] = classification
                
                processed_count += 1
                
                # Print progress every 10 rows
                if processed_count % 10 == 0:
                    print(f"Completed {processed_count}/{rows_to_process} rows...")
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
        
        print("-" * 50)
        print(f"Finished processing {processed_count} rows out of {total_rows} total rows.")
        print(f"CSV file saved to: {csv_file_path}")
        
        # Save the updated DataFrame back to Excel
        output_file = file_path.replace('.xlsx', f'_classified_{processed_count}rows.xlsx')
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Copy original sheets first
            original_xl = pd.ExcelFile(file_path)
            for sheet_name in original_xl.sheet_names:
                if sheet_name == 'Filter_Data':
                    # Write the updated Filter_Data sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Copy other sheets as-is
                    original_df = pd.read_excel(file_path, sheet_name=sheet_name)
                    original_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Excel results saved to: {output_file}")
        
        # Print classification summary for processed rows only
        if 'ai_result' in df.columns:
            # Only count classifications for the rows we actually processed
            processed_df = df.iloc[:processed_count]
            classification_counts = processed_df['ai_result'].value_counts()
            print(f"\nClassification Summary (for {processed_count} processed rows):")
            for classification, count in classification_counts.items():
                print(f"  {classification}: {count}")
        
        # Calculate and print metrics
        metrics, mismatched_records = calculate_metrics(df)
        print_metrics_report(metrics, mismatched_records)
        
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
    excel_file = "20250611.xlsx"
    max_rows = 4000
    classify_method = "multi_agents"
    
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