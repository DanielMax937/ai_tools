import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from llama_index.llms.azure_openai import AzureOpenAI
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import json
import pathlib

load_dotenv()

# Initialize AI model
llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    temperature=0.0
)

# Create cache directory
CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_cache(cache_file: str) -> Dict:
    """
    Load cache from a JSON file.
    
    Args:
        cache_file (str): Cache file name
        
    Returns:
        Dict: Cached data
    """
    cache_path = CACHE_DIR / cache_file
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
    return {}

def save_cache(cache_file: str, data: Dict):
    """
    Save cache to a JSON file.
    
    Args:
        cache_file (str): Cache file name
        data (Dict): Data to cache
    """
    cache_path = CACHE_DIR / cache_file
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache: {str(e)}")

def is_relevant_ask_hn(title: str, text: str = "") -> Tuple[bool, str]:
    """
    Use AI to determine if an ASK HN post is about what people are working on or new ideas.
    Use local cache to avoid repeated AI calls.
    
    Args:
        title (str): The title of the ASK HN post
        text (str): The text content of the post, if any
        
    Returns:
        Tuple[bool, str]: (is_relevant, explanation)
    """
    # Load cache
    cache = load_cache('relevance_cache.json')
    cache_key = f"{title}_{text}"
    
    # Check cache first
    if cache_key in cache:
        return cache[cache_key]['is_relevant'], cache[cache_key]['reason']
    
    prompt = f"""
    Analyze this Hacker News post and determine if it's asking about:
    1. What people are currently working on
    2. What new ideas/projects people have
    3. What people have been doing recently
    
    Title: {title}
    Text: {text}
    
    Return your analysis in the following format:
    RELEVANT: true/false
    REASON: your explanation
    """
    
    try:
        response = llm.complete(prompt)
        response_text = response.text
        
        # Parse AI response
        is_relevant = "RELEVANT: true" in response_text.lower()
        reason = response_text.split("REASON:")[1].strip() if "REASON:" in response_text else "No explanation provided"
        
        # Cache the result
        cache[cache_key] = {
            'is_relevant': is_relevant,
            'reason': reason
        }
        save_cache('relevance_cache.json', cache)
        
        return is_relevant, reason
        
    except Exception as e:
        print(f"Error in AI analysis: {str(e)}")
        return False, "Error in AI analysis"

def load_processed_news() -> set:
    """
    Load the set of processed news IDs from cache file.
    
    Returns:
        set: Set of processed news IDs
    """
    cache = load_cache('processed_news.json')
    return set(cache.get('processed_ids', []))

def save_processed_news(news_id: str):
    """
    Save a news ID to the processed news cache.
    
    Args:
        news_id (str): The ID of the processed news
    """
    cache = load_cache('processed_news.json')
    processed_ids = set(cache.get('processed_ids', []))
    processed_ids.add(news_id)
    cache['processed_ids'] = list(processed_ids)
    save_cache('processed_news.json', cache)

def get_hn_front_page(find_relevant_ask_hn: bool = False) -> List[Dict]:
    """
    Search for "what are you working on" related posts on Hacker News using the Algolia API.
    Uses advanced search parameters to match the web interface search.
    
    Args:
        find_relevant_ask_hn (bool): If True, keep fetching stories until finding a relevant ASK HN post
    
    Returns:
        List[Dict]: List of news items containing title, url, points, author, etc.
    """
    try:
        stories = []
        page = 0
        found_relevant_ask = False
        processed_news = load_processed_news()
        
        while True:
            # Search with advanced parameters matching web interface
            response = requests.get(
                'http://hn.algolia.com/api/v1/search',
                params={
                    'query': 'what are you working on',
                    'tags': 'ask_hn',
                    'hitsPerPage': 30,
                    'page': page,
                    'numericFilters': [], 
                },
                timeout=10
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            current_page_stories = []
            
            for hit in data.get('hits', []):
                # Skip if already processed
                if hit.get('objectID') in processed_news:
                    print(f"Skipping already processed story: {hit.get('title')}")
                    continue
                    
                # Convert Unix timestamp to readable date
                created_at = datetime.fromtimestamp(hit.get('created_at_i', 0))
                
                story = {
                    'id': hit.get('objectID', ''),
                    'title': hit.get('title', ''),
                    'url': hit.get('url', ''),
                    'points': hit.get('points', 0),
                    'author': hit.get('author', 'unknown'),
                    'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'comments_count': hit.get('num_comments', 0),
                    'comments_url': f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    'type': 'ask_hn',
                    'relevance_score': hit.get('relevance_score', 0)  # Add relevance score for debugging
                }
                
                # If we're looking for relevant ones, analyze it
                if find_relevant_ask_hn:
                    is_relevant, reason = is_relevant_ask_hn(story['title'])
                    # if is_relevant:
                    story['ai_analysis'] = reason
                    stories.append(story)
                    found_relevant_ask = True
                    break
                else:
                    current_page_stories.append(story)
            
            # If we're not looking for relevant ASK HN or we've found one
            if not find_relevant_ask_hn:
                stories.extend(current_page_stories)
                break
            elif found_relevant_ask:
                break
            
            # If we haven't found a relevant ASK HN post, continue to next page
            page += 1
            if page >= 3:  # Increase to 3 pages since we're sorting by popularity
                print("Reached page limit without finding a relevant post.")
                break
        
        # If we're not looking for relevant ASK HN, limit to 30 stories
        if not find_relevant_ask_hn:
            stories = stories[:30]
            
        return stories
        
    except Exception as e:
        print(f"Error fetching Hacker News: {str(e)}")
        return []

def get_hn_story_details(story_id: str) -> Dict:
    """
    Fetch detailed information about a specific Hacker News story.
    
    Args:
        story_id (str): The ID of the story to fetch
    
    Returns:
        Dict: Detailed story information
    """
    try:
        response = requests.get(
            f'http://hn.algolia.com/api/v1/items/{story_id}',
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching story details: {str(e)}")
        return {}

def search_hn_stories(query: str, tags: str = None, limit: int = 20) -> List[Dict]:
    """
    Search Hacker News stories using keywords.
    
    Args:
        query (str): Search query
        tags (str, optional): Filter by tags (e.g., 'story', 'comment', 'poll')
        limit (int): Maximum number of results to return (default: 20)
    
    Returns:
        List[Dict]: List of matching stories
    """
    try:
        params = {
            'query': query,
            'hitsPerPage': limit
        }
        if tags:
            params['tags'] = tags
            
        response = requests.get(
            'http://hn.algolia.com/api/v1/search',
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        return data.get('hits', [])
        
    except Exception as e:
        print(f"Error searching stories: {str(e)}")
        return []

def get_ask_hn_comments(story_id: str) -> List[Dict]:
    """
    Fetch comments for an ASK HN post.
    
    Args:
        story_id (str): The ID of the ASK HN post
        
    Returns:
        List[Dict]: List of comments with author and text
    """
    try:
        response = requests.get(
            f'http://hn.algolia.com/api/v1/items/{story_id}',
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract all comments recursively
        comments = []
        
        def extract_comments(comment_data):
            if not comment_data:
                return
                
            # Convert timestamp to readable date
            created_at = datetime.fromtimestamp(comment_data.get('created_at_i', 0))
            
            comment = {
                'author': comment_data.get('author', ''),
                'text': comment_data.get('text', ''),
                'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'points': comment_data.get('points', 0)
            }
            comments.append(comment)
            
            # Process child comments
            for child in comment_data.get('children', []):
                extract_comments(child)
        
        # Process all top-level comments
        for comment in data.get('children', []):
            extract_comments(comment)
            
        return comments
        
    except Exception as e:
        print(f"Error fetching comments: {str(e)}")
        return []

def analyze_comment_ideas(comment_text: str) -> Dict:
    """
    Analyze a comment using AI to extract product ideas and insights in Chinese.
    Use local cache to avoid repeated AI calls.
    
    Args:
        comment_text (str): The text content of the comment
        
    Returns:
        Dict: Analysis results in Chinese including product idea, target users, and use cases
    """
    # Load cache
    cache = load_cache('comment_analysis_cache.json')
    
    # Create a simple hash of the comment text as cache key
    cache_key = str(hash(comment_text))
    
    # Check cache first
    if cache_key in cache:
        return cache[cache_key]
    
    prompt = f"""
    分析这条 Hacker News 评论，提取其中的产品创意、项目或见解信息。
    如果评论中没有包含任何产品创意或项目，返回"NO_IDEA"。
    所有分析内容请用中文回答。
    
    评论内容:
    {comment_text}
    
    请按以下格式分析并返回：
    IDEA_TYPE: [无创意, 产品, 项目, 概念]
    SUMMARY: 创意/项目的简要概述
    TARGET_USERS: 目标用户群体
    USE_CASE: 解决什么问题/应用场景
    STAGE: [概念阶段, 开发中, 已上线, 未说明]
    """
    
    try:
        response = llm.complete(prompt)
        response_text = response.text.strip()
        
        # Parse the response
        lines = response_text.split('\n')
        result = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
                
        # Convert NO_IDEA to Chinese if present
        if result.get('IDEA_TYPE') == 'NO_IDEA':
            result['IDEA_TYPE'] = '无创意'
        
        # Cache the result
        cache[cache_key] = result
        save_cache('comment_analysis_cache.json', cache)
        
        return result
    except Exception as e:
        print(f"Error in comment analysis: {str(e)}")
        error_result = {"IDEA_TYPE": "错误", "SUMMARY": "分析过程中出错"}
        
        # Cache the error result too
        cache[cache_key] = error_result
        save_cache('comment_analysis_cache.json', cache)
        
        return error_result

def analyze_ask_hn_ideas(story_id: str, comment_limit: int = 10) -> List[Dict]:
    """
    Analyze comments in an ASK HN post to extract product ideas and insights.
    
    Args:
        story_id (str): The ID of the ASK HN post
        comment_limit (int): Maximum number of comments to analyze (default: 10)
        
    Returns:
        List[Dict]: List of analyzed comments with ideas and insights
    """
    try:
        # Get all comments
        comments = get_ask_hn_comments(story_id)
        analyzed_comments = []
        valid_ideas_count = 0
        
        # Limit to first N comments
        comments = comments[:comment_limit]
        print(f"\nAnalyzing first {len(comments)} comments for ideas and insights...")
        
        for i, comment in enumerate(comments, 1):
            print(f"Processing comment {i}/{len(comments)}...")
            
            # Skip empty comments
            if not comment.get('text'):
                continue
                
            # Analyze the comment
            analysis = analyze_comment_ideas(comment['text'])
            
            # Skip if no idea found or error
            if analysis.get('IDEA_TYPE') in ['无创意', '错误', 'NO_IDEA']:
                print(f"Skipping comment {i} - No valid idea found")
                continue
                
            analyzed_comment = {
                'author': comment['author'],
                'created_at': comment['created_at'],
                'text': comment['text'],
                'analysis': analysis
            }
            analyzed_comments.append(analyzed_comment)
            valid_ideas_count += 1
            print(f"Found valid idea in comment {i} - Type: {analysis.get('IDEA_TYPE')}")
        
        print(f"\nAnalysis complete. Found {valid_ideas_count} valid ideas in {len(comments)} comments.")
        return analyzed_comments
        
    except Exception as e:
        print(f"Error analyzing comments: {str(e)}")
        return []

def send_idea_summary_email(story_id: str, comment_limit: int = 10):
    """
    Analyze comments for product ideas and send a summary email.
    
    Args:
        story_id (str): The ID of the ASK HN post
        comment_limit (int): Maximum number of comments to analyze
    """
    try:
        # Get story details first
        story_details = get_hn_story_details(story_id)
        story_title = story_details.get('title', 'Unknown Title')
        story_author = story_details.get('author', 'Unknown Author')
        story_url = f"https://news.ycombinator.com/item?id={story_id}"
        
        # Get analyzed ideas
        ideas = analyze_ask_hn_ideas(story_id, comment_limit)
        
        # Create HTML content
        html_content = f"""
        <html>
        <head>
        <style>
        .container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .idea-card {{
            margin-bottom: 30px;
            padding: 20px;
            background: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .idea-type {{
            color: #2c5282;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        .highlight {{
            background-color: #f0f7ff;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        </style>
        </head>
        <body>
        <div class="container">
            <div class="header">
                <h2>HackerNews 创意洞察报告</h2>
                <p>原始帖子：<a href="{story_url}">{story_title}</a></p>
                <p>作者：{story_author}</p>
                <p>分析评论数：{comment_limit}</p>
                <p>发现创意数：{len(ideas)}</p>
            </div>
            
            <div class="ideas-section">
                <h3>创意汇总</h3>
        """
        
        # Add each idea
        for idx, idea in enumerate(ideas, 1):
            html_content += f"""
                <div class="idea-card">
                    <div class="idea-type">创意 #{idx} - {idea['analysis'].get('IDEA_TYPE', 'N/A')}</div>
                    <div class="meta">
                        作者: {idea['author']} | 发布时间: {idea['created_at']}
                    </div>
                    <p><strong>概述：</strong>{idea['analysis'].get('SUMMARY', 'N/A')}</p>
                    <p><strong>目标用户：</strong>{idea['analysis'].get('TARGET_USERS', 'N/A')}</p>
                    <p><strong>使用场景：</strong>{idea['analysis'].get('USE_CASE', 'N/A')}</p>
                    <p><strong>项目阶段：</strong><span class="highlight">{idea['analysis'].get('STAGE', 'N/A')}</span></p>
                </div>
            """
        
        html_content += """
            </div>
        </div>
        </body>
        </html>
        """
        
        # Send email
        sender = os.getenv('EMAIL_SENDER')
        password = os.getenv('EMAIL_PASSWORD')
        receiver = os.getenv('EMAIL_RECEIVER')
        smtp_server = "smtp.126.com"
        smtp_port = 465

        if not all([sender, password, receiver]):
            print("Error: Missing email configuration. Please check your .env file.")
            return

        # Create message
        message = MIMEMultipart('alternative')
        message['From'] = Header(sender)
        message['To'] = Header(receiver)
        message['Subject'] = Header(f'HN创意洞察: {story_title[:30]}...', 'utf-8')

        # Add HTML content
        message.attach(MIMEText(html_content, 'html', 'utf-8'))

        # Send email
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, message.as_string())
            
        print("Email sent successfully!")
        
    except Exception as e:
        print(f"Error sending idea summary email: {str(e)}")

if __name__ == "__main__":
    print("Searching for popular 'what are you working on' posts on Hacker News...")
    print("Using advanced search parameters: sort by popularity, all time range")
    
    # Get stories with find_relevant_ask_hn=True
    stories = get_hn_front_page(find_relevant_ask_hn=True)
    
    if not stories:
        print("No new relevant posts found.")
        exit(0)
    
    for story in stories:
        print(f"\nTitle: {story['title']}")
        print(f"Type: {story['type']}")
        print(f"URL: {story['comments_url']}")
        print(f"Points: {story['points']}")
        print(f"Author: {story['author']}")
        print(f"Comments: {story['comments_count']}")
        print(f"Created: {story['created_at']}")
        print(f"Relevance Score: {story.get('relevance_score', 'N/A')}")
        
        # Since this is an ASK HN post and we found it relevant, analyze and send email
        print("\nThis is a relevant ASK HN post!")
        print(f"AI Analysis: {story['ai_analysis']}")
        print("\nAnalyzing comments and sending email report...")
        send_idea_summary_email(story['id'], comment_limit=2000)
        
        # Record this news as processed
        save_processed_news(story['id'])
        print("News recorded as processed.")
        
        # We only process the first relevant post
        break
            
        print("-" * 80)
