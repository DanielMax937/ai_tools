from dotenv import load_dotenv
import os
from typing import List, Dict
from llama_index.core.tools.function_tool import FunctionTool
import arxiv
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
from llama_index.llms.azure_openai import AzureOpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

load_dotenv()
client = arxiv.Client()


def search_arxiv_papers(
    query: str,
) -> List[Dict[str, str]]:
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []
    # Use the Client for searching
    client = arxiv.Client()
    
    # Execute the search
    search = client.results(search)

    for result in search:
        paper_info = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'summary': result.summary,
                'published': result.published,
                'journal_ref': result.journal_ref,
                'doi': result.doi,
                'primary_category': result.primary_category,
                'categories': result.categories,
                'pdf_url': result.pdf_url,
                'arxiv_url': result.entry_id
            }
        papers.append(paper_info)

    return papers

def get_paper_details(paper_id: str) -> Dict[str, str]:
    search = arxiv.Search(id_list=[paper_id])
    paper = next(search.results())
    return {
        "title": paper.title,
        "authors": ", ".join([author.name for author in paper.authors]),
        "summary": paper.summary,
        "url": paper.pdf_url,
        "published": paper.published.strftime("%Y-%m-%d"),
        "categories": ", ".join(paper.categories),
        "doi": paper.doi if paper.doi else "Not available",
        "primary_category": paper.primary_category
    }

# Create tools list
tools = [
    FunctionTool.from_defaults(
        fn=search_arxiv_papers,
        name="search_arxiv_papers",
        description="Search for arXiv papers based on a query"
    ),
    FunctionTool.from_defaults(
        fn=get_paper_details,
        name="get_paper_details",
        description="Get detailed information about a specific arXiv paper using its ID"
    )
]
# function_calling_llm = Gemini(model="models/gemini-2.0-flash-exp", api_key=os.getenv('GOOGLE_API_KEY'))
# function_calling_llm = OpenAI(model="ep-20241222103914-fgbcp", api_key='8474f5ae-6e12-40d1-8e06-d453d9797fc3', api_base='https://ark.cn-beijing.volces.com/api/v3')
function_calling_llm = AzureOpenAI(
    engine="gpt-4o", model="gpt-4o", temperature=0.0
)
# Setup chatbot-style prefix messages
def create_prefix_message():
    return [
        ChatMessage(
            role="system",
            content=(
                """
                You are an expert at searching through arxiv papers,
                retrieving paper details, and summarizing those details
                into a report that you send as an email.
                """
            ),
        ),
    ]

prefix_messages = create_prefix_message()

# Initialize the agent
agent = FunctionCallingAgentWorker(
    tools=tools, # type: ignore
    llm=function_calling_llm,
    prefix_messages=prefix_messages,
    max_function_calls=20,
    allow_parallel_tool_calls=True,
    verbose=True,
).as_agent()

def send_email(content: str, is_html: bool = True):
    """
    Send email using 126 mail SMTP server
    
    Args:
        content (str): The email content to send
        is_html (bool): Whether the content is HTML format
    """
    try:
        # Email configuration
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
        message['Subject'] = Header('ArXiv 研究论文摘要', 'utf-8')

        # Add content with proper MIME type
        if is_html:
            message.attach(MIMEText(content, 'html', 'utf-8'))
        else:
            message.attach(MIMEText(content, 'plain', 'utf-8'))

        # Create SMTP SSL connection
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, message.as_string())
            
        print("Email sent successfully!")
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Task-specific logic in a chatbot-like flow
def chatbot():
    print("🤖: Hi! I can help you research academic papers from arXiv based on your interests. Let's start!")
    
    search_keyword = "AI"
    # Create initial search prompt
    search_prompt = (
        f"Research task: {search_keyword}\n"
        f"1. First, search for relevant papers using the search_arxiv_papers tool with appropriate filters.\n"
        f"2. After getting the results, analyze the most relevant papers and get detailed information if needed using get_paper_details.\n"
        f"3. Create a comprehensive research summary including:\n"
        f"   - Key findings and insights from the papers\n"
        f"   - Common themes and trends\n"
        f"   - Important methodologies or approaches\n"
        f"4. Format the response as a professional email with proper citations.\n"
        f"Please proceed with the research."
    )
    
    # Get research results and summary
    research_result = agent.chat(search_prompt)
    
    # Create formatting and translation prompt
    format_prompt = (
        "Please format and translate the following research summary email to Chinese. Follow these rules:\n"
        "1. Remove any email headers, greetings, or signatures\n"
        "2. Keep only the main content of the research summary\n"
        "3. Translate all content to Chinese while maintaining the academic accuracy\n"
        "4. Keep the original paper titles in English but add Chinese translations in parentheses\n"
        "5. Format the text with clear sections and bullet points for readability\n"
        "6. Keep all technical terms in both English and Chinese for clarity\n\n"
        f"Here's the content to format and translate:\n{research_result.response}"
    )
    
    # Get formatted and translated content
    formatted_result = agent.chat(format_prompt)
    
    # Create HTML formatting prompt
    html_prompt = (
        "Please format the following Chinese research summary into beautiful HTML format. Follow these rules:\n"
        "1. Use proper HTML structure with CSS styling\n"
        "2. Create a responsive design that looks good on both desktop and mobile\n"
        "3. Use appropriate colors, fonts, and spacing\n"
        "4. Format different sections with clear visual hierarchy\n"
        "5. Make paper titles stand out\n"
        "6. Use bullet points for lists\n"
        "7. Add subtle borders and background colors to separate sections\n"
        "8. Include a table of contents at the top\n"
        "Here's the CSS template to use:\n"
        '''
        <style>
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        .toc {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .paper-title {
            color: #2c5282;
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .paper-meta {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        .highlight {
            background-color: #f0f7ff;
            padding: 2px 5px;
            border-radius: 3px;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        </style>
        '''
        f"\nHere's the content to format:\n{formatted_result.response}"
    )
    
    # Get HTML formatted content
    html_result = agent.chat(html_prompt)
    
    print("\n🤖: Research complete, translated, and formatted!\n")
    send_email(html_result.response, is_html=True)


if __name__ == "__main__":
    chatbot()