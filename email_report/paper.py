from dotenv import load_dotenv
import os
from typing import List, Dict
from llama_index.core.tools.function_tool import FunctionTool
import arxiv
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
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
print("api key is", os.getenv('GOOGLE_API_KEY'))
function_calling_llm = Gemini(model="models/gemini-2.0-flash-exp", api_key=os.getenv('GOOGLE_API_KEY'))

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

def send_email(content: str):
    try:
        # Email configuration
        sender = os.getenv('EMAIL_SENDER')  # Your 126 email address
        password = os.getenv('EMAIL_PASSWORD')  # Your 126 email password or authorization code
        receiver = os.getenv('EMAIL_RECEIVER') # Recipient email address
        smtp_server = "smtp.126.com"
        smtp_port = 465  # SSL port

        if not all([sender, password, receiver]):
            print("Error: Missing email configuration. Please check your .env file.")
            return

        # Create message
        message = MIMEMultipart()
        message['From'] = Header(sender)
        message['To'] = Header(receiver)
        message['Subject'] = Header('ArXiv Research Summary', 'utf-8')

        # Add content
        message.attach(MIMEText(content, 'plain', 'utf-8'))

        # Create SMTP SSL connection
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            # Login
            server.login(sender, password)
            
            # Send email
            server.sendmail(sender, receiver, message.as_string())
            
        print("Email sent successfully!")
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Task-specific logic in a chatbot-like flow
def chatbot():
    print("🤖: Hi! I can help you research academic papers from arXiv based on your interests. Let's start!")
    
    # human_input = input("What topic would you like to research? ")
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
    res = agent.chat(search_prompt)
    print("\n🤖: Research complete!\n")
    send_email(res.response)


if __name__ == "__main__":
    chatbot()
    # send_email("Hello, this is a test email.")