import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

load_dotenv()

def fetch_persona_data() -> Dict:
    """
    从 API 获取用户画像和推荐数据
    
    Returns:
        Dict: 包含用户画像、推荐 URL 和标签的数据
    """
    try:
        response = requests.get(
            'http://127.0.0.1:8000/kl/tags/persona-with-recommendations',
            timeout=1000
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching persona data: {str(e)}")
        return {"success": False, "error": str(e)}

def format_persona_html(data: Dict) -> str:
    """
    将用户画像数据格式化为 HTML 内容
    
    Args:
        data (Dict): API 返回的数据
        
    Returns:
        str: 格式化后的 HTML 内容
    """
    if not data.get("success"):
        return f"<p>获取数据失败: {data.get('error', '未知错误')}</p>"
    
    data = data.get("data", {})
    
    # 格式化用户画像
    persona = data.get("persona", "").replace("\n", "<br>")
    
    # 格式化推荐 URL
    urls = "\n".join([
        f'<li><a href="{url}" target="_blank">{url}</a></li>'
        for url in data.get("urls", [])
    ])
    
    # 格式化标签
    tags = "\n".join([
        f'<span class="tag">{tag["name"]}<span class="tag-count">{tag["count"]}</span></span>'
        for tag in data.get("tags", [])
    ])
    
    # 构建 HTML 内容
    html = f"""
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
    .section {{
        margin-bottom: 30px;
        padding: 20px;
        background: #fff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .persona {{
        white-space: pre-line;
        background: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
    }}
    .urls {{
        list-style-type: none;
        padding: 0;
    }}
    .urls li {{
        margin-bottom: 10px;
    }}
    .urls a {{
        color: #2c5282;
        text-decoration: none;
    }}
    .urls a:hover {{
        text-decoration: underline;
    }}
    .tags {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .tag {{
        background: #e2e8f0;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.9em;
    }}
    .tag-count {{
        background: #4a5568;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        margin-left: 5px;
        font-size: 0.8em;
    }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="section">
            <h2>用户画像分析</h2>
            <div class="persona">
                {persona}
            </div>
        </div>
        
        <div class="section">
            <h2>推荐网站</h2>
            <ul class="urls">
                {urls}
            </ul>
        </div>
        
        <div class="section">
            <h2>兴趣标签</h2>
            <div class="tags">
                {tags}
            </div>
        </div>
    </div>
    </body>
    </html>
    """
    
    return html

def send_email(content: str, is_html: bool = True):
    """
    使用 126 邮箱 SMTP 服务器发送邮件
    
    Args:
        content (str): 邮件内容
        is_html (bool): 是否为 HTML 格式
    """
    try:
        # 邮件配置
        sender = os.getenv('EMAIL_SENDER')
        password = os.getenv('EMAIL_PASSWORD')
        receiver = os.getenv('EMAIL_RECEIVER')
        smtp_server = "smtp.126.com"
        smtp_port = 465

        if not all([sender, password, receiver]):
            print("Error: Missing email configuration. Please check your .env file.")
            return

        # 创建邮件
        message = MIMEMultipart('alternative')
        message['From'] = Header(sender)
        message['To'] = Header(receiver)
        message['Subject'] = Header('用户画像分析报告', 'utf-8')

        # 添加内容
        if is_html:
            message.attach(MIMEText(content, 'html', 'utf-8'))
        else:
            message.attach(MIMEText(content, 'plain', 'utf-8'))

        # 创建 SMTP SSL 连接
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, message.as_string())
            
        print("Email sent successfully!")
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")

def main():
    """主函数"""
    print("Fetching user persona and recommendations...")
    
    # 获取数据
    data = fetch_persona_data()
    
    if data.get("success"):
        print("Data fetched successfully!")
        
        # 格式化为 HTML
        html_content = format_persona_html(data)
        
        # 发送邮件
        print("Sending email report...")
        send_email(html_content)
    else:
        print(f"Failed to fetch data: {data.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
