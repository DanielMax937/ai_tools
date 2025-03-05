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

def fetch_url_list_from_url(url: str, persona: str) -> Dict:
    """
    从 API 获取 URL 的子链接列表
    
    Args:
        url (str): 目标 URL
        
    Returns:
        Dict: API 返回的数据，包含文章链接和选择器信息
    """
    try:
        response = requests.post(
            'http://127.0.0.1:8000/kl/website/analyze',
            json={"url": url, "persona": persona},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching sub-URLs for {url}: {str(e)}")
        return {"success": False, "error": str(e)}

def process_recommendations(data: Dict) -> Dict:
    """
    处理推荐数据，包括验证、排序和增强推荐内容
    
    Args:
        data (Dict): API 返回的原始数据
        
    Returns:
        Dict: 处理后的推荐数据
    """
    if not data.get("success"):
        return {"urls": [], "tags": []}
    
    recommendations = data.get("data", {})
    urls = recommendations.get("urls", [])
    tags = recommendations.get("tags", [])
    
    # 验证 URL 有效性并获取子链接
    valid_urls = []
    persona = data.get("data", {}).get("persona", "")
    print("\n验证推荐网站...")
    for url in urls:
        try:
            # 简单的 HEAD 请求验证 URL 是否可访问
            response = requests.head(url, timeout=5)
            if response.status_code < 400:  # 2xx 或 3xx 状态码
                # 获取子链接
                sub_urls_data = fetch_url_list_from_url(url, persona)
                valid_urls.append({
                    "url": url,
                    "status": "有效",
                    "status_code": response.status_code,
                    "sub_urls": sub_urls_data.get("data", {}).get("article_links", []) if sub_urls_data.get("success") else []
                })
                print(f"✅ {url} - 状态码: {response.status_code}")
            else:
                valid_urls.append({
                    "url": url,
                    "status": "无效",
                    "status_code": response.status_code
                })
                print(f"❌ {url} - 状态码: {response.status_code}")
        except Exception as e:
            valid_urls.append({
                "url": url,
                "status": "错误",
                "error": str(e)
            })
            print(f"❌ {url} - 错误: {str(e)}")
    
    # 按标签数量排序
    sorted_tags = sorted(tags, key=lambda x: x.get("count", 0), reverse=True)
    
    # 分类标签
    tag_categories = {
        "高频标签": [tag for tag in sorted_tags if tag.get("count", 0) >= 5],
        "中频标签": [tag for tag in sorted_tags if 2 <= tag.get("count", 0) < 5],
        "低频标签": [tag for tag in sorted_tags if tag.get("count", 0) < 2]
    }
    
    return {
        "urls": valid_urls,
        "tags": sorted_tags,
        "tag_categories": tag_categories
    }

def display_recommendations(processed_data: Dict):
    """
    显示处理后的推荐数据
    
    Args:
        processed_data (Dict): 处理后的推荐数据
    """
    print("\n推荐网站列表:")
    print("-" * 50)
    for i, url_data in enumerate(processed_data.get("urls", []), 1):
        status_icon = "✅" if url_data.get("status") == "有效" else "❌"
        status_info = f"状态码: {url_data.get('status_code')}" if "status_code" in url_data else f"错误: {url_data.get('error', '未知错误')}"
        print(f"{i}. {status_icon} {url_data.get('url')} - {status_info}")
        # 显示子链接
        if url_data.get("sub_urls"):
            print(f"   子链接 ({len(url_data['sub_urls'])}个):")
            for j, sub_url in enumerate(url_data["sub_urls"][:5], 1):  # 只显示前5个子链接
                print(f"     {j}. {sub_url}")
            if len(url_data["sub_urls"]) > 5:
                print(f"     ... 还有 {len(url_data['sub_urls']) - 5} 个链接")
        print()  # 添加空行分隔不同网站
    print("-" * 50)
    
    # 显示标签分类
    print("\n标签分类:")
    print("-" * 50)
    for category, tags in processed_data.get("tag_categories", {}).items():
        if tags:
            print(f"\n{category} ({len(tags)}个):")
            for tag in tags:
                print(f"  • {tag.get('name')} ({tag.get('count')})")
    print("-" * 50)

def main():
    """主函数"""
    print("Fetching user persona and recommendations...")
    
    # 获取数据
    data = fetch_persona_data()
    
    if data.get("success"):
        print("Data fetched successfully!")
        
        # 处理推荐数据
        processed_recommendations = process_recommendations(data)
        
        # 显示推荐数据
        display_recommendations(processed_recommendations)
        
        # 格式化为 HTML
        html_content = format_persona_html(data)
        
        # 发送邮件
        print("\nSending email report...")
        send_email(html_content)
    else:
        print(f"Failed to fetch data: {data.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
