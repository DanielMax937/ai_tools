import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from llama_index.llms.azure_openai import AzureOpenAI
import asyncio
from playwright.async_api import async_playwright
import time

load_dotenv()

# 全局配置
MAX_SUB_URLS = 3  # 每个主 URL 最多处理的子 URL 数量

# Initialize AI model
llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    temperature=0.0
)

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

async def scrape_page_content(url: str) -> str:
    """
    使用 Playwright 抓取页面内容
    
    Args:
        url (str): 目标 URL
        
    Returns:
        str: 页面内容
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=30000)
            
            # 等待页面加载完成
            await page.wait_for_load_state("networkidle")
            
            # 获取页面主要内容
            # 尝试获取文章内容 - 这里的选择器需要根据目标网站调整
            content = await page.evaluate("""
                () => {
                    // 尝试找到文章主体
                    const articleSelectors = [
                        'article', 
                        '.article-content', 
                        '.post-content',
                        '.entry-content',
                        'main',
                        '#content'
                    ];
                    
                    let content = '';
                    
                    // 尝试不同的选择器
                    for (const selector of articleSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            content = element.innerText;
                            break;
                        }
                    }
                    
                    // 如果没有找到特定内容，获取 body 内容
                    if (!content) {
                        content = document.body.innerText;
                    }
                    
                    // 清理内容
                    return content
                        .replace(/\\s+/g, ' ')
                        .trim()
                        .substring(0, 15000); // 限制长度
                }
            """)
            
            await browser.close()
            return content
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return f"无法抓取内容: {str(e)}"

def summarize_content(content: str, url: str) -> str:
    """
    使用 AI 总结页面内容
    
    Args:
        content (str): 页面内容
        url (str): 页面 URL
        
    Returns:
        str: 内容摘要
    """
    try:
        if not content or len(content) < 100:
            return "无足够内容可供总结"
        
        prompt = f"""
        请对以下网页内容进行简洁的总结，提取关键信息和主要观点。总结应该不超过 150 字。
        
        网页: {url}
        
        内容:
        {content[:10000]}  # 限制输入长度
        
        总结:
        """
        
        response = llm.complete(prompt)
        summary = response.text.strip()
        
        return summary
    except Exception as e:
        print(f"Error summarizing content for {url}: {str(e)}")
        return f"无法生成摘要: {str(e)}"

async def process_sub_url(url: str) -> Dict:
    """
    处理子 URL：抓取内容并生成摘要
    
    Args:
        url (str): 子 URL
        
    Returns:
        Dict: 包含 URL 和摘要的字典
    """
    try:
        print(f"处理子链接: {url}")
        content = await scrape_page_content(url)
        summary = summarize_content(content, url)
        
        return {
            "url": url,
            "summary": summary
        }
    except Exception as e:
        print(f"Error processing sub URL {url}: {str(e)}")
        return {
            "url": url,
            "summary": f"处理失败: {str(e)}"
        }

async def process_all_sub_urls(urls: List[str], max_concurrent: int = 3, max_urls: int = None) -> List[Dict]:
    """
    并发处理多个子 URL
    
    Args:
        urls (List[str]): 子 URL 列表
        max_concurrent (int): 最大并发数
        max_urls (int): 最大处理 URL 数，如果为 None 则使用全局配置
        
    Returns:
        List[Dict]: 处理结果列表
    """
    # 使用全局配置或传入的参数
    if max_urls is None:
        max_urls = MAX_SUB_URLS
    
    # 限制处理的 URL 数量
    limited_urls = urls[:max_urls]
    if len(urls) > max_urls:
        print(f"⚠️ 限制处理 URL 数量为 {max_urls}（原始数量: {len(urls)}）")
    
    # 使用信号量限制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(url):
        async with semaphore:
            return await process_sub_url(url)
    
    # 创建任务
    tasks = [process_with_semaphore(url) for url in limited_urls]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    return results

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

async def process_recommendations_with_summaries(processed_data: Dict) -> Dict:
    """
    为推荐数据添加内容摘要
    
    Args:
        processed_data (Dict): 处理后的推荐数据
        
    Returns:
        Dict: 添加了摘要的推荐数据
    """
    urls_with_summaries = []
    
    for url_data in processed_data.get("urls", []):
        if url_data.get("status") != "有效" or not url_data.get("sub_urls"):
            urls_with_summaries.append(url_data)
            continue
        
        # 处理子链接
        sub_urls = url_data.get("sub_urls", [])
        total_sub_urls = len(sub_urls)
        limited_sub_urls = min(total_sub_urls, MAX_SUB_URLS)
        
        print(f"\n处理 {url_data.get('url')} 的子链接 ({limited_sub_urls}/{total_sub_urls} 个)...")
        
        # 处理子链接，使用全局限制
        sub_url_summaries = await process_all_sub_urls(sub_urls)
        
        # 更新 URL 数据
        url_data_with_summaries = url_data.copy()
        url_data_with_summaries["sub_url_summaries"] = sub_url_summaries
        urls_with_summaries.append(url_data_with_summaries)
    
    # 更新处理后的数据
    result = processed_data.copy()
    result["urls"] = urls_with_summaries
    
    return result

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

def format_recommendations_html(processed_data: Dict) -> str:
    """
    将推荐网站和子链接格式化为 HTML 内容
    
    Args:
        processed_data (Dict): 处理后的推荐数据
        
    Returns:
        str: 格式化后的 HTML 内容
    """
    # 构建推荐网站和子链接的 HTML
    sites_html = ""
    for i, url_data in enumerate(processed_data.get("urls", []), 1):
        if url_data.get("status") != "有效" or not url_data.get("sub_urls"):
            continue
            
        main_url = url_data.get("url")
        sub_url_summaries = url_data.get("sub_url_summaries", [])
        
        # 构建子链接 HTML
        sub_urls_html = ""
        for sub_data in sub_url_summaries:
            sub_url = sub_data.get("url")
            summary = sub_data.get("summary", "无摘要")
            
            sub_urls_html += f"""
            <li class="sub-url-item">
                <a href="{sub_url}" target="_blank" class="sub-url-link">{sub_url}</a>
                <div class="summary">
                    <p>{summary}</p>
                </div>
            </li>
            """
        
        # 添加网站和子链接
        sites_html += f"""
        <div class="site">
            <h3><a href="{main_url}" target="_blank">{main_url}</a></h3>
            <ul class="sub-urls">
                {sub_urls_html}
            </ul>
        </div>
        """
    
    # 构建完整 HTML
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
    .header {{
        text-align: center;
        margin-bottom: 30px;
    }}
    .site {{
        margin-bottom: 40px;
        padding: 20px;
        background: #fff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .site h3 {{
        margin-top: 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }}
    .site h3 a {{
        color: #2c5282;
        text-decoration: none;
    }}
    .sub-urls {{
        list-style-type: none;
        padding: 0;
    }}
    .sub-url-item {{
        margin-bottom: 20px;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 5px;
    }}
    .sub-url-link {{
        color: #4a5568;
        text-decoration: none;
        font-weight: bold;
        display: block;
        margin-bottom: 10px;
    }}
    .sub-url-link:hover {{
        text-decoration: underline;
        color: #2c5282;
    }}
    .summary {{
        font-size: 0.95em;
        color: #4a5568;
        line-height: 1.5;
        border-left: 3px solid #e2e8f0;
        padding-left: 15px;
        margin-top: 10px;
    }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h1>推荐网站及文章摘要</h1>
            <p>以下是根据您的兴趣推荐的网站及其最新文章摘要</p>
        </div>
        
        {sites_html}
    </div>
    </body>
    </html>
    """
    
    return html

async def main_async():
    """异步主函数"""
    print(f"Fetching user persona and recommendations... (最大子 URL 数量: {MAX_SUB_URLS})")
    
    # 获取数据
    data = fetch_persona_data()
    
    if data.get("success"):
        print("Data fetched successfully!")
        
        # 处理推荐数据
        processed_recommendations = process_recommendations(data)
        
        # 显示推荐数据
        display_recommendations(processed_recommendations)
        
        # 添加内容摘要
        processed_recommendations_with_summaries = await process_recommendations_with_summaries(processed_recommendations)
        
        # 格式化为 HTML（包含推荐网站、子链接和摘要）
        html_content = format_recommendations_html(processed_recommendations_with_summaries)
        
        # 发送邮件
        print("\nSending email report...")
        send_email(html_content)
    else:
        print(f"Failed to fetch data: {data.get('error', 'Unknown error')}")

def main():
    """主函数"""
    # 运行异步主函数
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
