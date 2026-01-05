import asyncio
import argparse
from typing import Dict, List
from playwright.async_api import async_playwright
from llama_index.llms.azure_openai import AzureOpenAI
import os
from dotenv import load_dotenv
import json
import time

# 加载环境变量
load_dotenv()

# 全局配置
MAX_SUB_URLS = 3  # 每个主 URL 最多处理的子 URL 数量

# 初始化 AI 模型
llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    temperature=0.0
)

async def scrape_page_content(url: str) -> str:
    """
    使用 Playwright 抓取页面内容
    
    Args:
        url (str): 目标 URL
        
    Returns:
        str: 页面内容
    """
    try:
        print(f"开始抓取: {url}")
        start_time = time.time()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # 设置超时并导航到页面
            await page.goto(url, timeout=30000)
            
            # 等待页面加载完成
            await page.wait_for_load_state("networkidle")
            
            # 获取页面标题
            title = await page.title()
            
            # 获取页面主要内容
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
                    let usedSelector = '';
                    
                    // 尝试不同的选择器
                    for (const selector of articleSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            content = element.innerText;
                            usedSelector = selector;
                            break;
                        }
                    }
                    
                    // 如果没有找到特定内容，获取 body 内容
                    if (!content) {
                        content = document.body.innerText;
                        usedSelector = 'body';
                    }
                    
                    // 清理内容
                    return {
                        content: content.replace(/\\s+/g, ' ').trim().substring(0, 15000),
                        selector: usedSelector
                    };
                }
            """)
            
            await browser.close()
            
            elapsed_time = time.time() - start_time
            print(f"抓取完成: {url}")
            print(f"页面标题: {title}")
            print(f"使用选择器: {content['selector']}")
            print(f"内容长度: {len(content['content'])} 字符")
            print(f"耗时: {elapsed_time:.2f} 秒")
            
            return content['content']
    except Exception as e:
        print(f"抓取错误 {url}: {str(e)}")
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
        
        print(f"开始总结内容: {url}")
        start_time = time.time()
        
        prompt = f"""
        请对以下网页内容进行简洁的总结，提取关键信息和主要观点。总结应该不超过 150 字。
        
        网页: {url}
        
        内容:
        {content[:10000]}  # 限制输入长度
        
        总结:
        """
        
        response = llm.complete(prompt)
        summary = response.text.strip()
        
        elapsed_time = time.time() - start_time
        print(f"总结完成: {url}")
        print(f"总结长度: {len(summary)} 字符")
        print(f"耗时: {elapsed_time:.2f} 秒")
        
        return summary
    except Exception as e:
        print(f"总结错误 {url}: {str(e)}")
        return f"无法生成摘要: {str(e)}"

async def process_url(url: str) -> Dict:
    """
    处理单个 URL：抓取内容并生成摘要
    
    Args:
        url (str): 目标 URL
        
    Returns:
        Dict: 包含 URL、内容和摘要的字典
    """
    try:
        print(f"\n===== 处理 URL: {url} =====")
        
        # 抓取内容
        content = await scrape_page_content(url)
        
        # 生成摘要
        summary = summarize_content(content, url)
        
        result = {
            "url": url,
            "content_length": len(content),
            "summary": summary,
            "summary_length": len(summary)
        }
        
        print(f"===== 处理完成: {url} =====\n")
        return result
    except Exception as e:
        print(f"处理错误 {url}: {str(e)}")
        return {
            "url": url,
            "error": str(e)
        }

async def process_urls(urls: List[str], max_concurrent: int = 2) -> List[Dict]:
    """
    并发处理多个 URL
    
    Args:
        urls (List[str]): URL 列表
        max_concurrent (int): 最大并发数
        
    Returns:
        List[Dict]: 处理结果列表
    """
    # 限制处理的 URL 数量
    limited_urls = urls[:MAX_SUB_URLS]
    if len(urls) > MAX_SUB_URLS:
        print(f"⚠️ 限制处理 URL 数量为 {MAX_SUB_URLS}（原始数量: {len(urls)}）")
    
    # 使用信号量限制并发数
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(url):
        async with semaphore:
            return await process_url(url)
    
    # 创建任务
    tasks = [process_with_semaphore(url) for url in limited_urls]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    return results

async def main_async():
    """异步主函数"""
    parser = argparse.ArgumentParser(description='测试网页抓取和内容总结')
    parser.add_argument('urls', nargs='+', help='要抓取的 URL 列表')
    parser.add_argument('--concurrent', type=int, default=2, help='最大并发数 (默认: 2)')
    parser.add_argument('--output', type=str, help='输出结果的 JSON 文件路径')
    parser.add_argument('--max-urls', type=int, help=f'每个主 URL 最多处理的子 URL 数量 (默认: {MAX_SUB_URLS})')
    
    args = parser.parse_args()
    
    # 更新全局配置
    global MAX_SUB_URLS
    if args.max_urls is not None:
        MAX_SUB_URLS = args.max_urls
        print(f"已设置最大子 URL 数量为: {MAX_SUB_URLS}")
    
    print(f"开始处理 URL，最大并发数: {args.concurrent}，最大子 URL 数量: {MAX_SUB_URLS}")
    
    # 处理 URL
    results = await process_urls(args.urls, args.concurrent)
    
    # 显示结果摘要
    print("\n===== 处理结果摘要 =====")
    for i, result in enumerate(results, 1):
        url = result.get("url")
        if "error" in result:
            print(f"{i}. ❌ {url} - 错误: {result.get('error')}")
        else:
            print(f"{i}. ✅ {url}")
            print(f"   内容长度: {result.get('content_length')} 字符")
            print(f"   摘要长度: {result.get('summary_length')} 字符")
            print(f"   摘要: {result.get('summary')[:100]}...")
        print()
    
    # 保存结果到 JSON 文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {args.output}")

def main():
    """主函数"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 