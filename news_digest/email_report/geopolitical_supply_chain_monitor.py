"""Geopolitical Supply Chain Monitor for China Futures.

Scans Google News RSS for configured country/commodity pairs, uses an LLM
(OpenAI-compatible) to verify supply disruption/geopolitical risk relevance,
prints a structured JSON report of confirmed signals, and emails the results.
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

from openai import OpenAI


load_dotenv()

# Model for geopolitical monitor (overridable via env / Ark endpoint id)
# If GEO_OPENAI_MODEL is not set, fallback to ARK_MODEL_SEED_18 for convenience.
GEO_OPENAI_MODEL = os.getenv("GEO_OPENAI_MODEL", os.getenv("ARK_MODEL_SEED_18", ""))

# Initialize Ark-compatible OpenAI client for geopolitical monitor (expects ARK_API_KEY env var)
_ark_api_key = os.getenv("ARK_API_KEY")
if not _ark_api_key:
    print("Warning: ARK_API_KEY not set; geopolitical relevance analysis will be skipped.")
    geo_client = None
else:
    geo_client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=_ark_api_key,
    )


BASE_RSS_URL = "https://news.google.com/rss/search"


# Translation maps for Chinese display
COUNTRY_TRANSLATIONS = {
    "Indonesia": "印度尼西亚",
    "Myanmar": "缅甸",
    "Guinea": "几内亚",
    "Venezuela": "委内瑞拉",
    "South Africa": "南非",
    "Iran": "伊朗",
    "Red Sea": "红海",
    "Australia": "澳大利亚",
    "Chile/Peru": "智利/秘鲁",
    "Russia": "俄罗斯",
    "Malaysia": "马来西亚",
}

COMMODITY_TRANSLATIONS = {
    "Nickel": "镍",
    "Palm Oil": "棕榈油",
    "Rubber": "橡胶",
    "Tin": "锡",
    "Alumina": "氧化铝",
    "Bitumen": "沥青",
    "Manganese": "锰",
    "Platinum": "铂金",
    "Methanol": "甲醇",
    "Crude Oil": "原油",
    "EC (Shipping)": "EC（航运）",
    "Iron Ore": "铁矿石",
    "Lithium": "锂",
    "Copper": "铜",
}


# Configuration map: country -> configuration about commodities and keywords
WAR_ROOM_MAP: Dict[str, Dict[str, Any]] = {
    "Indonesia": {
        "commodities": ["Nickel", "Palm Oil", "Rubber"],
        "keywords": ["Export ban", "Tax", "RKAB", "Protest"],
    },
    "Myanmar": {
        "commodities": ["Tin"],
        "keywords": ["Wa State", "Mining ban", "Suspension"],
    },
    "Guinea": {
        "commodities": ["Alumina"],
        "keywords": ["Bauxite", "Coup", "Strike", "Explosion"],
    },
    "Venezuela": {
        "commodities": ["Bitumen"],
        "keywords": ["Oil", "Sanction", "PDVSA", "Maduro"],
    },
    "South Africa": {
        "commodities": ["Manganese", "Platinum"],
        "keywords": ["Eskom", "Transnet", "Strike", "Unrest"],
    },
    "Iran": {
        "commodities": ["Methanol", "Crude Oil"],
        "keywords": ["Gas cut", "Protest", "Shutdown"],
    },
    "Red Sea": {
        "commodities": ["EC (Shipping)"],
        "keywords": ["Houthi", "Maersk", "Attack"],
    },
    "Australia": {
        "commodities": ["Iron Ore", "Lithium"],
        "keywords": ["Cyclone", "Export", "Port Hedland"],
    },
    "Chile/Peru": {
        "commodities": ["Copper"],
        "keywords": ["Strike", "Road blockade", "Codelco"],
    },
    "Russia": {
        "commodities": ["Nickel"],
        "keywords": ["LME", "Sanction"],
    },
    "Malaysia": {
        "commodities": ["Palm Oil"],
        "keywords": ["Labor shortage", "Tax"],
    },
}


@dataclass
class NewsItem:
    country: str
    commodity: str
    title: str
    link: str
    pub_date: str


@dataclass
class AnalyzedNewsItem(NewsItem):
    is_relevant: bool
    sentiment: str
    reason: str


def build_query(country: str, keywords: List[str]) -> str:
    """Build the Google News search query string."""
    keyword_clause = " OR ".join(f'"{kw}"' for kw in keywords)
    return f'"{country}" AND ({keyword_clause})'


def build_rss_url(country: str, keywords: List[str]) -> str:
    """Construct the Google News RSS URL for a given country and keyword set."""
    params = {
        "q": build_query(country, keywords),
        "when": "24h",
        "ceid": "US:en",
        "hl": "en-US",
        "gl": "US",
    }
    return f"{BASE_RSS_URL}?{urlencode(params)}"


def fetch_rss(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch RSS feed content, returning XML text or None on error."""
    try:
        print(f"  Fetching RSS: {url}")
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        print(f"  Error fetching RSS: {exc}")
        return None


def parse_rss_items(xml_text: str) -> List[Tuple[str, str, str, str]]:
    """Parse RSS XML and extract (title, link, pub_date, description)."""
    items: List[Tuple[str, str, str, str]] = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            date_el = item.find("pubDate")
            desc_el = item.find("description")
            title = title_el.text if title_el is not None else ""
            link = link_el.text if link_el is not None else ""
            pub_date = date_el.text if date_el is not None else ""
            description = desc_el.text if desc_el is not None else ""
            if title and link:
                items.append((title, link, pub_date, description))
    except Exception as exc:  # noqa: BLE001
        print(f"  Error parsing RSS XML: {exc}")
    return items


def analyze_relevance(
    title: str,
    description: str,
    country: str,
    commodities: List[str],
) -> Dict[str, Any]:
    """Use LLM API to determine if a news title is a relevant risk signal.

    If no client is available, returns a default non-relevant response.
    """
    if geo_client is None:
        return {
            "is_relevant": False,
            "sentiment": "中性",
            "reason": "LLM客户端未配置，跳过分析。",
        }

    system_prompt = (
        "你是一位大宗商品交易分析师。判断以下新闻是否暗示给定国家及其关键商品存在供应中断或地缘政治风险。"
        "返回JSON格式：{\"is_relevant\": bool, \"impacted_commodities\": list[str], "
        "\"sentiment\": \"看涨\"/\"看跌\"/\"中性\", "
        "\"reason\": \"简短说明\"}。请用中文回答。"
    )

    try:
        response = geo_client.chat.completions.create(
            model=GEO_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {title}\nDescription: {description}"},
            ],
        )
        content = (response.choices[0].message.content or "")  # type: ignore[assignment]
        # Try to parse JSON from the model output. If parsing fails, fall back to neutral.
        content = content.strip()
        if not content:
            raise ValueError("Empty response from LLM")
        # Handle potential leading text before JSON
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = content[first_brace : last_brace + 1]
        else:
            json_str = content
        data = json.loads(json_str)
        # Ensure required keys
        return {
            "is_relevant": bool(data.get("is_relevant", False)),
            "sentiment": str(data.get("sentiment", "中性")),
            "reason": str(data.get("reason", "")),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  Error in AI analysis for title '{title}': {exc}")
        return {
            "is_relevant": False,
            "sentiment": "中性",
            "reason": "LLM分析过程中出错。",
        }


def send_email_report(subject: str, html_content: str) -> None:
    """Send the resulting report via email using .env configuration.

    Uses the same EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECEIVER pattern
    as other email_report scripts.
    """
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.126.com")
    smtp_port_str = os.getenv("EMAIL_SMTP_PORT", "465")

    if not all([sender, password, receiver]):
        print("Error: Missing email configuration. Please check your .env file.")
        return

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        print(f"Invalid SMTP port value: {smtp_port_str}")
        return

    try:
        message = MIMEMultipart("alternative")
        message["From"] = Header(sender)
        message["To"] = Header(receiver)
        message["Subject"] = Header(subject, "utf-8")

        message.attach(MIMEText(html_content, "html", "utf-8"))

        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, message.as_string())

        print("Email sent successfully!")
    except Exception as exc:  # noqa: BLE001
        print(f"Error sending email: {exc}")


def build_html_report(analyzed_items: List[AnalyzedNewsItem]) -> str:
    """Build a simple HTML report for email."""
    if not analyzed_items:
        return "<html><body><h2>过去24小时未检测到相关供应链中断。</h2></body></html>"

    rows = []
    for item in analyzed_items:
        country_cn = COUNTRY_TRANSLATIONS.get(item.country, item.country)
        commodity_cn = COMMODITY_TRANSLATIONS.get(item.commodity, item.commodity)
        rows.append(
            f"<tr>"
            f"<td>{country_cn}</td>"
            f"<td>{commodity_cn}</td>"
            f"<td><a href='{item.link}'>{item.title}</a></td>"
            f"<td>{item.pub_date}</td>"
            f"<td>{item.sentiment}</td>"
            f"<td>{item.reason}</td>"
            f"</tr>"
        )

    table_body = "\n".join(rows)
    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
          th {{ background-color: #f2f2f2; }}
        </style>
      </head>
      <body>
        <h2>地缘供应链监控 - 红色预警（过去24小时）</h2>
        <table>
          <thead>
            <tr>
              <th>国家</th>
              <th>商品</th>
              <th>标题</th>
              <th>发布日期</th>
              <th>情绪</th>
              <th>原因</th>
            </tr>
          </thead>
          <tbody>
            {table_body}
          </tbody>
        </table>
      </body>
    </html>
    """
    return html


def run_monitor() -> Dict[str, Any]:
    """Run the full monitoring pipeline and return a JSON-serializable report."""
    print("Starting Geopolitical Supply Chain Monitor (China Futures)...")
    all_hits: List[AnalyzedNewsItem] = []

    for country, cfg in WAR_ROOM_MAP.items():
        commodities: List[str] = cfg["commodities"]
        keywords: List[str] = cfg["keywords"]

        print(f"\nScanning country: {country}")
        rss_url = build_rss_url(country, keywords)
        xml_text = fetch_rss(rss_url)
        if not xml_text:
            continue

        all_items = parse_rss_items(xml_text)
        print(f"  Found {len(all_items)} raw news items.")
        items = all_items[:10]
        if len(all_items) > len(items):
            print(f"  Limiting to {len(items)} items for analysis.")

        for title, link, pub_date, description in items:
            print(f"    Analyzing: {title} [{country}]")
            analysis = analyze_relevance(title, description, country, commodities)
            if not analysis.get("is_relevant", False):
                continue

            impacted = analysis.get("impacted_commodities") or commodities
            for commodity in impacted:
                analyzed_item = AnalyzedNewsItem(
                    country=country,
                    commodity=commodity,
                    title=title,
                    link=link,
                    pub_date=pub_date,
                    is_relevant=True,
                    sentiment=analysis.get("sentiment", "neutral"),
                    reason=analysis.get("reason", ""),
                )
                all_hits.append(analyzed_item)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "hits": [asdict(item) for item in all_hits],
    }

    print("\n=== Red Alert Report (JSON) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Email the report
    html = build_html_report(all_hits)
    subject = "地缘供应链监控 - 红色预警（过去24小时）"
    send_email_report(subject, html)

    return report


if __name__ == "__main__":
    try:
        run_monitor()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)
