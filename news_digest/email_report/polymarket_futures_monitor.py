"""Polymarket Futures Monitor — Orderbook-Based Insider Detection.

Scans Polymarket prediction markets for events related to futures/commodities,
then analyses live orderbook data, price history, and recent trades to detect
potential insider trading signals.

Insider Detection Core (6 signals):
  1. Large Orders       – bid/ask size > 5% of daily volume (counted as one signal)
  2. Order Imbalance   – bid/ask depth ratio > 2.5× or < 0.4× (one-sided pressure)
  3. Thin Orderbook    – depth < 5% of daily volume (easy to manipulate)
  4. Aggressive Bidding – tight spread (< 1%) combined with large orders (urgency)
  5. Sudden Price Move  – price changed > 15% in last 1h vs 24h average (historical)
  6. Trade Clustering   – single wallet made ≥20 trades OR ≥40% of recent volume

Confidence: high (≥4 signals or classic pattern), medium (3), low (0–2).
A market is flagged ``is_insider=True`` when ≥3 signals fire.
"""

from __future__ import annotations

import json
import os
import random
import smtplib
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# LLM configuration (same Ark/Volcengine pattern as the other scripts)
# ---------------------------------------------------------------------------
PM_OPENAI_MODEL = os.getenv(
    "GEO_OPENAI_MODEL", os.getenv("ARK_MODEL_SEED_18", "")
)

_ark_api_key = os.getenv("ARK_API_KEY")
if not _ark_api_key:
    print("Warning: ARK_API_KEY not set; LLM analysis will be skipped.")
    llm_client: Optional[OpenAI] = None
else:
    llm_client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=_ark_api_key,
    )

# ---------------------------------------------------------------------------
# Polymarket API endpoints
# ---------------------------------------------------------------------------
POLYMARKET_ENDPOINTS = [
    # Commodities / Futures tag
    (
        "https://gamma-api.polymarket.com/events/pagination"
        "?tag_id=101031&limit=20&archived=false"
        "&order=volume24hr&ascending=false&offset=0&active=true&closed=false"
    ),
    # Politics (offset=40 to capture geo events deeper in list)
    (
        "https://gamma-api.polymarket.com/events/pagination"
        "?limit=20&active=true&archived=false&tag_slug=politics"
        "&closed=false&order=volume24hr&ascending=false&offset=40"
    ),
    # World events
    (
        "https://gamma-api.polymarket.com/events/pagination"
        "?limit=20&active=true&archived=false&tag_slug=world"
        "&closed=false&order=volume24hr&ascending=false&offset=20"
    ),
]

CLOB_BOOK_URL = "https://clob.polymarket.com/book"

# ---------------------------------------------------------------------------
# HTTP client – browser-like headers to avoid Cloudflare bot detection
# ---------------------------------------------------------------------------
# Polymarket uses Cloudflare which performs TLS fingerprinting.  Python's
# `requests` (urllib3) is blocked at the TLS-handshake level.  `httpx` has a
# different TLS stack that passes the gamma-api check.  The CLOB API has even
# stricter fingerprinting, so we fall back to calling `curl` via subprocess
# for orderbook fetches.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://polymarket.com",
    "Referer": "https://polymarket.com/",
}

_http_client = httpx.Client(
    headers=_BROWSER_HEADERS,
    timeout=20,
    follow_redirects=True,
)

# ---------------------------------------------------------------------------
# Rate-limited GET – avoids triggering Polymarket anti-bot / rate limits
# ---------------------------------------------------------------------------
API_MIN_INTERVAL = float(os.getenv("PM_API_MIN_INTERVAL", "1.5"))
API_JITTER = float(os.getenv("PM_API_JITTER", "1.0"))

_last_request_time = 0.0
_rate_lock = threading.Lock()


def _rate_limit_wait() -> None:
    """Block until the minimum inter-request interval has elapsed."""
    global _last_request_time  # noqa: PLW0603

    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        wait = API_MIN_INTERVAL - elapsed
        if wait > 0:
            jitter = random.uniform(0, API_JITTER)
            total_wait = wait + jitter
            print(f"      [rate-limit] waiting {total_wait:.1f}s …")
            time.sleep(total_wait)
        _last_request_time = time.monotonic()


def _rate_limited_get(url: str, **kwargs: Any) -> httpx.Response:
    """GET via ``httpx`` with rate-limiting and browser headers.

    Used as a fallback; prefer ``_curl_get_json`` for Polymarket endpoints.
    """
    _rate_limit_wait()
    return _http_client.get(url, **kwargs)


CURL_MAX_RETRIES = 4
CURL_RETRY_BACKOFF = 3.0  # base seconds; grows exponentially per retry


def _curl_get_json(url: str, params: Optional[Dict[str, str]] = None) -> Any:
    """GET via ``curl`` subprocess – bypasses strict TLS fingerprinting.

    Retries up to ``CURL_MAX_RETRIES`` times with exponential back-off on
    TLS / connection errors.  Returns parsed JSON or ``None`` on failure.
    """
    if params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urlencode(params)

    cmd = [
        "curl", "-s",
        "--ipv4",           # force IPv4 – more reliable through Cloudflare
        "--tlsv1.2",
        "--max-time", "15",
        "-H", f"User-Agent: {_BROWSER_HEADERS['User-Agent']}",
        "-H", "Accept: application/json",
        "-H", "Accept-Language: en-US,en;q=0.9",
        "-H", "Origin: https://polymarket.com",
        "-H", "Referer: https://polymarket.com/",
        url,
    ]

    for attempt in range(1, CURL_MAX_RETRIES + 1):
        _rate_limit_wait()
        try:
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=20,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            # Retry on TLS (35), connection-reset (56), or recv-failure (52/55/7)
            retriable = {7, 35, 52, 55, 56}
            if result.returncode in retriable and attempt < CURL_MAX_RETRIES:
                backoff = CURL_RETRY_BACKOFF * (2 ** (attempt - 1)) + random.uniform(1, 3)
                print(f"      curl attempt {attempt}/{CURL_MAX_RETRIES} failed (exit {result.returncode}), retrying in {backoff:.0f}s …")
                time.sleep(backoff)
                continue
            if result.returncode != 0:
                print(f"      curl failed (exit {result.returncode}) after {attempt} attempt(s)")
                return None
        except subprocess.TimeoutExpired:
            if attempt < CURL_MAX_RETRIES:
                backoff = CURL_RETRY_BACKOFF * (2 ** (attempt - 1))
                print(f"      curl timed out (attempt {attempt}), retrying in {backoff:.0f}s …")
                time.sleep(backoff)
                continue
            print(f"      curl timed out after {attempt} attempts")
            return None
        except json.JSONDecodeError as exc:
            print(f"      curl returned non-JSON: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"      curl error: {exc}")
            return None
    return None

    return _http_session.get(url, **kwargs)


# ---------------------------------------------------------------------------
# Insider detection thresholds (orderbook-based)
# ---------------------------------------------------------------------------
SIZE_THRESHOLD_PCT = 0.05    # Large order = 5% of daily volume
SIZE_THRESHOLD_MIN = 20_000.0  # … or $20 000 minimum
DEPTH_THRESHOLD_PCT = 0.05   # Thin book = depth < 5% of daily volume
DEPTH_THRESHOLD_MIN = 10_000.0  # … or $10 000 minimum
IMBALANCE_HIGH = 2.5         # Severe bid-heavy imbalance ratio
IMBALANCE_LOW = 0.4          # Severe ask-heavy imbalance ratio
TIGHT_SPREAD_PCT = 0.01      # Aggressive bidding spread threshold (1%)
# Historical comparison
PRICE_CHANGE_THRESHOLD = 0.15  # Sudden price move > 15% vs 24h avg
# Trade clustering
CLUSTER_MIN_TRADES = 20         # Same wallet ≥ 20 trades in recent window
CLUSTER_VOLUME_PCT = 0.40       # Same wallet ≥ 40% of recent traded volume

CLOB_PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"
DATA_API_TRADES_URL = "https://data-api.polymarket.com/trades"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class InsiderDetails:
    """Granular orderbook metrics behind the insider detection."""

    large_bid_order: bool = False
    large_ask_order: bool = False
    order_imbalance: bool = False
    thin_orderbook: bool = False
    aggressive_bidding: bool = False
    sudden_price_move: bool = False
    trade_clustering: bool = False
    largest_bid_size: float = 0.0
    largest_ask_size: float = 0.0
    imbalance_ratio: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread: float = 0.0
    # Historical comparison
    price_24h_avg: float = 0.0
    price_recent: float = 0.0
    price_change_pct: float = 0.0
    # Trade clustering
    top_wallet_trades: int = 0
    top_wallet_volume_pct: float = 0.0
    unique_wallets: int = 0


@dataclass
class MarketAnalysis:
    """Result of insider-trading analysis on a single prediction market."""

    market_question: str
    current_price: float
    yes_price: float
    no_price: float
    volume_24h: float
    total_volume: float
    is_insider: bool
    confidence: str  # "low" | "medium" | "high"
    signals: List[str] = field(default_factory=list)
    details: InsiderDetails = field(default_factory=InsiderDetails)


@dataclass
class FuturesEvent:
    """A Polymarket event determined to be related to futures/commodities."""

    event_id: str
    title: str
    description: str
    slug: str
    image_url: str
    related_commodities: List[str]
    impact_analysis: str
    markets: List[MarketAnalysis] = field(default_factory=list)
    has_insider_signal: bool = False


# ---------------------------------------------------------------------------
# 1. Fetch events from Polymarket
# ---------------------------------------------------------------------------
def fetch_polymarket_events() -> List[Dict[str, Any]]:
    """Fetch events from all configured Polymarket endpoints and deduplicate.

    Uses ``curl`` subprocess to bypass Cloudflare TLS fingerprinting.
    """
    seen: Dict[str, Dict[str, Any]] = {}

    for url in POLYMARKET_ENDPOINTS:
        print(f"  Fetching: {url[:90]}...")
        data = _curl_get_json(url)
        if data is None:
            continue
        events = data.get("data", [])
        print(f"    Got {len(events)} events.")
        for event in events:
            eid = event.get("id")
            if eid and eid not in seen:
                seen[eid] = event

    print(f"\nTotal unique events: {len(seen)}")
    return list(seen.values())


# ---------------------------------------------------------------------------
# 2. LLM batch filter – which events are futures/commodity-related?
# ---------------------------------------------------------------------------
LLM_BATCH_SIZE = 10  # events per LLM call


def _classify_event_batch(
    batch: List[Dict[str, Any]],
    batch_num: int,
    total_batches: int,
) -> Dict[str, Dict[str, Any]]:
    """Send one batch of events to the LLM and return relevant ones.

    Returns a dict keyed by event ``id`` → ``{"commodities": …, "impact_analysis": …}``.
    """
    event_lines: List[str] = []
    for idx, ev in enumerate(batch, 1):
        eid = ev.get("id", "")
        title = ev.get("title", "")
        desc = (ev.get("description") or "")[:200]
        event_lines.append(f"{idx}. [{eid}] {title} — {desc}")

    events_block = "\n".join(event_lines)

    system_prompt = (
        "你是一位大宗商品与期货市场分析师。下面是一组预测市场事件列表。\n"
        "请判断哪些事件与期货/大宗商品相关——"
        "可以是直接相关（如黄金、原油、铜价预测），也可以是间接相关"
        "（如地缘政治事件可能影响石油、天然气、粮食等大宗商品供应）。\n\n"
        "只返回相关的事件。返回一个JSON数组，每个元素格式如下：\n"
        '[{"id": "事件ID", "commodities": ["受影响商品"], '
        '"impact_analysis": "简要影响分析"}]\n\n'
        "如果没有任何相关事件，返回空数组 []。\n"
        "请用中文回答。只输出JSON，不要输出其他内容。"
    )

    print(f"  Batch {batch_num}/{total_batches} ({len(batch)} events)…")
    try:
        response = llm_client.chat.completions.create(
            model=PM_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": events_block},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return {}

        # Extract JSON array from response
        first_bracket = content.find("[")
        last_bracket = content.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            json_str = content[first_bracket : last_bracket + 1]
        else:
            json_str = content

        items = json.loads(json_str)
        if not isinstance(items, list):
            print("    Warning: LLM did not return a JSON array.")
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for item in items:
            eid = str(item.get("id", ""))
            if eid and item.get("commodities"):
                result[eid] = {
                    "commodities": item["commodities"],
                    "impact_analysis": item.get("impact_analysis", ""),
                }

        print(f"    → {len(result)} relevant in this batch")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"    Error in batch LLM analysis: {exc}")
        return {}


def batch_check_futures_relevance(
    events: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Classify events in batches of ``LLM_BATCH_SIZE`` via LLM.

    Returns a dict keyed by event ``id`` whose values are
    ``{"commodities": [...], "impact_analysis": "..."}`` for relevant events.
    """
    if llm_client is None:
        return {}

    total_batches = (len(events) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    print(f"  {len(events)} events → {total_batches} batch(es) of ≤{LLM_BATCH_SIZE}")

    merged: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(events), LLM_BATCH_SIZE):
        batch = events[i : i + LLM_BATCH_SIZE]
        batch_num = i // LLM_BATCH_SIZE + 1
        partial = _classify_event_batch(batch, batch_num, total_batches)
        merged.update(partial)

    print(f"  LLM identified {len(merged)} futures-related events total.")
    return merged


# ---------------------------------------------------------------------------
# 3. Orderbook fetch & insider-trading detection
# ---------------------------------------------------------------------------
def fetch_orderbook(token_id: str) -> Dict[str, Any]:
    """Fetch the live orderbook from the CLOB API for *token_id*.

    Uses ``curl`` via subprocess because the CLOB API has strict TLS
    fingerprinting that blocks Python HTTP libraries.

    Returns ``{"bids": [{"price":…, "size":…}, …], "asks": […]}`` or empty
    lists on failure.
    """
    data = _curl_get_json(CLOB_BOOK_URL, params={"token_id": token_id})
    if data is None:
        return {"bids": [], "asks": []}
    return {
        "bids": data.get("bids", []),
        "asks": data.get("asks", []),
    }


def _process_orderbook(book: Dict[str, Any]) -> Dict[str, float]:
    """Derive aggregate metrics from a raw orderbook side.

    Returns a dict with: bid/ask top prices, largest sizes, total depths.
    """
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    def _top_price(levels: List[Dict]) -> float:
        if not levels:
            return 0.0
        return float(levels[0].get("price", 0))

    def _largest_size(levels: List[Dict]) -> float:
        if not levels:
            return 0.0
        return max(float(lv.get("size", 0)) for lv in levels)

    def _total_depth(levels: List[Dict]) -> float:
        return sum(float(lv.get("price", 0)) * float(lv.get("size", 0)) for lv in levels)

    bid_depth = _total_depth(bids)
    ask_depth = _total_depth(asks)
    imbalance = bid_depth / ask_depth if ask_depth > 0 else (999.0 if bid_depth > 0 else 1.0)

    top_bid = _top_price(bids)
    top_ask = _top_price(asks)
    spread = (top_ask - top_bid) if (top_ask > 0 and top_bid > 0) else 1.0

    return {
        "top_bid": top_bid,
        "top_ask": top_ask,
        "largest_bid_size": _largest_size(bids),
        "largest_ask_size": _largest_size(asks),
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance_ratio": imbalance,
        "spread": spread,
    }


# ---------------------------------------------------------------------------
# 3b. Price history — detect sudden price movements (historical comparison)
# ---------------------------------------------------------------------------
def fetch_price_history(token_id: str) -> Optional[Dict[str, Any]]:
    """Fetch 24h price history at hourly resolution for *token_id*.

    Returns ``{"avg_24h": float, "recent": float, "change_pct": float}``
    or ``None`` on failure / insufficient data.

    API: GET /prices-history?market=<token>&interval=1d&fidelity=60
    """
    data = _curl_get_json(
        CLOB_PRICES_HISTORY_URL,
        params={"market": token_id, "interval": "1d", "fidelity": "60"},
    )
    if data is None:
        return None
    history = data.get("history", [])
    if len(history) < 3:
        return None

    prices = [float(h.get("p", 0)) for h in history if h.get("p") is not None]
    if not prices:
        return None

    avg_24h = sum(prices) / len(prices)
    # "recent" = average of last 2 data points (~last 1-2 hours)
    recent = sum(prices[-2:]) / len(prices[-2:])

    change_pct = abs(recent - avg_24h) / avg_24h if avg_24h > 0 else 0.0
    return {"avg_24h": avg_24h, "recent": recent, "change_pct": change_pct}


# ---------------------------------------------------------------------------
# 3c. Recent trades — detect trade clustering (same address repeatedly)
# ---------------------------------------------------------------------------
def fetch_recent_trades(condition_id: str) -> Optional[Dict[str, Any]]:
    """Fetch recent trades for a market and analyse wallet clustering.

    Uses the public data-api: GET /trades?market=<conditionId>&limit=200

    Returns ``{"top_wallet_trades": int, "top_wallet_volume_pct": float,
    "unique_wallets": int, "total_trades": int}`` or ``None`` on failure.
    """
    data = _curl_get_json(
        DATA_API_TRADES_URL,
        params={"market": condition_id, "limit": "200", "takerOnly": "false"},
    )
    if data is None or not isinstance(data, list) or len(data) < 2:
        return None

    # Aggregate per-wallet: trade count + total $-volume
    wallet_trade_count: Counter = Counter()
    wallet_volume: defaultdict = defaultdict(float)
    total_volume = 0.0

    for trade in data:
        wallet = trade.get("proxyWallet", "") or ""
        if not wallet:
            continue
        size = float(trade.get("size", 0))
        price = float(trade.get("price", 0))
        dollar_vol = size * price
        wallet_trade_count[wallet] += 1
        wallet_volume[wallet] += dollar_vol
        total_volume += dollar_vol

    if not wallet_trade_count or total_volume <= 0:
        return None

    # Find the most active wallet
    top_wallet, top_count = wallet_trade_count.most_common(1)[0]
    top_vol_pct = wallet_volume[top_wallet] / total_volume if total_volume > 0 else 0.0

    return {
        "top_wallet_trades": top_count,
        "top_wallet_volume_pct": top_vol_pct,
        "unique_wallets": len(wallet_trade_count),
        "total_trades": len(data),
    }


def detect_insider_trading(market: Dict[str, Any]) -> MarketAnalysis:
    """Run the 6-signal insider detection on *market*.

    Signals checked (max 6):
      1. Large Orders       – bid or ask size > 5 % of daily volume (combined as 1)
      2. Order Imbalance    – imbalance ratio > 2.5 or < 0.4
      3. Thin Orderbook     – bid/ask depth < 5 % of daily volume
      4. Aggressive Bidding – tight spread (< 1 %) AND large order present
      5. Sudden Price Move  – price changed > 15 % in last 1 h vs 24 h average
      6. Trade Clustering   – single wallet ≥ 20 trades OR ≥ 40 % of recent volume

    Confidence:
      - high   : 4+ signals, or classic pattern, or behavioural combo
      - medium : 3 signals
      - low    : 0–2 signals

    ``is_insider=True`` requires ≥3 signals.
    """
    question = market.get("question") or market.get("groupItemTitle") or "Unknown"

    # Current price (Yes / No)
    yes_price = 0.0
    no_price = 0.0
    try:
        outcome_prices = json.loads(market.get("outcomePrices", "[]"))
        if len(outcome_prices) >= 1:
            yes_price = float(outcome_prices[0])
        if len(outcome_prices) >= 2:
            no_price = float(outcome_prices[1])
    except (json.JSONDecodeError, IndexError, ValueError):
        pass
    current_price = yes_price

    volume_24h = float(market.get("volume24hr") or 0)
    total_volume = float(market.get("volume") or 0)
    condition_id = market.get("conditionId", "")

    # Derived thresholds based on daily volume
    size_threshold = max(SIZE_THRESHOLD_MIN, volume_24h * SIZE_THRESHOLD_PCT)
    depth_threshold = max(DEPTH_THRESHOLD_MIN, volume_24h * DEPTH_THRESHOLD_PCT)

    # Fetch orderbooks for YES token (first clobTokenId) --------------------
    metrics: Optional[Dict[str, float]] = None
    yes_token: Optional[str] = None
    try:
        clob_token_ids = json.loads(market.get("clobTokenIds", "[]"))
        if clob_token_ids:
            yes_token = clob_token_ids[0]
            yes_book = fetch_orderbook(yes_token)

            # If there's a NO token, fetch it too and merge depths
            no_book: Dict[str, Any] = {"bids": [], "asks": []}
            if len(clob_token_ids) > 1:
                no_book = fetch_orderbook(clob_token_ids[1])

            yes_m = _process_orderbook(yes_book)
            no_m = _process_orderbook(no_book)

            # Combine: use YES side as primary, add NO depth for fuller picture
            metrics = {
                "top_bid": yes_m["top_bid"],
                "top_ask": yes_m["top_ask"],
                "largest_bid_size": max(yes_m["largest_bid_size"], no_m["largest_ask_size"]),
                "largest_ask_size": max(yes_m["largest_ask_size"], no_m["largest_bid_size"]),
                "bid_depth": yes_m["bid_depth"] + no_m["ask_depth"],
                "ask_depth": yes_m["ask_depth"] + no_m["bid_depth"],
                "spread": yes_m["spread"],
            }
            total_bid = metrics["bid_depth"]
            total_ask = metrics["ask_depth"]
            metrics["imbalance_ratio"] = (
                total_bid / total_ask if total_ask > 0 else (999.0 if total_bid > 0 else 1.0)
            )
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"      Error processing orderbook: {exc}")

    # If we couldn't get orderbook data, return a clean "no data" result
    if metrics is None:
        return MarketAnalysis(
            market_question=question,
            current_price=current_price,
            yes_price=yes_price,
            no_price=no_price,
            volume_24h=volume_24h,
            total_volume=total_volume,
            is_insider=False,
            confidence="low",
            signals=["订单簿数据不可用"],
            details=InsiderDetails(),
        )

    # --- Signal detection (7 signals) --------------------------------------
    signals: List[str] = []
    det = InsiderDetails(
        largest_bid_size=metrics["largest_bid_size"],
        largest_ask_size=metrics["largest_ask_size"],
        imbalance_ratio=metrics["imbalance_ratio"],
        bid_depth=metrics["bid_depth"],
        ask_depth=metrics["ask_depth"],
        spread=metrics["spread"],
    )

    # 1 & 2. Large Orders (bid and/or ask — counted as ONE combined signal)
    large_bid = det.largest_bid_size > size_threshold
    large_ask = det.largest_ask_size > size_threshold
    if large_bid:
        det.large_bid_order = True
    if large_ask:
        det.large_ask_order = True
    if large_bid and large_ask:
        signals.append(
            f"大额挂单: 买 ${det.largest_bid_size:,.0f} / "
            f"卖 ${det.largest_ask_size:,.0f} "
            f"(阈值 ${size_threshold:,.0f})"
        )
    elif large_bid:
        signals.append(
            f"大额买单: ${det.largest_bid_size:,.0f} "
            f"(阈值 ${size_threshold:,.0f})"
        )
    elif large_ask:
        signals.append(
            f"大额卖单: ${det.largest_ask_size:,.0f} "
            f"(阈值 ${size_threshold:,.0f})"
        )

    # 3. Order Imbalance
    if det.imbalance_ratio > IMBALANCE_HIGH:
        det.order_imbalance = True
        signals.append(f"订单失衡(买方主导): {det.imbalance_ratio:.1f}×")
    elif det.imbalance_ratio < IMBALANCE_LOW:
        det.order_imbalance = True
        signals.append(f"订单失衡(卖方主导): {det.imbalance_ratio:.2f}×")

    # 4. Thin Orderbook
    if det.bid_depth < depth_threshold or det.ask_depth < depth_threshold:
        det.thin_orderbook = True
        thin_side = "买方" if det.bid_depth < det.ask_depth else "卖方"
        thin_val = min(det.bid_depth, det.ask_depth)
        signals.append(
            f"订单簿薄弱({thin_side}): ${thin_val:,.0f} "
            f"(阈值 ${depth_threshold:,.0f})"
        )

    # 5. Aggressive Bidding – tight spread + large order present
    mid_price = (metrics["top_bid"] + metrics["top_ask"]) / 2 if metrics["top_ask"] > 0 else 1.0
    spread_pct = det.spread / mid_price if mid_price > 0 else 1.0
    has_large_order = det.large_bid_order or det.large_ask_order
    if spread_pct < TIGHT_SPREAD_PCT and has_large_order:
        det.aggressive_bidding = True
        signals.append(f"激进报价: 价差 {spread_pct:.2%} + 大额挂单")

    # 6. Sudden Price Move – historical comparison
    if yes_token:
        ph = fetch_price_history(yes_token)
        if ph is not None:
            det.price_24h_avg = ph["avg_24h"]
            det.price_recent = ph["recent"]
            det.price_change_pct = ph["change_pct"]
            if ph["change_pct"] >= PRICE_CHANGE_THRESHOLD:
                det.sudden_price_move = True
                direction = "↑" if ph["recent"] > ph["avg_24h"] else "↓"
                signals.append(
                    f"价格突变{direction}: 近期 {ph['recent']:.3f} vs "
                    f"24h均值 {ph['avg_24h']:.3f} ({ph['change_pct']:.1%})"
                )

    # 7. Trade Clustering – same address buying repeatedly
    if condition_id:
        tc = fetch_recent_trades(condition_id)
        if tc is not None:
            det.top_wallet_trades = tc["top_wallet_trades"]
            det.top_wallet_volume_pct = tc["top_wallet_volume_pct"]
            det.unique_wallets = tc["unique_wallets"]
            is_cluster = (
                tc["top_wallet_trades"] >= CLUSTER_MIN_TRADES
                or tc["top_wallet_volume_pct"] >= CLUSTER_VOLUME_PCT
            )
            if is_cluster:
                det.trade_clustering = True
                signals.append(
                    f"交易聚集: 最活跃钱包 {tc['top_wallet_trades']} 笔交易, "
                    f"占总量 {tc['top_wallet_volume_pct']:.0%} "
                    f"(共 {tc['unique_wallets']} 个钱包/{tc['total_trades']} 笔)"
                )

    # --- Confidence calculation --------------------------------------------
    n_signals = len(signals)
    if n_signals >= 4:
        confidence = "high"
    elif n_signals >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    # Override to HIGH for the classic insider pattern
    if (
        det.large_bid_order
        and det.thin_orderbook
        and det.order_imbalance
        and det.imbalance_ratio > 3.0
    ):
        confidence = "high"

    # Override to HIGH: sudden price move + trade clustering = very suspicious
    if det.sudden_price_move and det.trade_clustering:
        confidence = "high"

    # Override to HIGH: behavioural signal + strong orderbook anomaly
    if (det.sudden_price_move or det.trade_clustering) and n_signals >= 3:
        confidence = "high"

    is_insider = n_signals >= 3  # require 3+ signals to flag as insider

    return MarketAnalysis(
        market_question=question,
        current_price=current_price,
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=volume_24h,
        total_volume=total_volume,
        is_insider=is_insider,
        confidence=confidence,
        signals=signals,
        details=det,
    )


# ---------------------------------------------------------------------------
# 4. HTML email report
# ---------------------------------------------------------------------------
def _confidence_badge(confidence: str) -> str:
    """Return an HTML badge for the confidence level."""
    colours = {
        "high": ("#d32f2f", "#fff"),
        "medium": ("#ff9800", "#fff"),
        "low": ("#9e9e9e", "#fff"),
    }
    bg, fg = colours.get(confidence, ("#9e9e9e", "#fff"))
    label = {"high": "高", "medium": "中", "low": "低"}.get(confidence, confidence)
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:3px;"
        f"background:{bg};color:{fg};font-size:11px;font-weight:600;'>"
        f"置信度: {label}</span>"
    )


def build_html_report(events: List[FuturesEvent]) -> str:
    """Build a styled HTML report with orderbook-based insider detection."""
    if not events:
        return (
            "<html><body>"
            "<h2>Polymarket期货监控 — 未检测到相关事件</h2>"
            "</body></html>"
        )

    event_rows: List[str] = []
    for event in events:
        # Build per-market cards
        market_cards: List[str] = []
        for m in event.markets:
            d = m.details

            # Header icon
            if m.is_insider and m.confidence == "high":
                status_icon = "🔴"
                status_text = "高度疑似内幕交易"
            elif m.is_insider:
                status_icon = "🟠"
                status_text = "疑似内幕交易"
            elif m.signals and m.signals != ["订单簿数据不可用"]:
                status_icon = "⚠️"
                status_text = "存在异常信号"
            else:
                status_icon = "✅"
                status_text = "正常"

            bg = "#fff3f3" if m.is_insider else "#f9f9f9"
            border = (
                "#d32f2f" if m.confidence == "high" and m.is_insider
                else "#ff9800" if m.is_insider
                else "#4caf50" if not m.signals or m.signals == ["订单簿数据不可用"]
                else "#ff9800"
            )

            # Signal list
            signal_items = ""
            for sig in m.signals:
                signal_items += f"<li>{sig}</li>"
            signal_html = (
                f"<ul style='margin:4px 0 0 16px;padding:0;font-size:12px;'>"
                f"{signal_items}</ul>"
            ) if signal_items else ""

            # Orderbook metrics bar
            metrics_parts: List[str] = []
            if d.bid_depth > 0 or d.ask_depth > 0:
                metrics_parts.append(
                    f"买深: ${d.bid_depth:,.0f} | "
                    f"卖深: ${d.ask_depth:,.0f} | "
                    f"失衡: {d.imbalance_ratio:.2f}× | "
                    f"价差: {d.spread:.4f}"
                )
            if d.price_24h_avg > 0:
                chg_color = "#d32f2f" if d.sudden_price_move else "#888"
                metrics_parts.append(
                    f"<span style='color:{chg_color};'>价格: "
                    f"{d.price_recent:.3f} vs 24h均值 {d.price_24h_avg:.3f} "
                    f"({d.price_change_pct:.1%})</span>"
                )
            if d.unique_wallets > 0:
                cl_color = "#d32f2f" if d.trade_clustering else "#888"
                metrics_parts.append(
                    f"<span style='color:{cl_color};'>钱包: "
                    f"最活跃 {d.top_wallet_trades} 笔 "
                    f"({d.top_wallet_volume_pct:.0%}), "
                    f"共 {d.unique_wallets} 个</span>"
                )
            metrics_html = (
                "<div style='font-size:11px;color:#888;margin-top:6px;'>"
                + " | ".join(metrics_parts)
                + "</div>"
            ) if metrics_parts else ""

            market_cards.append(
                f"<div style='margin-bottom:10px;padding:10px;background:{bg};"
                f"border-radius:4px;border-left:4px solid {border};'>"
                f"<div style='font-weight:600;'>"
                f"{status_icon} {m.market_question} "
                f"{_confidence_badge(m.confidence)}</div>"
                f"<div style='font-size:12px;color:#666;margin-top:4px;'>"
                f"<span style='color:#2e7d32;font-weight:600;'>Yes {m.yes_price:.0%}</span>"
                f" / "
                f"<span style='color:#c62828;font-weight:600;'>No {m.no_price:.0%}</span>"
                f" &nbsp;|&nbsp; "
                f"24h量: ${m.volume_24h:,.0f} | "
                f"总量: ${m.total_volume:,.0f}"
                f"</div>"
                f"<div style='margin-top:4px;height:8px;border-radius:4px;"
                f"background:linear-gradient(to right,"
                f"#4caf50 {m.yes_price:.0%},#ef5350 {m.yes_price:.0%});'>"
                f"</div>"
                f"<div style='font-size:12px;margin-top:4px;'>"
                f"<strong>{status_text}</strong>"
                f"</div>"
                f"{signal_html}"
                f"{metrics_html}"
                f"</div>"
            )

        markets_html = "\n".join(market_cards)
        commodities_str = "、".join(event.related_commodities)
        flag = "🔴" if event.has_insider_signal else ""

        desc_short = (event.description or "")[:200]
        if len(event.description or "") > 200:
            desc_short += "…"

        event_rows.append(
            f"<tr>"
            f"<td style='vertical-align:top;'>"
            f"<strong>{flag} {event.title}</strong><br/>"
            f"<span style='color:#666;font-size:12px;'>{desc_short}</span></td>"
            f"<td style='vertical-align:top;'>{commodities_str}</td>"
            f"<td style='vertical-align:top;'>{event.impact_analysis}</td>"
            f"<td>{markets_html}</td>"
            f"</tr>"
        )

    table_body = "\n".join(event_rows)

    insider_count = sum(1 for e in events if e.has_insider_signal)
    high_count = sum(
        1 for e in events for m in e.markets
        if m.is_insider and m.confidence == "high"
    )
    signal_count = sum(
        1 for e in events for m in e.markets if m.signals and m.signals != ["订单簿数据不可用"]
    )

    html = f"""\
<html>
  <head>
    <meta charset="utf-8"/>
    <style>
      body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
      table {{ border-collapse:collapse; width:100%; }}
      th, td {{ border:1px solid #ddd; padding:12px; font-size:13px; }}
      th {{ background-color:#1a237e; color:white; font-weight:bold; }}
      a {{ color:#0066cc; text-decoration:none; }}
      a:hover {{ text-decoration:underline; }}
      .summary {{ background:#f5f5f5; padding:15px; border-radius:8px; margin-bottom:20px; }}
      .legend {{ font-size:12px; color:#666; margin-top:10px; }}
    </style>
  </head>
  <body>
    <h2>Polymarket 期货事件 — 内幕交易监控</h2>
    <div class="summary">
      <strong>监控概览：</strong>
      扫描 {len(events)} 个期货相关事件，
      {signal_count} 个市场存在异常订单簿信号，
      {insider_count} 个事件触发内幕交易预警
      （其中 {high_count} 个为高置信度）。
    </div>
    <div class="legend">
      <strong>检测方法：</strong>订单簿分析 — 大额挂单(&gt;1%日成交量)、
      订单失衡(&gt;2.5×)、订单簿薄弱(&lt;5%日成交量)、激进报价(价差&lt;1%+大单)、
      价格突变(&gt;15% vs 24h均值)、交易聚集(同一钱包≥3笔或≥30%成交量)<br/>
      <strong>图例：</strong>
      🔴 高置信度内幕交易 |
      🟠 中等置信度 |
      ⚠️ 存在异常信号 |
      ✅ 正常
    </div>
    <table>
      <thead>
        <tr>
          <th style='width:25%;'>事件</th>
          <th style='width:10%;'>相关商品</th>
          <th style='width:20%;'>影响分析</th>
          <th style='width:45%;'>订单簿分析</th>
        </tr>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
  </body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# 5. Email sending (same pattern as geopolitical_supply_chain_monitor.py)
# ---------------------------------------------------------------------------
def send_email_report(subject: str, html_content: str) -> None:
    """Send the HTML report via email using .env configuration."""
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


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------
def run_monitor() -> None:
    """Run the full Polymarket futures monitoring pipeline."""
    print("=" * 60)
    print("Polymarket Futures Monitor")
    print(f"Run time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Step 1 – fetch events
    print("\n[Step 1] Fetching Polymarket events…")
    raw_events = fetch_polymarket_events()
    if not raw_events:
        print("No events fetched. Exiting.")
        return

    # Step 2 – single LLM call to classify all events at once
    print("\n[Step 2] Batch-filtering futures-related events via LLM (1 call)…")
    relevance_map = batch_check_futures_relevance(raw_events)

    # Step 3 – orderbook insider detection on relevant events only
    print(f"\n[Step 3] Analysing orderbooks for {len(relevance_map)} relevant events…")
    futures_events: List[FuturesEvent] = []

    for event in raw_events:
        eid = event.get("id", "")
        analysis = relevance_map.get(eid)
        if analysis is None:
            continue

        title = event.get("title", "")
        print(f"\n  {title}")
        print(f"    Commodities: {analysis['commodities']}")

        # Pick top 5 markets by 24h volume
        raw_markets = event.get("markets", [])
        raw_markets_sorted = sorted(
            raw_markets,
            key=lambda m: float(m.get("volume24hr") or 0),
            reverse=True,
        )
        top_markets = raw_markets_sorted[:3]
        if len(raw_markets) > 3:
            print(f"    ({len(raw_markets)} markets, analysing top 3 by volume)")

        market_results: List[MarketAnalysis] = []
        has_insider = False
        for mkt in top_markets:
            q = (mkt.get("question") or "")[:60]
            print(f"    Orderbook: {q}")
            result = detect_insider_trading(mkt)
            market_results.append(result)
            if result.is_insider:
                has_insider = True
                print(
                    f"      ⚠ INSIDER [{result.confidence}]: "
                    f"{', '.join(result.signals)}"
                )

        futures_events.append(
            FuturesEvent(
                event_id=eid,
                title=title,
                description=event.get("description", ""),
                slug=event.get("slug", ""),
                image_url=event.get("image", ""),
                related_commodities=analysis["commodities"],
                impact_analysis=analysis["impact_analysis"],
                markets=market_results,
                has_insider_signal=has_insider,
            )
        )

    print(f"\n[Result] Found {len(futures_events)} futures-related events.")

    # Step 4 – build & send email
    print("\n[Step 4] Building and sending email report…")
    html = build_html_report(futures_events)

    insider_count = sum(1 for e in futures_events if e.has_insider_signal)
    high_count = sum(
        1 for e in futures_events for m in e.markets
        if m.is_insider and m.confidence == "high"
    )
    subject = f"Polymarket期货监控 — {len(futures_events)}个相关事件"
    if insider_count > 0:
        subject += f" | 🔴 {insider_count}个内幕交易信号"
    if high_count > 0:
        subject += f"（{high_count}个高置信度）"

    send_email_report(subject, html)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total events scanned : {len(raw_events)}")
    print(f"  Futures-related      : {len(futures_events)}")
    print(f"  Insider signals      : {insider_count}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_monitor()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
