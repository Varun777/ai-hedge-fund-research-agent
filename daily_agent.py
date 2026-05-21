import os
import json
import requests
import feedparser
import yfinance as yf

from rapidfuzz import fuzz, process
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NEWS_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

HIGH_SIGNAL_KEYWORDS = [
    "earnings", "guidance", "upgrade", "downgrade", "acquisition",
    "merger", "buyout", "lawsuit", "ai", "partnership", "surge",
    "plunge", "beats", "misses", "raises", "cuts", "forecast",
    "outlook", "tariff", "layoffs", "restructuring", "contract",
    "market share", "deliveries", "sales", "reported", "results"
]

def normalize_name(name):
    if not name:
        return ""

    name = name.lower()

    remove_words = [
        "inc.", "inc", "corporation", "corp.", "corp", "company", "co.",
        "co", "plc", "ltd.", "ltd", "limited", "class a", "class b",
        "common stock", "ordinary shares", "american depositary shares",
        "adr", "holdings", "holding", "group", "the"
    ]

    for word in remove_words:
        name = name.replace(word, "")

    return " ".join(name.replace(",", " ").replace(".", " ").split())

def get_news():
    articles = []

    for feed_url in NEWS_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:30]:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", "")
            })

    return articles

def score_article(article):
    score = 0
    text = (article["title"] + " " + article["summary"]).lower()

    scoring_rules = {
        "earnings": 5,
        "reported": 4,
        "results": 4,
        "beats": 4,
        "misses": 4,
        "guidance": 4,
        "raises": 3,
        "cuts": 3,
        "upgrade": 3,
        "downgrade": 3,
        "acquisition": 3,
        "merger": 3,
        "buyout": 3,
        "surge": 3,
        "plunge": 3,
        "shares jumped": 3,
        "shares fell": 3,
        "lawsuit": 2,
        "restructuring": 2,
        "contract": 2,
        "market share": 2,
        "deliveries": 2,
        "sales": 1,
        "ai": 1,
    }

    for keyword, points in scoring_rules.items():
        if keyword in text:
            score += points

    return score

def classify_earnings_timing(article):
    text = (article["title"] + " " + article["summary"]).lower()

    if any(word in text for word in [
        "reported", "reports", "posted", "beat", "beats",
        "missed", "misses", "results"
    ]):
        return "Post-earnings reaction or reported-results setup"

    if any(word in text for word in [
        "ahead of earnings", "before earnings", "countdown to",
        "expected to report", "will report"
    ]):
        return "Pre-earnings setup"

    if "earnings" in text:
        return "Earnings-related; timing needs verification"

    return "Not earnings-driven"

def filter_high_signal_news(articles):
    filtered = []
    seen_titles = set()

    for article in articles:
        title_key = article["title"].strip().lower()
        text = (article["title"] + " " + article["summary"]).lower()

        if title_key in seen_titles:
            continue

        if any(keyword in text for keyword in HIGH_SIGNAL_KEYWORDS):
            article["signal_score"] = score_article(article)
            article["earnings_timing"] = classify_earnings_timing(article)
            filtered.append(article)
            seen_titles.add(title_key)

    return sorted(filtered, key=lambda x: x["signal_score"], reverse=True)[:15]

def load_security_master():
    securities = []

    nasdaq_text = requests.get(NASDAQ_LISTED_URL, timeout=20).text
    other_text = requests.get(OTHER_LISTED_URL, timeout=20).text

    for line in nasdaq_text.splitlines()[1:]:
        parts = line.split("|")

        if len(parts) < 7 or "File Creation Time" in line:
            continue

        symbol = parts[0]
        name = parts[1]
        test_issue = parts[3]
        etf = parts[6]

        if test_issue == "Y" or etf == "Y":
            continue

        securities.append({
            "symbol": symbol,
            "name": name,
            "normalized_name": normalize_name(name),
            "exchange": "NASDAQ"
        })

    for line in other_text.splitlines()[1:]:
        parts = line.split("|")

        if len(parts) < 7 or "File Creation Time" in line:
            continue

        symbol = parts[0]
        name = parts[1]
        exchange = parts[2]
        etf = parts[4]
        test_issue = parts[6]

        if test_issue == "Y" or etf == "Y":
            continue

        securities.append({
            "symbol": symbol,
            "name": name,
            "normalized_name": normalize_name(name),
            "exchange": exchange
        })

    return securities

def extract_companies_with_llm(news):
    prompt = f"""
Extract public companies mentioned in the following news articles.

Return JSON only. No markdown. No commentary.

Rules:
- Return only company names that appear in the news text.
- Do not guess.
- Do not include governments, countries, indexes, currencies, or sectors.
- If private company, mark publicly_traded_likely false.
- Include source link, signal score, and earnings timing.

Format:
[
  {{
    "company_name": "Nvidia",
    "publicly_traded_likely": true,
    "source_link": "https://example.com",
    "signal_score": 7,
    "earnings_timing": "Post-earnings reaction or reported-results setup"
  }}
]

News:
{news}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    try:
        return json.loads(response.output_text.strip())
    except json.JSONDecodeError:
        print("Could not parse JSON")
        print(response.output_text)
        return []

def resolve_company_with_security_master(company_name, securities):
    if not company_name or len(company_name.strip()) <= 1:
        return None

    normalized_query = normalize_name(company_name)

    if not normalized_query or len(normalized_query) <= 1:
        return None

    choices = {
        security["normalized_name"]: security
        for security in securities
        if security["normalized_name"]
    }

    match = process.extractOne(
        normalized_query,
        choices.keys(),
        scorer=fuzz.token_set_ratio
    )

    if not match:
        return None

    matched_name, score, _ = match
    security = choices[matched_name]

    if score < 82:
        print(f"Rejected low-confidence match: {company_name} -> {security['name']} ({score})")
        return None

    return {
        "company_name_from_news": company_name,
        "resolved_ticker": security["symbol"],
        "resolved_name": security["name"],
        "exchange": security["exchange"],
        "match_score": score
    }

def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return any([
            info.get("currentPrice") is not None,
            info.get("regularMarketPrice") is not None,
            info.get("previousClose") is not None,
            info.get("marketCap") is not None
        ])

    except Exception:
        return False

def get_market_reaction(history):
    if history is None or history.empty or len(history) < 21:
        return {
            "one_day_return_percent": None,
            "five_day_return_percent": None,
            "twenty_day_return_percent": None,
            "distance_from_six_month_high_percent": None,
            "distance_from_six_month_low_percent": None
        }

    closes = history["Close"]
    last_close = closes.iloc[-1]

    one_day_return = round(((last_close - closes.iloc[-2]) / closes.iloc[-2]) * 100, 2)
    five_day_return = round(((last_close - closes.iloc[-6]) / closes.iloc[-6]) * 100, 2)
    twenty_day_return = round(((last_close - closes.iloc[-21]) / closes.iloc[-21]) * 100, 2)

    six_month_high = history["High"].max()
    six_month_low = history["Low"].min()

    distance_from_high = round(((last_close - six_month_high) / six_month_high) * 100, 2)
    distance_from_low = round(((last_close - six_month_low) / six_month_low) * 100, 2)

    return {
        "one_day_return_percent": one_day_return,
        "five_day_return_percent": five_day_return,
        "twenty_day_return_percent": twenty_day_return,
        "distance_from_six_month_high_percent": distance_from_high,
        "distance_from_six_month_low_percent": distance_from_low
    }

def classify_entry_type(market_reaction):
    one_day = market_reaction.get("one_day_return_percent")
    five_day = market_reaction.get("five_day_return_percent")
    twenty_day = market_reaction.get("twenty_day_return_percent")
    distance_high = market_reaction.get("distance_from_six_month_high_percent")
    distance_low = market_reaction.get("distance_from_six_month_low_percent")

    if one_day is None or five_day is None or twenty_day is None:
        return "Needs more price history"

    if twenty_day > 8 and distance_high > -8:
        return "Breakout continuation"

    if twenty_day > 8 and five_day < -2:
        return "Pullback entry in uptrend"

    if twenty_day < -8 and distance_low < 10:
        return "Mean reversion or falling knife"

    if abs(one_day) > 5:
        return "Post-news reaction / overreaction check"

    if five_day > 3 and twenty_day > 5:
        return "Momentum continuation"

    return "Neutral setup"

def compute_dynamic_signal_score(base_score, market_data):
    score = base_score or 0

    market_cap = market_data.get("market_cap") or 0
    avg_volume = market_data.get("average_volume") or 0
    one_day = market_data.get("one_day_return_percent") or 0
    five_day = market_data.get("five_day_return_percent") or 0
    twenty_day = market_data.get("twenty_day_return_percent") or 0
    match_score = market_data.get("ticker_match_score") or 0

    if market_cap > 1_000_000_000_000:
        score += 4
    elif market_cap > 100_000_000_000:
        score += 3
    elif market_cap > 10_000_000_000:
        score += 2
    elif market_cap > 1_000_000_000:
        score += 1

    if avg_volume > 20_000_000:
        score += 3
    elif avg_volume > 5_000_000:
        score += 2
    elif avg_volume > 1_000_000:
        score += 1

    if abs(one_day) >= 5:
        score += 3
    elif abs(one_day) >= 2:
        score += 1

    if abs(five_day) >= 7:
        score += 3
    elif abs(five_day) >= 3:
        score += 1

    if abs(twenty_day) >= 12:
        score += 2

    if match_score >= 95:
        score += 1

    return score

def get_market_data_for_ticker(
    ticker,
    original_company_name=None,
    source_link=None,
    signal_score=None,
    match_score=None,
    earnings_timing=None
):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="6mo")

        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        six_month_high = None
        six_month_low = None
        six_month_return = None

        if not history.empty:
            six_month_high = round(history["High"].max(), 2)
            six_month_low = round(history["Low"].min(), 2)

            first_close = history["Close"].iloc[0]
            last_close = history["Close"].iloc[-1]

            if first_close:
                six_month_return = round(((last_close - first_close) / first_close) * 100, 2)

        market_reaction = get_market_reaction(history)
        entry_type = classify_entry_type(market_reaction)

        market_data = {
            "original_company_name_from_news": original_company_name,
            "ticker": ticker,
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "current_price": current_price,
            "market_cap": info.get("marketCap"),
            "average_volume": info.get("averageVolume"),
            "float_shares": info.get("floatShares"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            "operating_margins": info.get("operatingMargins"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "six_month_high": six_month_high,
            "six_month_low": six_month_low,
            "six_month_return_percent": six_month_return,
            "previous_close": info.get("previousClose"),
            "one_day_return_percent": market_reaction["one_day_return_percent"],
            "five_day_return_percent": market_reaction["five_day_return_percent"],
            "twenty_day_return_percent": market_reaction["twenty_day_return_percent"],
            "distance_from_six_month_high_percent": market_reaction["distance_from_six_month_high_percent"],
            "distance_from_six_month_low_percent": market_reaction["distance_from_six_month_low_percent"],
            "entry_type": entry_type,
            "source_link": source_link,
            "signal_score": signal_score,
            "ticker_match_score": match_score,
            "earnings_timing": earnings_timing
        }

        market_data["dynamic_signal_score"] = compute_dynamic_signal_score(
            signal_score,
            market_data
        )

        return market_data

    except Exception as e:
        print(f"Error fetching market data for {ticker}: {e}")
        return None

def build_context(news, market_data, rejected_companies):
    market_data = sorted(
        market_data,
        key=lambda x: x.get("dynamic_signal_score", 0),
        reverse=True
    )

    return {
        "news": news,
        "market_data": market_data,
        "allowed_tickers": [item["ticker"] for item in market_data],
        "rejected_companies": rejected_companies
    }

def build_prompt(context):
    return f"""
You are a hedge fund investment committee.

This is NOT financial advice.
These are research ideas for further diligence only.

CRITICAL RULES:
- ONLY generate ideas for tickers in allowed_tickers.
- Do NOT introduce new tickers.
- Do NOT generate an idea if ticker_match_score < 82.
- ONLY use numbers present in market_data.
- Do NOT invent price targets, returns, growth rates, probabilities, or valuation multiples.
- Do NOT use historical claims unless explicitly present in market_data or news.
- Do NOT say "historically tends to", "has consistently", "usually", or "traditionally" unless directly supported by provided context.
- Never say "verified market data." Say "available market data."
- Prefer 1-3 excellent ideas over 5 mediocre ideas.
- Every idea must have a clear catalyst within 1-3 months.
- Exclude speculative or weak ideas.

CLAIM DISCIPLINE:
Every important claim must be labeled as one of:
- Market data fact
- News-derived fact
- Inference
- Hypothesis

Do not present inferences or hypotheses as facts.

SIGNAL RULE:
- Use dynamic_signal_score as the primary ranking input.
- Explain why high-ranking ideas beat lower-ranking alternatives.
- Use market cap, liquidity, price reaction, and ticker confidence as signal quality inputs.

MARKET REACTION RULE:
- Use one_day_return_percent, five_day_return_percent, and twenty_day_return_percent.
- Use entry_type.
- Determine whether this is breakout continuation, pullback entry, mean reversion, post-news reaction, overreaction fade, or neutral setup.
- Do not call something mispriced unless the price reaction supports that view.

ENTRY / EXIT RULES:
- Entry zones must come from actual market_data.
- No fake precision.
- No invented targets.
- Target thesis should be logic-based only.

Provided context:
{context}

FORMAT:

# Daily Investment Thesis Memo

## Executive Summary

## Idea 1: Ticker / Company

News link:
Signal score:
Dynamic signal score:
Ticker match score:
Earnings timing:
Entry type:

Market reaction:

Market is pricing in:
What may actually be true:
Why the gap exists:
Evidence that closes the gap:
Why this is better than alternatives:

Claim discipline:
- Market data facts:
- News-derived facts:
- Inferences:
- Hypotheses:

Edge:
Why now:
Catalyst:

Fundamental setup:
Valuation setup:
Liquidity / tradability:
Entry zone:
Target thesis:
Stop / invalidation:
Time horizon:
Bear case:
Risk / reward:
Next diligence:
Data confidence:
Conviction score:
Status:

## Excluded / Rejected Ideas

## Portfolio Ranking
"""

def generate_report():
    print("Fetching news...")
    raw_news = get_news()

    print("Filtering and ranking news...")
    news = filter_high_signal_news(raw_news)

    if not news:
        news = raw_news[:15]

    print("Loading security master...")
    securities = load_security_master()

    print("Extracting companies with AI...")
    companies = extract_companies_with_llm(news)

    print("Resolving companies...")
    resolved_companies = []
    rejected_companies = []
    seen_tickers = set()

    companies = sorted(
        companies,
        key=lambda x: x.get("signal_score", 0),
        reverse=True
    )

    for company in companies:
        company_name = company.get("company_name")
        source_link = company.get("source_link")
        signal_score = company.get("signal_score", 0)
        earnings_timing = company.get("earnings_timing", "Not earnings-driven")

        if not company_name:
            continue

        resolved = resolve_company_with_security_master(company_name, securities)

        if not resolved:
            rejected_companies.append({
                "company_name": company_name,
                "reason": "Ticker resolution failed",
                "source_link": source_link
            })
            continue

        ticker = resolved["resolved_ticker"]

        if ticker in seen_tickers:
            continue

        if validate_ticker(ticker):
            resolved["source_link"] = source_link
            resolved["signal_score"] = signal_score
            resolved["earnings_timing"] = earnings_timing
            resolved_companies.append(resolved)
            seen_tickers.add(ticker)
        else:
            rejected_companies.append({
                "company_name": company_name,
                "reason": "Ticker validation failed",
                "source_link": source_link
            })

    print("Fetching market data...")
    market_data = []

    for item in resolved_companies[:12]:
        data = get_market_data_for_ticker(
            ticker=item["resolved_ticker"],
            original_company_name=item["company_name_from_news"],
            source_link=item.get("source_link"),
            signal_score=item.get("signal_score"),
            match_score=item.get("match_score"),
            earnings_timing=item.get("earnings_timing")
        )

        if data:
            market_data.append(data)

    context = build_context(news, market_data, rejected_companies)

    print("Generating investment memo...")
    prompt = build_prompt(context)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    report = response.output_text

    with open("daily_memo.md", "w") as file:
        file.write(report)

    print("\n===== DAILY INVESTMENT MEMO =====\n")
    print(report)

if __name__ == "__main__":
    generate_report()