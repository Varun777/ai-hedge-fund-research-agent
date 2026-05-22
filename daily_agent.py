import os
import json
import requests
import feedparser
import yfinance as yf
import smtplib

from rapidfuzz import fuzz, process
from dotenv import load_dotenv
from openai import OpenAI
from email.mime.text import MIMEText
from datetime import datetime

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
        return "Post-earnings reaction"

    if any(word in text for word in [
        "ahead of earnings", "before earnings", "countdown to",
        "expected to report", "will report"
    ]):
        return "Pre-earnings setup"

    if "earnings" in text:
        return "Earnings timing unclear"

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
            "normalized_name": normalize_name(name)
        })

    for line in other_text.splitlines()[1:]:
        parts = line.split("|")

        if len(parts) < 7 or "File Creation Time" in line:
            continue

        symbol = parts[0]
        name = parts[1]
        etf = parts[4]
        test_issue = parts[6]

        if test_issue == "Y" or etf == "Y":
            continue

        securities.append({
            "symbol": symbol,
            "name": name,
            "normalized_name": normalize_name(name)
        })

    return securities


def extract_companies_with_llm(news):
    prompt = f"""
Extract public companies mentioned in the following news.

Return ONLY valid JSON.

Rules:
- No markdown
- No commentary
- Only companies explicitly mentioned
- Include signal score
- Include earnings timing

Format:

[
  {{
    "company_name": "Nvidia",
    "signal_score": 8,
    "earnings_timing": "Post-earnings reaction"
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
    except Exception:
        print("Failed to parse company extraction")
        print(response.output_text)
        return []


def resolve_company_with_security_master(company_name, securities):
    normalized_query = normalize_name(company_name)

    if not normalized_query:
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
        "ticker": security["symbol"],
        "company_name": security["name"],
        "match_score": score
    }


def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get("marketCap") is not None
    except Exception:
        return False


def get_market_reaction(history):
    if history.empty or len(history) < 21:
        return {}

    closes = history["Close"]
    last_close = closes.iloc[-1]

    return {
        "one_day_return_percent": round(((last_close - closes.iloc[-2]) / closes.iloc[-2]) * 100, 2),
        "five_day_return_percent": round(((last_close - closes.iloc[-6]) / closes.iloc[-6]) * 100, 2),
        "twenty_day_return_percent": round(((last_close - closes.iloc[-21]) / closes.iloc[-21]) * 100, 2),
    }


def classify_entry_type(reaction):
    one_day = reaction.get("one_day_return_percent", 0)
    five_day = reaction.get("five_day_return_percent", 0)
    twenty_day = reaction.get("twenty_day_return_percent", 0)

    if twenty_day > 10:
        return "Momentum continuation"

    if five_day < -3 and twenty_day > 5:
        return "Pullback entry"

    if twenty_day < -10:
        return "Mean reversion or falling knife"

    if abs(one_day) > 5:
        return "Post-news reaction"

    return "Neutral"


def compute_dynamic_signal_score(base_score, market_data):
    score = base_score or 0

    market_cap = market_data.get("market_cap") or 0
    avg_volume = market_data.get("average_volume") or 0
    one_day = abs(market_data.get("one_day_return_percent") or 0)

    if market_cap > 1_000_000_000_000:
        score += 4
    elif market_cap > 100_000_000_000:
        score += 3
    elif market_cap > 10_000_000_000:
        score += 2

    if avg_volume > 20_000_000:
        score += 3
    elif avg_volume > 5_000_000:
        score += 2
    elif avg_volume > 1_000_000:
        score += 1

    if one_day > 5:
        score += 3
    elif one_day > 2:
        score += 1

    return score


def get_market_data_for_ticker(ticker, signal_score, earnings_timing):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="6mo")

        reaction = get_market_reaction(history)

        market_data = {
            "ticker": ticker,
            "company_name": info.get("longName"),
            "market_cap": info.get("marketCap"),
            "average_volume": info.get("averageVolume"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"),
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            "operating_margins": info.get("operatingMargins"),
            "earnings_timing": earnings_timing,
            "signal_score": signal_score,
            "entry_type": classify_entry_type(reaction),
            **reaction
        }

        market_data["dynamic_signal_score"] = compute_dynamic_signal_score(signal_score, market_data)

        return market_data

    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


def build_prompt(context):
    return f"""
You are a hedge fund investment committee.

This is NOT financial advice.

Only use facts from provided data.

Never invent:
- historical tendencies
- analyst sentiment
- management quality
- price targets
- probabilities

Every claim must be labeled:
- Market data fact
- News-derived fact
- Inference
- Hypothesis

Use dynamic_signal_score heavily.

Use entry_type.

Focus on:
- market expectations
- mispricing
- catalysts
- positioning
- risk/reward

Context:

{context}

FORMAT:

# Daily Investment Thesis Memo

## Executive Summary

## Investment Ideas

For each idea include:

- ticker
- dynamic signal score
- earnings timing
- entry type
- market reaction

Then include:

### Market data facts
### News-derived facts
### Inferences
### Hypotheses

Then:
- Why now
- Catalyst
- Entry zone
- Stop/invalidation
- Risk/reward
- Bear case
- Conviction score

## Portfolio Ranking
"""


def send_email(report):
    print("Preparing to send email...")

    sender = os.getenv("EMAIL_FROM")
    recipient = os.getenv("EMAIL_TO")
    password = os.getenv("EMAIL_PASSWORD")

    print(f"EMAIL_FROM set: {sender is not None}")
    print(f"EMAIL_TO set: {recipient is not None}")
    print(f"EMAIL_PASSWORD set: {password is not None}")

    if not sender or not recipient or not password:
        print("Missing email environment variables")
        return

    subject = f"Daily Investment Thesis Memo - {datetime.now().strftime('%Y-%m-%d')}"

    message = MIMEText(report, "plain")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient

    try:
        print("Connecting to Gmail SMTP...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            print("Logging into Gmail SMTP...")
            server.login(sender, password)

            print("Sending email...")
            server.sendmail(sender, recipient, message.as_string())

        print("Email sent successfully")

    except Exception as e:
        print(f"Email send failed: {type(e).__name__}: {e}")
        raise


def generate_report():
    print("Fetching news...")
    raw_news = get_news()

    print("Filtering news...")
    news = filter_high_signal_news(raw_news)

    print("Loading security master...")
    securities = load_security_master()

    print("Extracting companies...")
    companies = extract_companies_with_llm(news)

    resolved = []
    seen_tickers = set()

    for company in companies:
        result = resolve_company_with_security_master(
            company["company_name"],
            securities
        )

        if not result:
            continue

        ticker = result["ticker"]

        if ticker in seen_tickers:
            continue

        if validate_ticker(ticker):
            resolved.append({
                "ticker": ticker,
                "signal_score": company["signal_score"],
                "earnings_timing": company["earnings_timing"]
            })

            seen_tickers.add(ticker)

    print("Fetching market data...")
    market_data = []

    for item in resolved[:10]:
        data = get_market_data_for_ticker(
            ticker=item["ticker"],
            signal_score=item["signal_score"],
            earnings_timing=item["earnings_timing"]
        )

        if data:
            market_data.append(data)

    market_data = sorted(
        market_data,
        key=lambda x: x["dynamic_signal_score"],
        reverse=True
    )

    context = {
        "market_data": market_data
    }

    print("Generating memo...")
    prompt = build_prompt(context)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    report = response.output_text

    try:
        output_path = "/tmp/daily_memo.md" if os.getenv("WEBSITE_SITE_NAME") else "daily_memo.md"

        with open(output_path, "w") as file:
            file.write(report)

        print(f"Memo saved to {output_path}")

    except Exception as e:
        print(f"Could not save memo file: {type(e).__name__}: {e}")

    print("\n===== DAILY MEMO =====\n")
    print(report)

    send_email(report)


if __name__ == "__main__":
    generate_report()