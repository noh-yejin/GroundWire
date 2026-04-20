from datetime import datetime

from app.models import Article
from app.services.preprocessing import clean_article_content, preprocess_articles


def test_preprocess_articles_removes_duplicates_and_low_quality() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="AI 반도체 수요 증가",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="<p>AI 반도체 수요 증가가 수출 회복으로 이어지고 있다.</p>",
            collected_at=now,
        ),
        Article(
            id="2",
            title="AI 반도체 수요 증가",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="중복 기사",
            collected_at=now,
        ),
        Article(
            id="3",
            title="짧음",
            source="블로그",
            published_at=now,
            url="https://example.com/3",
            content="광고",
            collected_at=now,
        ),
    ]

    processed = preprocess_articles(articles)

    assert len(processed) == 1
    assert processed[0].content.startswith("AI 반도체 수요 증가")
    assert processed[0].content_quality >= 0.45


def test_clean_article_content_removes_repeated_title_and_source() -> None:
    title = "[속보] 中 '기준금리 역할' LPR 11개월째 동결…1년물 3.0%·5년물 3.5%"
    content = "[속보] 中 '기준금리 역할' LPR 11개월째 동결…1년물 3.0%·5년물 3.5% &nbsp;&nbsp; 연합뉴스"

    cleaned = clean_article_content(title, content, "연합뉴스")

    assert cleaned == title


def test_clean_article_content_keeps_non_duplicate_followup_text() -> None:
    title = "China keeps benchmark lending rates unchanged as economic growth revs up"
    content = (
        "China keeps benchmark lending rates unchanged as economic growth revs up, Mideast risks loom &nbsp;&nbsp; CNBC "
        "China leaves lending benchmarks unchanged for 11th month in April &nbsp;&nbsp; Reuters "
        "View Full Coverage on Google News"
    )

    cleaned = clean_article_content(title, content, "CNBC")

    assert "View Full Coverage" not in cleaned
    assert "CNBC" not in cleaned


def test_preprocess_articles_filters_live_update_aggregators() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="Stock futures fall on renewed U.S.-Iran worries, but losses kept in check: Live updates - CNBC",
            source="CNBC",
            published_at=now,
            url="https://example.com/1",
            content="Stock futures fall on renewed U.S.-Iran worries, but losses kept in check: Live updates &nbsp;&nbsp; CNBC",
            collected_at=now,
        ),
        Article(
            id="2",
            title="중국 LPR 11개월째 동결",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/2",
            content="중국이 사실상 기준금리 역할을 하는 LPR을 11개월째 동결했다.",
            collected_at=now,
        ),
    ]

    processed = preprocess_articles(articles)

    assert len(processed) == 1
    assert processed[0].title == "중국 LPR 11개월째 동결"
