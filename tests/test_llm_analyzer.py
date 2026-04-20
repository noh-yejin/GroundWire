from datetime import datetime

from app.models import AnalysisResult, Article, EvidenceSnippet, ImpactLabel, IssuePriority, ReliabilityScore, RiskLevel, SentimentLabel
from app.services.llm_analyzer import LLMAnalyzer, build_local_summary, derive_key_points


def test_build_local_summary_avoids_duplicate_evidence_and_titles() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5%",
            source="한국경제",
            published_at=now,
            url="https://example.com/1",
            content="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5%",
        ),
        Article(
            id="2",
            title="中, '사실상 기준금리' 11개월째 동결…LPR 1년 3.0%·5년 3.5%",
            source="연합인포맥스",
            published_at=now,
            url="https://example.com/2",
            content="중국이 사실상 기준금리 역할을 하는 LPR을 11개월째 동결했다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(
            article_id="1",
            source="한국경제",
            quote="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5% - 한국경제",
            url="https://example.com/1",
        ),
        EvidenceSnippet(
            article_id="2",
            source="연합인포맥스",
            quote="中, '사실상 기준금리' 11개월째 동결…LPR 1년 3.0%·5년 3.5%",
            url="https://example.com/2",
        ),
    ]
    reliability = ReliabilityScore(0.82, 0.9, 0.9, 0.8, 0.85)

    summary = build_local_summary("금리 · LPR", articles, evidence, reliability)

    assert "한국경제. 中" not in summary
    assert summary.count("11개월째 동결") <= 1


def test_derive_key_points_skips_evidence_that_matches_titles() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5%",
            source="한국경제",
            published_at=now,
            url="https://example.com/1",
            content="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5%",
        ),
        Article(
            id="2",
            title="中, '사실상 기준금리' 11개월째 동결…LPR 1년 3.0%·5년 3.5%",
            source="연합인포맥스",
            published_at=now,
            url="https://example.com/2",
            content="중국이 사실상 기준금리 역할을 하는 LPR을 11개월째 동결했다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(
            article_id="1",
            source="한국경제",
            quote="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5% - 한국경제",
            url="https://example.com/1",
        ),
        EvidenceSnippet(
            article_id="2",
            source="연합인포맥스",
            quote="중국이 사실상 기준금리 역할을 하는 LPR을 11개월째 동결했다.",
            url="https://example.com/2",
        ),
    ]

    points = derive_key_points(evidence, articles)

    assert all("한국경제" not in point for point in points)
    assert any("중국이 사실상 기준금리 역할을 하는 LPR을 11개월째 동결했다" in point for point in points)


def test_analyze_fallback_summary_uses_real_articles_not_empty_counts() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5%",
            source="한국경제",
            published_at=now,
            url="https://example.com/1",
            content="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5%",
        ),
        Article(
            id="2",
            title="中, '사실상 기준금리' 11개월째 동결…LPR 1년 3.0%·5년 3.5%",
            source="연합인포맥스",
            published_at=now,
            url="https://example.com/2",
            content="中, '사실상 기준금리' 11개월째 동결…LPR 1년 3.0%·5년 3.5%",
        ),
    ]
    evidence = [
        EvidenceSnippet(
            article_id="1",
            source="한국경제",
            quote="中 '기준금리 역할' LPR 11개월째 동결…1년물 3%·5년물 3.5%",
            url="https://example.com/1",
        )
    ]
    reliability = ReliabilityScore(0.5, 0.6, 0.8, 0.5, 0.5)

    analysis = LLMAnalyzer().analyze("금리 · LPR", articles, evidence, reliability, hold_reason="근거 부족")

    assert "0건 수집" not in analysis.summary


def test_analyze_prefers_remote_analysis_output(monkeypatch) -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="반도체 투자 확대 발표",
            source="Reuters",
            published_at=now,
            url="https://example.com/1",
            content="정부가 반도체 투자 확대를 발표했다.",
        ),
        Article(
            id="2",
            title="정부 반도체 투자 확대",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/2",
            content="정부가 생산 투자 확대 계획을 공개했다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(article_id="1", source="Reuters", quote="정부가 반도체 투자 확대를 발표했다.", url="https://example.com/1"),
        EvidenceSnippet(article_id="2", source="연합뉴스", quote="정부가 생산 투자 확대 계획을 공개했다.", url="https://example.com/2"),
    ]
    reliability = ReliabilityScore(0.82, 0.9, 0.9, 0.8, 0.85)
    analyzer = LLMAnalyzer()
    analyzer.client = object()

    monkeypatch.setattr(
        analyzer,
        "_analyze_bundle_with_openai",
        lambda *args, **kwargs: (
            AnalysisResult(
                summary="정부의 반도체 투자 확대 발표가 공식화됐고 생산 투자 확대 계획도 확인됐다.",
                keywords=["반도체", "투자"],
                key_signals=["투자 확대"],
                key_points=["반도체 투자 확대가 공식 발표됐다."],
                trend_summary="관련 보도가 이어지는 흐름입니다.",
                sentiment=SentimentLabel.POSITIVE,
                market_impact=ImpactLabel.POSITIVE,
                policy_risk=RiskLevel.MEDIUM,
                volatility_risk=RiskLevel.LOW,
                risk_points=["세부 집행 일정은 추가 확인이 필요합니다."],
                grounded=True,
                priority=IssuePriority.PRIORITY,
                hold_reason=None,
            ),
            ["정부가 반도체 투자 확대를 발표했다."],
        ),
    )

    analysis = analyzer.analyze("반도체 투자 확대", articles, evidence, reliability, hold_reason=None)

    assert "공식화" in analysis.summary
    assert analysis.key_signals[0] == "투자 확대"


def test_single_article_issue_skips_remote_analysis(monkeypatch) -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="연준 인사들, 물가 둔화 확인 필요 강조",
            source="CNBC",
            published_at=now,
            url="https://example.com/1",
            content="연준 관계자들은 금리 인하에 앞서 물가 둔화가 더 명확해져야 한다고 언급했다.",
        )
    ]
    evidence = [
        EvidenceSnippet(
            article_id="1",
            source="CNBC",
            quote="연준 관계자들은 금리 인하에 앞서 물가 둔화가 더 명확해져야 한다고 언급했다.",
            url="https://example.com/1",
        )
    ]
    reliability = ReliabilityScore(0.62, 0.6, 0.9, 0.6, 0.6)
    analyzer = LLMAnalyzer()
    analyzer.client = object()

    def fail(*_args, **_kwargs):
        raise AssertionError("remote path should not be used")

    monkeypatch.setattr(analyzer, "_analyze_with_openai", fail)
    monkeypatch.setattr(analyzer, "_extract_candidate_claims_with_openai", fail)

    analysis = analyzer.analyze("물가 · 둔화", articles, evidence, reliability, hold_reason="기사 수 부족: 1건")

    assert analysis.hold_reason == "기사 수 부족: 1건"


def test_remote_analysis_cache_reuses_previous_result() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="정부 반도체 투자 확대",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="정부가 반도체 투자 확대 계획을 공개했다.",
        ),
        Article(
            id="2",
            title="반도체 투자 확대 발표",
            source="Reuters",
            published_at=now,
            url="https://example.com/2",
            content="정부가 반도체 투자 확대를 발표했다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(article_id="1", source="연합뉴스", quote="정부가 반도체 투자 확대 계획을 공개했다.", url="https://example.com/1"),
        EvidenceSnippet(article_id="2", source="Reuters", quote="정부가 반도체 투자 확대를 발표했다.", url="https://example.com/2"),
    ]
    reliability = ReliabilityScore(0.82, 0.9, 0.9, 0.8, 0.85)
    analyzer = LLMAnalyzer()
    calls = {"count": 0}

    class FakeResponses:
        def parse(self, **_kwargs):
            calls["count"] += 1

            class FakeResponse:
                output_parsed = type(
                    "Parsed",
                    (),
                    {
                        "summary": "정부가 반도체 투자 확대를 발표했다.",
                        "keywords": ["반도체", "투자"],
                        "key_signals": ["투자 확대"],
                        "key_points": ["반도체 투자 확대 발표"],
                        "trend_summary": "관련 보도가 이어지고 있다.",
                        "sentiment": SentimentLabel.POSITIVE,
                        "market_impact": ImpactLabel.POSITIVE,
                        "policy_risk": RiskLevel.MEDIUM,
                        "volatility_risk": RiskLevel.LOW,
                        "risk_points": ["추가 세부안 확인 필요"],
                        "grounded": True,
                        "priority": IssuePriority.PRIORITY,
                        "hold_reason": None,
                    },
                )()

            return FakeResponse()

    analyzer.client = type("FakeClient", (), {"responses": FakeResponses()})()

    first = analyzer._analyze_with_openai("반도체 투자 확대", articles, evidence, reliability)
    second = analyzer._analyze_with_openai("반도체 투자 확대", articles, evidence, reliability)

    assert first is not None
    assert second is not None
    assert calls["count"] == 1


def test_extract_candidate_claims_merges_remote_and_heuristic(monkeypatch) -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="미국 CPI 발표 이후 금리 인하 기대 약화",
            source="Reuters",
            published_at=now,
            url="https://example.com/1",
            content="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐다.",
        ),
        Article(
            id="2",
            title="월가, CPI 이후 금리 인하 시점 재조정",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/2",
            content="월가는 기준금리 인하 시점이 늦춰질 수 있다고 봤다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(article_id="1", source="Reuters", quote="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐다.", url="https://example.com/1"),
        EvidenceSnippet(article_id="2", source="연합뉴스", quote="월가는 기준금리 인하 시점이 늦춰질 수 있다고 봤다.", url="https://example.com/2"),
    ]
    analyzer = LLMAnalyzer()
    analyzer.client = object()
    monkeypatch.setattr(
        analyzer,
        "_extract_candidate_claims_with_openai",
        lambda *_args, **_kwargs: ["미국 CPI 발표 이후 금리 인하 기대가 약화됐다."],
    )

    claims, remote_used = analyzer._extract_candidate_claims(
        "cpi · 금리 · 인하",
        articles,
        evidence,
        ["cpi", "금리", "인하"],
        allow_remote_extraction=True,
    )

    assert remote_used is True
    assert "미국 CPI 발표 이후 금리 인하 기대가 약화됐다" in claims[0]
    assert any("월가는 기준금리 인하 시점이 늦춰질 수 있다고 봤다" in claim for claim in claims)


def test_remote_bundle_cache_reuses_combined_response() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="정부 반도체 투자 확대",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="정부가 반도체 투자 확대 계획을 공개했다.",
        ),
        Article(
            id="2",
            title="반도체 투자 확대 발표",
            source="Reuters",
            published_at=now,
            url="https://example.com/2",
            content="정부가 반도체 투자 확대를 발표했다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(article_id="1", source="연합뉴스", quote="정부가 반도체 투자 확대 계획을 공개했다.", url="https://example.com/1"),
        EvidenceSnippet(article_id="2", source="Reuters", quote="정부가 반도체 투자 확대를 발표했다.", url="https://example.com/2"),
    ]
    reliability = ReliabilityScore(0.82, 0.9, 0.9, 0.8, 0.85)
    analyzer = LLMAnalyzer()
    calls = {"count": 0}

    class FakeResponses:
        def parse(self, **_kwargs):
            calls["count"] += 1

            class FakeResponse:
                output_parsed = type(
                    "Parsed",
                    (),
                    {
                        "summary": "정부가 반도체 투자 확대를 발표했다.",
                        "keywords": ["반도체", "투자"],
                        "key_signals": ["투자 확대"],
                        "key_points": ["반도체 투자 확대 발표"],
                        "trend_summary": "관련 보도가 이어지고 있다.",
                        "sentiment": SentimentLabel.POSITIVE,
                        "market_impact": ImpactLabel.POSITIVE,
                        "policy_risk": RiskLevel.MEDIUM,
                        "volatility_risk": RiskLevel.LOW,
                        "risk_points": ["추가 세부안 확인 필요"],
                        "grounded": True,
                        "priority": IssuePriority.PRIORITY,
                        "hold_reason": None,
                        "claims": ["정부가 반도체 투자 확대를 발표했다."],
                    },
                )()

            return FakeResponse()

    analyzer.client = type("FakeClient", (), {"responses": FakeResponses()})()

    first = analyzer._analyze_bundle_with_openai("반도체 투자 확대", articles, evidence, reliability)
    second = analyzer._analyze_bundle_with_openai("반도체 투자 확대", articles, evidence, reliability)

    assert first is not None
    assert second is not None
    assert first[1] == ["정부가 반도체 투자 확대를 발표했다."]
    assert calls["count"] == 1


def test_analyze_lightweight_marks_mode() -> None:
    now = datetime.utcnow()
    analyzer = LLMAnalyzer()
    articles = [
        Article(
            id="1",
            title="연준 인사들, 물가 둔화 확인 필요 강조",
            source="CNBC",
            published_at=now,
            url="https://example.com/1",
            content="연준 관계자들은 금리 인하에 앞서 물가 둔화가 더 명확해져야 한다고 언급했다.",
        )
    ]
    evidence = [
        EvidenceSnippet(
            article_id="1",
            source="CNBC",
            quote="연준 관계자들은 금리 인하에 앞서 물가 둔화가 더 명확해져야 한다고 언급했다.",
            url="https://example.com/1",
        )
    ]
    reliability = ReliabilityScore(0.54, 0.6, 0.9, 0.6, 0.6)

    analysis = analyzer.analyze_lightweight("물가 · 둔화", articles, evidence, reliability, "기사 수 부족: 1건")

    assert analysis.hold_reason == "기사 수 부족: 1건"
    assert analysis.grounding_details["llm"]["analysis_mode"] == "lightweight_hold"


def test_analyze_includes_structured_decision_details(monkeypatch) -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="미국 CPI 발표 이후 금리 인하 기대 약화",
            source="Reuters",
            published_at=now,
            url="https://example.com/1",
            content="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐다.",
        ),
        Article(
            id="2",
            title="월가, CPI 이후 금리 인하 시점 재조정",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/2",
            content="월가는 기준금리 인하 시점이 늦춰질 수 있다고 봤다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(article_id="1", source="Reuters", quote="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐다.", url="https://example.com/1"),
        EvidenceSnippet(article_id="2", source="연합뉴스", quote="월가는 기준금리 인하 시점이 늦춰질 수 있다고 봤다.", url="https://example.com/2"),
    ]
    reliability = ReliabilityScore(0.82, 0.9, 0.9, 0.8, 0.85)
    analyzer = LLMAnalyzer()
    analyzer.client = None

    analysis = analyzer.analyze("cpi · 금리 · 인하", articles, evidence, reliability, hold_reason=None)

    assert analysis.grounding_details["decision"]["status"] in {"READY", "HOLD"}
    assert analysis.grounding_details["decision"]["reasons"]


def test_review_hold_for_promotion_returns_ready_analysis(monkeypatch) -> None:
    now = datetime.utcnow()
    analyzer = LLMAnalyzer()
    articles = [
        Article(
            id="1",
            title="미국 CPI 발표 이후 금리 인하 기대 약화",
            source="Reuters",
            published_at=now,
            url="https://example.com/1",
            content="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐다.",
        )
    ]
    evidence = [
        EvidenceSnippet(article_id="1", source="Reuters", quote="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐다.", url="https://example.com/1")
    ]
    reliability = ReliabilityScore(0.82, 0.9, 0.9, 0.8, 0.85)
    analysis = AnalysisResult(
        summary="보류된 이슈입니다. 사유: grounding 검증 부족. 미국 CPI 발표 이후 금리 인하 기대가 약화됐다.",
        keywords=["금리"],
        key_signals=["금리 인하"],
        key_points=["포인트"],
        trend_summary="안정",
        sentiment=SentimentLabel.NEUTRAL,
        market_impact=ImpactLabel.NEUTRAL,
        policy_risk=RiskLevel.LOW,
        volatility_risk=RiskLevel.LOW,
        risk_points=[],
        grounded=False,
        priority=IssuePriority.GENERAL,
        hold_reason="grounding 검증 부족",
        grounding_details={
            "claims": [
                {"claim": "미국 CPI 발표 이후 기준금리 인하 기대가 줄었다.", "ready": True, "support_count": 3, "trusted_support_count": 3, "external_support_count": 1, "score": 0.82},
                {"claim": "월가는 연준의 금리 인하 시점이 늦춰질 수 있다고 봤다.", "ready": True, "support_count": 3, "trusted_support_count": 3, "external_support_count": 1, "score": 0.81},
                {"claim": "증시 변동성이 확대됐다.", "ready": True, "support_count": 3, "trusted_support_count": 2, "external_support_count": 1, "score": 0.79},
            ],
            "grounding": {"grounded_ratio": 0.667, "issue_score": 0.739},
            "decision": {"ready_claim_count": 3, "total_claim_count": 4, "grounded_claim_count": 3},
            "llm": {"analysis_mode": "combined_remote"},
        },
    )

    class FakeResponses:
        def parse(self, **_kwargs):
            class FakeResponse:
                output_parsed = type(
                    "Parsed",
                    (),
                    {
                        "promote_to_ready": True,
                        "rationale": "핵심 claim 3개가 모두 다중 근거로 뒷받침돼 승격 가능하다.",
                        "revised_summary": "미국 CPI 이후 금리 인하 기대 약화와 증시 변동성 확대가 확인됐다.",
                    },
                )()

            return FakeResponse()

    analyzer.client = type("FakeClient", (), {"responses": FakeResponses()})()

    reviewed = analyzer.review_hold_for_promotion("cpi · 금리 · 인하", articles, evidence, reliability, analysis)

    assert reviewed is not None
    assert reviewed.hold_reason is None
    assert reviewed.grounding_details["llm"]["second_pass_promoted"] is True
