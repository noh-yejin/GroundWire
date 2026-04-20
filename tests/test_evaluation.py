from pathlib import Path

from app.models import (
    AnalysisResult,
    ImpactLabel,
    Issue,
    IssuePriority,
    IssueStatus,
    ReliabilityScore,
    RiskLevel,
    SentimentLabel,
)
from app.sample_data import load_sample_articles
from app.services.evaluation import evaluate_issues_against_goldens, load_golden_expectations


def _make_issue(topic: str, status: str, summary: str, mode: str, hold_reason: str | None, grounded_ratio: float) -> Issue:
    article = load_sample_articles()[0]
    return Issue(
        id=topic,
        topic=topic,
        keywords=["금리"],
        articles=[article],
        evidence=[],
        reliability=ReliabilityScore(0.8, 0.8, 0.8, 0.8, 0.8),
        analysis=AnalysisResult(
            summary=summary,
            keywords=["금리"],
            key_signals=["금리 인하"],
            key_points=["핵심 포인트"],
            trend_summary="안정",
            sentiment=SentimentLabel.NEUTRAL,
            market_impact=ImpactLabel.NEUTRAL,
            policy_risk=RiskLevel.LOW,
            volatility_risk=RiskLevel.LOW,
            risk_points=[],
            grounded=status == "READY",
            priority=IssuePriority.GENERAL,
            hold_reason=hold_reason,
            grounding_details={
                "grounding": {"grounded_ratio": grounded_ratio},
                "llm": {"analysis_mode": mode},
            },
        ),
        status=IssueStatus(status),
        updated_at=article.published_at,
    )


def test_evaluate_issues_against_goldens_reports_pass_rate() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "golden_issue_expectations.json"
    expectations = load_golden_expectations(fixture_path)
    issues = [
        _make_issue("cpi · 금리 · 인하", "HOLD", "미국 CPI 발표 이후 금리 인하 기대가 약화됐다.", "combined_remote", "grounding 검증 부족", 0.83),
        _make_issue("개선 · 메모리 · 반도체", "HOLD", "메모리 반도체와 AI 인프라 투자 확대가 언급됐다.", "combined_remote", "grounding 검증 부족", 0.66),
        _make_issue("HBM · sk하이닉스 · 삼성전자", "HOLD", "HBM 수요와 실적 기대가 언급됐다.", "lightweight_hold", "기사 수 부족: 1건", 0.0),
    ]

    report = evaluate_issues_against_goldens(issues, expectations)

    assert report["total"] == 3
    assert report["passed"] == 3
    assert report["pass_rate"] == 1.0
