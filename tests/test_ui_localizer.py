from app.services.ui_localizer import UIDisplayLocalizer


def test_ui_localizer_fallback_translates_common_finance_phrases() -> None:
    localizer = UIDisplayLocalizer()
    localizer.client = None

    label = localizer.localize_label("Stock futures fall as U.S.-Iran tensions flare up again")
    point = localizer.localize_point("Dow futures fall over 350 points as Iranian war tensions escalate: Live updates", max_chars=40)

    assert "주가지수 선물" in label or "다우 선물" in point
    assert "실시간 업데이트" not in point
