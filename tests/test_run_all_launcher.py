from pathlib import Path


def test_run_all_uses_non_interactive_streamlit_flags():
    script_text = Path("run_all.py").read_text(encoding="utf-8")

    assert "--server.headless=true" in script_text
    assert "--server.headless=false" not in script_text
    assert "--browser.gatherUsageStats=false" in script_text
