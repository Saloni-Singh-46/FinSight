"""Unit & Integration tests for Multi-Agent Graph Orchestration."""

from finsight.orchestration.graph import FinSightOrchestrator
from finsight.tools.financial_tools import SandboxedPythonExecutor


def test_sandboxed_python_executor():
    res = SandboxedPythonExecutor.execute_expression("round(10.82 * 1.25, 2)")
    assert res["success"] is True
    assert abs(res["result"] - 13.525) < 0.02

    # Test disallowed operations
    bad_res = SandboxedPythonExecutor.execute_expression("__import__('os').system('dir')")
    assert bad_res["success"] is False


def test_agent_graph_full_run():
    orchestrator = FinSightOrchestrator()
    final_state = orchestrator.execute("Analyze Tata Motors for the next 12 months")

    assert final_state.status == "COMPLETED"
    assert final_state.company == "Tata Motors"
    assert final_state.ticker == "TATAMOTORS.NS"
    assert len(final_state.trace_events) >= 6
    assert len(final_state.final_memo) > 500
    assert "Tata Motors" in final_state.final_memo
    assert "BULLISH" in final_state.final_memo
    assert final_state.critic_output.get("status") in ["PASS", "REVISE"]
    assert len(final_state.citations) > 0


def test_out_of_scope_rejection():
    orchestrator = FinSightOrchestrator()
    state = orchestrator.execute("Tell me about crypto trading bots")
    assert state.in_scope is False or "Out of scope" in state.final_memo
