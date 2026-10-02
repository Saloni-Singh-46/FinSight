"""Generate institutional offline demo run assets in demo/ directory."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finsight.memo.exporter import MemoExporter
from finsight.orchestration.graph import FinSightOrchestrator
from finsight.tools.financial_tools import FinSightToolRegistry
from finsight.utils.config import logger


def generate_demo_assets():
    demo_dir = Path("demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing FinSight Tool Registry & Orchestrator...")
    tools = FinSightToolRegistry()
    orchestrator = FinSightOrchestrator(tools=tools)

    query = "Analyze Tata Motors for the next 12 months"
    logger.info(f"Executing full demo research workflow for: '{query}'...")
    state = orchestrator.execute(query=query)

    # Save to demo JSON
    demo_file = demo_dir / "tata_motors_demo_run.json"
    with open(demo_file, "w", encoding="utf-8") as f:
        f.write(state.model_dump_json(indent=2))
    logger.info(f"Saved demo state to {demo_file}")

    # Export markdown, HTML, and PDF
    exporter = MemoExporter(output_dir="demo")
    md_path = exporter.export_markdown(state.final_memo, "tata_motors_demo")
    html_path = exporter.export_html(state.final_memo, "tata_motors_demo", company=state.company)
    pdf_path = exporter.export_pdf(state.final_memo, "tata_motors_demo", company=state.company)

    logger.info("Exported Demo Memo:")
    logger.info(f"  Markdown: {md_path}")
    logger.info(f"  HTML:     {html_path}")
    logger.info(f"  PDF:      {pdf_path}")

    print("\n=======================================================")
    print("DEMO ASSETS GENERATION COMPLETE")
    print(f"Company: {state.company} ({state.ticker})")
    print(f"Job ID:  {state.job_id}")
    print(f"Citations: {len(state.citations)}")
    print(f"Trace events: {len(state.trace_events)}")
    print("=======================================================\n")


if __name__ == "__main__":
    generate_demo_assets()
