"""Unit tests for Filings RAG, Hybrid Retrieval, Grounding, and Ablation."""

import pytest

from finsight.rag.chunking import structure_aware_chunk
from finsight.rag.evaluator import RAGEvaluator
from finsight.rag.generator import GroundedRAGGenerator
from finsight.rag.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_filings_chunks():
    doc1 = """
    Tata Motors Group delivered strong operational performance in FY24.
    Consolidated revenue reached record INR 4,37,928 Crore, growing 26.6% YoY.
    Free cash flow generated was INR 26,900 Crore, enabling massive debt reduction.
    
    | Segment | Revenue (Cr) | EBITDA Margin % |
    | JLR | 2,89,900 | 14.2% |
    | CV | 78,800 | 11.2% |
    | PV | 52,400 | 6.8% |
    
    JLR order book remains robust with strong demand for Range Rover and Defender models.
    EBIT margin guidance for JLR is maintained at 8.5% for the medium term.
    """

    doc2 = """
    In the Electric Vehicle (EV) segment, Tata Motors maintained market leadership with ~70% market share.
    Capital expenditure (capex) guidance for FY25 is planned at INR 8,000 Crore.
    The group aims to achieve net zero automotive debt by FY25.
    Key risk factors include semiconductor component supply and raw material steel inflation.
    CRISIL upgraded credit rating to AA+ with Stable outlook.
    """

    c1 = structure_aware_chunk(
        doc1,
        doc_name="TataMotors_AR_FY24.pdf",
        company="Tata Motors",
        page_number=12,
        section="Financial Review",
    )
    c2 = structure_aware_chunk(
        doc2,
        doc_name="TataMotors_AR_FY24.pdf",
        company="Tata Motors",
        page_number=28,
        section="Outlook & Risks",
    )
    return c1 + c2


def test_structure_aware_chunking():
    text = "Tata Motors reported strong earnings.\n\n| Col1 | Col2 |\n|---|---|\n| A | B |\n\nManagement outlook is positive."
    chunks = structure_aware_chunk(text, doc_name="Report.pdf", page_number=5)
    assert len(chunks) >= 2
    assert any(c.is_table for c in chunks)
    assert chunks[0].doc_name == "Report.pdf"
    assert chunks[0].chunk_id.startswith("chk_")


def test_hybrid_retrieval(sample_filings_chunks):
    retriever = HybridRetriever(top_k=3)
    retriever.index_chunks(sample_filings_chunks)

    results = retriever.retrieve_hybrid("What is JLR EBIT margin guidance?", k=2)
    assert len(results) > 0
    assert "rerank_score" in results[0]
    assert "8.5%" in results[0]["chunk"].content or "JLR" in results[0]["chunk"].content


def test_grounded_answer_generator(sample_filings_chunks):
    retriever = HybridRetriever(top_k=2)
    retriever.index_chunks(sample_filings_chunks)
    retrieved = retriever.retrieve_hybrid("What is the EV market share and debt target?")

    generator = GroundedRAGGenerator()
    res = generator.generate_grounded_answer(
        "EV market share and debt", retrieved, company="Tata Motors"
    )
    assert res["grounded"] is True
    assert len(res["citations"]) > 0
    assert "[TataMotors_AR_FY24.pdf, p." in res["answer"]
    assert "evidence_store" in res


def test_rag_ablation(sample_filings_chunks):
    evaluator = RAGEvaluator()
    ablation = evaluator.run_retrieval_ablation(sample_filings_chunks, k=3)
    assert "ablation_table" in ablation
    table = ablation["ablation_table"]
    assert "Dense-Only" in table
    assert "BM25-Only" in table
    assert "Hybrid + Cross-Encoder Rerank" in table
