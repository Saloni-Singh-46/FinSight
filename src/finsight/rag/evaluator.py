"""RAG Evaluation Suite and 4-Way Retrieval Ablation Study."""

from typing import Any

import numpy as np

from finsight.rag.chunking import DocumentChunk
from finsight.rag.hybrid_retriever import HybridRetriever


class RAGEvaluator:
    """Evaluates Hit Rate@k, MRR, Faithfulness, and runs ablation benchmarks."""

    def __init__(self, retriever: HybridRetriever | None = None):
        self.retriever = retriever or HybridRetriever()

    def build_gold_qa_set(self) -> list[dict[str, Any]]:
        """Constructs 30 curated financial Q&A benchmark items grounded in filings."""
        qa_set = [
            {
                "question": "What drove Tata Motors free cash flow turnaround in FY24?",
                "target_keywords": ["jlr", "cash flow", "fcf", "ebitda", "turnaround"],
                "expected_section": "Financial Performance",
            },
            {
                "question": "What is JLR's EBIT margin guidance for the medium term?",
                "target_keywords": ["ebit", "margin", "jlr", "guidance", "8.5%"],
                "expected_section": "JLR Outlook",
            },
            {
                "question": "How is Tata Motors managing its net automotive debt reduction?",
                "target_keywords": [
                    "debt",
                    "reduction",
                    "deleverage",
                    "automotive",
                    "zero net debt",
                ],
                "expected_section": "Capital Structure",
            },
            {
                "question": "What is the market share of Tata Motors in the Indian electric passenger vehicle (EV) segment?",
                "target_keywords": ["ev", "electric", "market share", "passenger vehicle", "70%"],
                "expected_section": "EV Strategy",
            },
            {
                "question": "What are the key supply chain risks identified in commercial vehicle manufacturing?",
                "target_keywords": [
                    "supply chain",
                    "semiconductor",
                    "raw material",
                    "freight",
                    "risk",
                ],
                "expected_section": "Risk Factors",
            },
            {
                "question": "What are the revenue growth drivers for Jaguar Land Rover's Defender and Range Rover models?",
                "target_keywords": ["defender", "range rover", "order book", "luxury", "volume"],
                "expected_section": "Product Mix",
            },
            {
                "question": "What is the annual planned capital expenditure (capex) for FY25 across PV and EV segments?",
                "target_keywords": ["capex", "capital expenditure", "investment", "fy25", "crore"],
                "expected_section": "Capex Guidance",
            },
            {
                "question": "How did raw material commodity prices impact commercial vehicle operating margins?",
                "target_keywords": ["steel", "commodity", "raw material", "margin", "inflation"],
                "expected_section": "Cost Structure",
            },
            {
                "question": "What is the status of the proposed demerger into Commercial Vehicles and Passenger Vehicles?",
                "target_keywords": [
                    "demerger",
                    "split",
                    "commercial vehicles",
                    "passenger vehicles",
                    "nclt",
                ],
                "expected_section": "Corporate Restructuring",
            },
            {
                "question": "What is Tata Motors credit rating outlook from CRISIL and ICRA?",
                "target_keywords": ["rating", "crisil", "icra", "aa+", "stable"],
                "expected_section": "Credit Profile",
            },
        ]

        # Expand to 30 items with domain variations
        variations = []
        for i, base in enumerate(qa_set):
            variations.append(
                {
                    "question": f"Management commentary on {base['target_keywords'][0]} and {base['target_keywords'][1]}",
                    "target_keywords": base["target_keywords"],
                    "expected_section": base["expected_section"],
                }
            )
            variations.append(
                {
                    "question": f"Detailed breakdown of {base['target_keywords'][0]} performance and risks in FY24",
                    "target_keywords": base["target_keywords"],
                    "expected_section": base["expected_section"],
                }
            )

        return (qa_set + variations)[:30]

    def run_retrieval_ablation(
        self,
        chunks: list[DocumentChunk],
        gold_qa: list[dict[str, Any]] | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        """Runs rigorous 4-way ablation: Dense-only vs BM25-only vs Hybrid vs Hybrid+Rerank."""
        qa = gold_qa or self.build_gold_qa_set()
        self.retriever.index_chunks(chunks)

        configs = {
            "Dense-Only": {"use_dense": True, "use_bm25": False, "rerank": False},
            "BM25-Only": {"use_dense": False, "use_bm25": True, "rerank": False},
            "Hybrid (Dense + BM25)": {"use_dense": True, "use_bm25": True, "rerank": False},
            "Hybrid + Cross-Encoder Rerank": {"use_dense": True, "use_bm25": True, "rerank": True},
        }

        ablation_results = {}

        for name, cfg in configs.items():
            hit_count = 0
            reciprocal_ranks = []
            precisions = []

            for item in qa:
                q = item["question"]
                keywords = item["target_keywords"]

                if cfg["use_dense"] and not cfg["use_bm25"]:
                    res = self.retriever.retrieve_dense(q, k=k)
                    retrieved_chunks = [r[0] for r in res]
                elif cfg["use_bm25"] and not cfg["use_dense"]:
                    res = self.retriever.retrieve_bm25(q, k=k)
                    retrieved_chunks = [r[0] for r in res]
                elif cfg["rerank"]:
                    res = self.retriever.retrieve_hybrid(q, k=k)
                    retrieved_chunks = [r["chunk"] for r in res]
                else:
                    # Hybrid without reranking
                    res = self.retriever.retrieve_hybrid(q, k=k)
                    retrieved_chunks = [
                        r["chunk"]
                        for r in sorted(res, key=lambda x: x["hybrid_score"], reverse=True)
                    ]

                # Check hits
                found_rank = 0
                hits_in_k = 0
                for rank, chk in enumerate(retrieved_chunks, start=1):
                    content = chk.content.lower()
                    # Check if any target keywords match
                    if any(kw.lower() in content for kw in keywords):
                        hits_in_k += 1
                        if found_rank == 0:
                            found_rank = rank

                if found_rank > 0:
                    hit_count += 1
                    reciprocal_ranks.append(1.0 / found_rank)
                else:
                    reciprocal_ranks.append(0.0)

                precisions.append(hits_in_k / max(1, k))

            hit_rate = hit_count / len(qa)
            mrr = np.mean(reciprocal_ranks)
            mean_precision = np.mean(precisions)

            ablation_results[name] = {
                "hit_rate_at_5": round(float(hit_rate * 100), 2),
                "mrr": round(float(mrr), 4),
                "precision_at_5": round(float(mean_precision * 100), 2),
                "samples": len(qa),
            }

        return {
            "ablation_table": ablation_results,
            "total_documents_indexed": len(chunks),
            "benchmark_questions": len(qa),
        }
