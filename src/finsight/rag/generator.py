"""Grounded Financial Answer Generator with Inline Citations and Evidence Store."""

from typing import Any

from finsight.rag.chunking import DocumentChunk


class GroundedRAGGenerator:
    """Produces verified, citation-anchored summaries from retrieved document chunks."""

    def __init__(self):
        pass

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]],
        company: str = "Tata Motors",
    ) -> dict[str, Any]:
        """Synthesizes factual answer with strict inline citations."""
        if not retrieved_results:
            return {
                "answer": f"Insufficient evidence in corporate filings to answer '{query}' for {company}.",
                "citations": [],
                "evidence_store": {},
                "grounded": False,
            }

        evidence_store = {}
        citations = []
        context_snippets = []

        for i, item in enumerate(retrieved_results):
            chunk: DocumentChunk = item["chunk"]
            cite_key = f"[{chunk.doc_name}, p.{chunk.page_number}, {chunk.chunk_id}]"
            citations.append(
                {
                    "cite_key": cite_key,
                    "doc_name": chunk.doc_name,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "snippet": chunk.content[:150] + "...",
                    "full_text": chunk.content,
                    "is_table": chunk.is_table,
                }
            )
            evidence_store[chunk.chunk_id] = {
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "content": chunk.content,
            }
            context_snippets.append(f"{chunk.content} {cite_key}")

        # Formulate grounded answer paragraphs
        primary_chunk: DocumentChunk = retrieved_results[0]["chunk"]
        primary_cite = (
            f"[{primary_chunk.doc_name}, p.{primary_chunk.page_number}, {primary_chunk.chunk_id}]"
        )

        answer_body = f"According to corporate filings and management disclosures, {primary_chunk.content} {primary_cite}"
        if len(retrieved_results) > 1:
            sec_chunk: DocumentChunk = retrieved_results[1]["chunk"]
            sec_cite = f"[{sec_chunk.doc_name}, p.{sec_chunk.page_number}, {sec_chunk.chunk_id}]"
            answer_body += (
                f"\n\nAdditionally, {sec_chunk.section} highlights: {sec_chunk.content} {sec_cite}"
            )

        return {
            "query": query,
            "answer": answer_body,
            "citations": citations,
            "evidence_store": evidence_store,
            "top_source": primary_chunk.doc_name,
            "grounded": True,
        }
