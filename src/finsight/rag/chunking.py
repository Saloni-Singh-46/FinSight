"""Structure-Aware Financial Document Ingestion and Chunking."""

import hashlib
from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_name: str
    company: str
    fiscal_year: str
    page_number: int
    section: str
    content: str
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self):
        return hash(self.chunk_id)

    def __eq__(self, other):
        if isinstance(other, DocumentChunk):
            return self.chunk_id == other.chunk_id
        return False


def structure_aware_chunk(
    text: str,
    doc_name: str,
    company: str = "Tata Motors",
    fiscal_year: str = "FY24",
    page_number: int = 1,
    section: str = "Business Highlights",
    chunk_size_words: int = 200,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    """Splits text and tables into metadata-enriched chunks with deterministic IDs."""
    chunks = []

    # Check if section contains markdown tables
    paragraphs = text.split("\n\n")
    current_chunk_words = []
    current_is_table = False

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        words = para_clean.split()
        is_table = "|" in para_clean and "-|-" in para_clean or para_clean.count("|") > 4

        # If a single table, preserve intact
        if is_table:
            if current_chunk_words:
                content_str = " ".join(current_chunk_words)
                cid = hashlib.md5(
                    f"{doc_name}_{page_number}_{content_str[:40]}".encode()
                ).hexdigest()[:8]
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"chk_{cid}",
                        doc_name=doc_name,
                        company=company,
                        fiscal_year=fiscal_year,
                        page_number=page_number,
                        section=section,
                        content=content_str,
                        is_table=current_is_table,
                    )
                )
                current_chunk_words = []

            cid = hashlib.md5(f"{doc_name}_{page_number}_{para_clean[:40]}".encode()).hexdigest()[
                :8
            ]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk_{cid}",
                    doc_name=doc_name,
                    company=company,
                    fiscal_year=fiscal_year,
                    page_number=page_number,
                    section=section,
                    content=para_clean,
                    is_table=True,
                )
            )
            continue

        current_chunk_words.extend(words)

        if len(current_chunk_words) >= chunk_size_words:
            content_str = " ".join(current_chunk_words)
            cid = hashlib.md5(f"{doc_name}_{page_number}_{content_str[:40]}".encode()).hexdigest()[
                :8
            ]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chk_{cid}",
                    doc_name=doc_name,
                    company=company,
                    fiscal_year=fiscal_year,
                    page_number=page_number,
                    section=section,
                    content=content_str,
                    is_table=False,
                )
            )
            # Apply sliding overlap
            current_chunk_words = current_chunk_words[-overlap_words:]

    if current_chunk_words:
        content_str = " ".join(current_chunk_words)
        cid = hashlib.md5(f"{doc_name}_{page_number}_{content_str[:40]}".encode()).hexdigest()[:8]
        chunks.append(
            DocumentChunk(
                chunk_id=f"chk_{cid}",
                doc_name=doc_name,
                company=company,
                fiscal_year=fiscal_year,
                page_number=page_number,
                section=section,
                content=content_str,
                is_table=False,
            )
        )

    return chunks
