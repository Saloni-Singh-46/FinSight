"""Institutional Research Memo Exporter (Markdown, HTML, PDF)."""

from pathlib import Path

from finsight.utils.config import logger


class MemoExporter:
    """Exports structured research memos to Markdown, HTML, and institutional PDF."""

    def __init__(self, output_dir: str = "runs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_markdown(self, memo_content: str, job_id: str) -> str:
        """Saves memo to markdown file."""
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        file_path = job_dir / "research_memo.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(memo_content)
        return str(file_path)

    def export_html(self, memo_content: str, job_id: str, company: str = "Tata Motors") -> str:
        """Converts Markdown memo into an executive styled HTML report."""
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        file_path = job_dir / "research_memo.html"

        import markdown

        html_body = markdown.markdown(memo_content, extensions=["tables", "fenced_code"])

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinSight Institutional Memo - {company}</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #111827;
            --border: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --accent: #2563eb;
            --green: #10b981;
            --red: #ef4444;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: var(--surface);
            padding: 40px;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }}
        h1, h2, h3 {{ color: #ffffff; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
        h1 {{ color: var(--accent); font-size: 26px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 14px;
            border: 1px solid var(--border);
            text-align: left;
        }}
        th {{ background-color: #1f2937; color: #93c5fd; }}
        code {{
            background: #1f2937;
            color: #38bdf8;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }}
        .disclaimer {{
            margin-top: 40px;
            padding: 15px;
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid var(--red);
            font-size: 12px;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>"""

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return str(file_path)

    def _format_markdown_for_reportlab(self, text: str) -> str:
        """Escapes XML and converts basic Markdown tags to ReportLab supported tags."""
        import html
        import re

        # Escape XML entities first
        safe_text = html.escape(text)
        # Bold: **text** -> <b>text</b>
        safe_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_text)
        # Italic: *text* -> <i>text</i>
        safe_text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", safe_text)
        # Inline code: `code` -> <font face="Courier">code</font>
        safe_text = re.sub(r"`(.+?)`", r'<font face="Courier" color="#1e40af">\1</font>', safe_text)
        return safe_text

    def export_pdf(self, memo_content: str, job_id: str, company: str = "Tata Motors") -> str:
        """Generates executive PDF document using ReportLab."""
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        file_path = job_dir / "research_memo.pdf"

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

            doc = SimpleDocTemplate(
                str(file_path),
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36,
            )
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "DocTitle",
                parent=styles["Heading1"],
                fontSize=16,
                leading=20,
                textColor=colors.HexColor("#1e40af"),
                spaceAfter=6,
            )
            h2_style = ParagraphStyle(
                "DocH2",
                parent=styles["Heading2"],
                fontSize=12,
                leading=16,
                textColor=colors.HexColor("#1e293b"),
                spaceBefore=10,
                spaceAfter=4,
            )
            h3_style = ParagraphStyle(
                "DocH3",
                parent=styles["Heading3"],
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#334155"),
                spaceBefore=6,
                spaceAfter=2,
            )
            body_style = ParagraphStyle(
                "DocBody",
                parent=styles["Normal"],
                fontSize=8.5,
                leading=12,
                textColor=colors.HexColor("#334155"),
                spaceAfter=4,
            )
            table_hdr_style = ParagraphStyle(
                "TableHdr",
                parent=styles["Normal"],
                fontSize=8,
                leading=10,
                textColor=colors.white,
                fontName="Helvetica-Bold",
            )
            table_cell_style = ParagraphStyle(
                "TableCell",
                parent=styles["Normal"],
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#1e293b"),
            )
            disclaimer_style = ParagraphStyle(
                "DocDisc",
                parent=styles["Italic"],
                fontSize=7.5,
                leading=10,
                textColor=colors.HexColor("#64748b"),
            )

            elements = []
            elements.append(
                Paragraph(f"FINSIGHT INSTITUTIONAL EQUITY RESEARCH: {company.upper()}", title_style)
            )
            elements.append(
                Paragraph(
                    "<b>Confidential Investment Research Report</b> | Generated by FinSight Multi-Agent AI System",
                    disclaimer_style,
                )
            )
            elements.append(Spacer(1, 10))

            lines = memo_content.split("\n")
            table_buffer = []

            def flush_table():
                if not table_buffer:
                    return
                # Filter separator lines like |---|---|
                valid_rows = [
                    row
                    for row in table_buffer
                    if not all(c.strip().startswith("-") or not c.strip() for c in row)
                ]
                if valid_rows:
                    num_cols = max(len(r) for r in valid_rows)
                    formatted_rows = []
                    for row_idx, row in enumerate(valid_rows):
                        # Pad row if necessary
                        row = row + [""] * (num_cols - len(row))
                        style = table_hdr_style if row_idx == 0 else table_cell_style
                        formatted_rows.append(
                            [Paragraph(self._format_markdown_for_reportlab(c), style) for c in row]
                        )

                    col_w = max(40, int(540 / max(num_cols, 1)))
                    t = Table(formatted_rows, colWidths=[col_w] * num_cols)
                    t.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                                (
                                    "ROWBACKGROUNDS",
                                    (0, 1),
                                    (-1, -1),
                                    [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")],
                                ),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            ]
                        )
                    )
                    elements.append(t)
                    elements.append(Spacer(1, 6))
                table_buffer.clear()

            for line in lines:
                l = line.strip()
                if not l:
                    flush_table()
                    continue

                if l.startswith("|") and l.endswith("|"):
                    cells = [c.strip() for c in l.split("|")[1:-1]]
                    if cells:
                        table_buffer.append(cells)
                    continue
                else:
                    flush_table()

                if l.startswith("# "):
                    elements.append(
                        Paragraph(self._format_markdown_for_reportlab(l[2:]), title_style)
                    )
                elif l.startswith("## "):
                    elements.append(Paragraph(self._format_markdown_for_reportlab(l[3:]), h2_style))
                elif l.startswith("### "):
                    elements.append(Paragraph(self._format_markdown_for_reportlab(l[4:]), h3_style))
                else:
                    elements.append(Paragraph(self._format_markdown_for_reportlab(l), body_style))

            flush_table()
            elements.append(Spacer(1, 15))
            elements.append(
                Paragraph(
                    "<b>DISCLAIMER:</b> For educational and research purposes only. Not financial advice.",
                    disclaimer_style,
                )
            )
            doc.build(elements)
            logger.info(f"Generated PDF memo report at {file_path}")
            return str(file_path)
        except Exception as e:
            logger.warning(f"PDF generation note ({e}), returning text path.")
            return self.export_markdown(memo_content, job_id)
