#!/usr/bin/env python3
"""
extract_structured.py
---------------------
Extract structured data from unstructured documents (HTML, Markdown, plain text)
using the Anthropic Claude API.

Usage:
    python extract_structured.py <input_file> <schema_file> [--output <output_file>]

Arguments:
    input_file   Path to the document to extract from (.html, .htm, .md, .markdown, .txt, or any text file)
    schema_file  Path to a JSON file describing the desired output schema
    --output     (Optional) Path to write the extracted JSON; defaults to stdout

Examples:
    python extract_structured.py invoice.html schema.json
    python extract_structured.py report.md schema.json --output result.json
    python extract_structured.py notes.txt schema.json
"""

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit(
        "Error: 'anthropic' package not found.\n"
        "Install it with:  pip install anthropic"
    )

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit(
        "Error: 'beautifulsoup4' package not found.\n"
        "Install it with:  pip install beautifulsoup4"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

HTML_EXTENSIONS = {".html", ".htm"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}


# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------

def _attrs(tag, *names):
    """Return attribute value(s) from a BeautifulSoup tag, or empty string."""
    return " ".join(str(tag.get(n, "")).strip() for n in names if tag.get(n))


def _list_items(tag, ordered: bool, depth: int = 0) -> str:
    """Recursively convert <ul>/<ol> to Markdown."""
    lines = []
    for i, li in enumerate(tag.find_all("li", recursive=False), start=1):
        prefix = f"{i}. " if ordered else "- "
        pad = "  " * depth
        # Text of this li excluding nested lists
        nested = li.find(["ul", "ol"])
        if nested:
            text = li.get_text(separator=" ", strip=True)
            # strip the nested list text from the li text
            text = text.replace(nested.get_text(separator=" ", strip=True), "").strip()
        else:
            text = li.get_text(separator=" ", strip=True)
        lines.append(f"{pad}{prefix}{text}")
        if nested:
            lines.append(
                _list_items(nested, nested.name == "ol", depth + 1)
            )
    return "\n".join(lines)


def html_to_markdown(html: str) -> str:
    """
    Convert HTML to a clean Markdown string, preserving as much
    semantic content as possible (headings, lists, tables, links, images,
    code blocks, blockquotes).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove purely presentational / script / style tags
    for tag in soup(["script", "style", "noscript", "head"]):
        tag.decompose()

    lines: list[str] = []

    def walk(node):
        from bs4 import NavigableString, Tag

        # Plain text node
        if isinstance(node, NavigableString):
            text = str(node)
            if text.strip():
                lines.append(text)
            return

        # Skip non-Tag nodes (Comment, Doctype, ProcessingInstruction, etc.)
        if not isinstance(node, Tag):
            return

        name = node.name

        # ---- Headings ----
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            lines.append(f"\n{'#' * level} {node.get_text(strip=True)}\n")
            return

        # ---- Paragraph / div / section ----
        if name in ("p", "div", "section", "article", "main", "header", "footer", "aside"):
            lines.append("\n")
            for child in node.children:
                walk(child)
            lines.append("\n")
            return

        # ---- Line breaks ----
        if name == "br":
            lines.append("  \n")
            return

        # ---- Horizontal rule ----
        if name == "hr":
            lines.append("\n---\n")
            return

        # ---- Bold / strong ----
        if name in ("strong", "b"):
            lines.append(f"**{node.get_text(strip=True)}**")
            return

        # ---- Italic / em ----
        if name in ("em", "i"):
            lines.append(f"*{node.get_text(strip=True)}*")
            return

        # ---- Inline code ----
        if name == "code" and node.parent and node.parent.name != "pre":
            lines.append(f"`{node.get_text()}`")
            return

        # ---- Code block ----
        if name == "pre":
            code = node.get_text()
            lang = ""
            code_tag = node.find("code")
            if code_tag:
                cls = code_tag.get("class", [])
                for c in cls:
                    if c.startswith("language-"):
                        lang = c.replace("language-", "")
                        break
            lines.append(f"\n```{lang}\n{code}\n```\n")
            return

        # ---- Blockquote ----
        if name == "blockquote":
            inner = node.get_text(separator="\n", strip=True)
            quoted = "\n".join(f"> {l}" for l in inner.splitlines())
            lines.append(f"\n{quoted}\n")
            return

        # ---- Unordered list ----
        if name == "ul":
            lines.append("\n" + _list_items(node, ordered=False) + "\n")
            return

        # ---- Ordered list ----
        if name == "ol":
            lines.append("\n" + _list_items(node, ordered=True) + "\n")
            return

        # ---- Hyperlink ----
        if name == "a":
            text = node.get_text(strip=True)
            href = node.get("href", "").strip()
            if href and text:
                lines.append(f"[{text}]({href})")
            elif text:
                lines.append(text)
            return

        # ---- Image ----
        if name == "img":
            alt = node.get("alt", "").strip()
            src = node.get("src", "").strip()
            lines.append(f"![{alt}]({src})")
            return

        # ---- Table ----
        if name == "table":
            rows = node.find_all("tr")
            if not rows:
                return
            md_rows = []
            for r_i, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                cell_texts = [c.get_text(separator=" ", strip=True) for c in cells]
                md_rows.append("| " + " | ".join(cell_texts) + " |")
                if r_i == 0:
                    md_rows.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
            lines.append("\n" + "\n".join(md_rows) + "\n")
            return

        # ---- Definition list ----
        if name == "dl":
            for child in node.find_all(["dt", "dd"]):
                if child.name == "dt":
                    lines.append(f"\n**{child.get_text(strip=True)}**\n")
                else:
                    lines.append(f":   {child.get_text(strip=True)}\n")
            return

        # ---- Skip purely structural wrappers by descending ----
        # Guard: only descend into Tag nodes (not Comment, Doctype, etc.)
        children = getattr(node, "children", None)
        if children:
            for child in children:
                walk(child)

    walk(soup)

    # Collapse excessive blank lines
    md = "".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_document(path: Path) -> tuple[str, str]:
    """
    Load a document and return (content_as_markdown, detected_type).
    detected_type is one of 'html', 'markdown', 'text'.
    """
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")

    if suffix in HTML_EXTENSIONS:
        print(f"[info] Detected HTML document — converting to Markdown …", file=sys.stderr)
        return html_to_markdown(raw), "html"

    if suffix in MARKDOWN_EXTENSIONS:
        print(f"[info] Detected Markdown document.", file=sys.stderr)
        return raw, "markdown"

    # Heuristic: if file starts with an HTML tag treat as HTML regardless of extension
    stripped = raw.lstrip()
    if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html") or stripped.startswith("<HTML"):
        print(f"[info] Detected HTML content (by content sniffing) — converting to Markdown …", file=sys.stderr)
        return html_to_markdown(raw), "html"

    print(f"[info] Treating document as plain text.", file=sys.stderr)
    return raw, "text"


# ---------------------------------------------------------------------------
# Schema loading & prompt construction
# ---------------------------------------------------------------------------

def load_schema(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Could not parse schema JSON — {e}")


def build_prompt(content: str, schema: dict) -> str:
    schema_str = json.dumps(schema, indent=2)
    return textwrap.dedent(f"""\
        You are a precise data-extraction assistant.

        ## Task
        Extract structured data from the document below and return it as a
        single valid JSON object that conforms **exactly** to the JSON Schema
        provided.

        ## Rules
        - Return ONLY the JSON object — no explanation, no markdown fences,
          no preamble, no trailing text.
        - If a field cannot be found in the document, use `null` for optional
          fields or an empty string/array for required ones.
        - Preserve the original casing and formatting of extracted values
          unless the schema specifies otherwise.
        - Do not invent data that is not present in the document.

        ## JSON Schema
        ```json
        {schema_str}
        ```

        ## Document
        {content}
    """)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def extract_with_claude(content: str, schema: dict) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(content, schema)

    print(f"[info] Sending document to Claude ({MODEL}) …", file=sys.stderr)

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_response = message.content[0].text.strip()

    # Strip markdown code fences if the model adds them despite instructions
    raw_response = re.sub(r"^```(?:json)?\s*", "", raw_response)
    raw_response = re.sub(r"\s*```$", "", raw_response)

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(
            f"[warning] Claude returned non-JSON output. Raw response:\n{raw_response}",
            file=sys.stderr,
        )
        sys.exit(f"Error: Failed to parse Claude's response as JSON — {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured JSON data from an unstructured document using Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_file", type=Path, help="Document to extract data from")
    parser.add_argument("schema_file", type=Path, help="JSON Schema file describing the output")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Write extracted JSON to this file (default: stdout)"
    )
    parser.add_argument(
        "--pretty", action="store_true", default=True,
        help="Pretty-print output JSON (default: True)"
    )
    parser.add_argument(
        "--show-markdown", action="store_true",
        help="Print the converted Markdown to stderr (useful for debugging HTML conversion)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_file.exists():
        sys.exit(f"Error: Input file not found: {args.input_file}")
    if not args.schema_file.exists():
        sys.exit(f"Error: Schema file not found: {args.schema_file}")

    # Load and optionally convert document
    content, doc_type = load_document(args.input_file)

    if args.show_markdown:
        print("\n" + "=" * 60, file=sys.stderr)
        print("CONVERTED MARKDOWN CONTENT:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(content, file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)

    # Load schema
    schema = load_schema(args.schema_file)

    # Extract via Claude
    result = extract_with_claude(content, schema)

    # Output
    indent = 2 if args.pretty else None
    json_out = json.dumps(result, indent=indent, ensure_ascii=False)

    if args.output:
        args.output.write_text(json_out, encoding="utf-8")
        print(f"[info] Extracted data written to: {args.output}", file=sys.stderr)
    else:
        print(json_out)


if __name__ == "__main__":
    main()
