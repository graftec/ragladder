#!/usr/bin/env python3
"""Convert the Swiss Obligationenrecht (SR 220) Akoma Ntoso XML into a BEIR
`corpus.jsonl`.

This is a **demo-side** converter: it is domain-specific (it knows Fedlex's
Akoma Ntoso layout) and therefore lives with the data, NOT inside the
`ragladder` package. The package only ever consumes the BEIR JSONL contract
(see the README); converting a corpus *into* that contract is an
upstream step like this one.

Each retrievable unit is one article (`<article eId="art_N">`). Its embedded
`text` is the Randtitel (marginal note, if any) followed by the article body;
footnotes (`<authorialNote>`) are stripped. The hierarchy breadcrumb and article
number are kept in `metadata` (not embedded).

Usage:
    python data/build_or_dataset.py \
        --xml data/SR-220-01012026-DE.xml \
        --out data/or-corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

AKN = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
FEDLEX = "http://fedlex.admin.ch/"
A = f"{{{AKN}}}"  # akn namespace prefix for tags
ROLE = f"{{{FEDLEX}}}role"

STRUCTURAL_TAGS = {f"{A}part", f"{A}title", f"{A}chapter", f"{A}section", f"{A}level"}


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean(text: str) -> str:
    """Collapse whitespace; the XML is riddled with newlines and indentation."""
    return re.sub(r"\s+", " ", text).strip()


def _text_without_footnotes(elem: ET.Element) -> str:
    """All descendant text of `elem`, skipping <authorialNote> subtrees."""
    parts: list[str] = []

    def walk(node: ET.Element):
        if node.tag == f"{A}authorialNote":
            return  # footnote — not part of the normative text
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(elem)
    return _clean(" ".join(parts))


def _direct_child(elem: ET.Element, localname: str) -> ET.Element | None:
    for child in elem:
        if _localname(child.tag) == localname:
            return child
    return None


def _heading_text(elem: ET.Element) -> str:
    """`<num> <heading>` of a container, e.g. '2. Betreffend Nebenpunkte'."""
    num = _direct_child(elem, "num")
    heading = _direct_child(elem, "heading")
    pieces = [
        _text_without_footnotes(e) for e in (num, heading) if e is not None
    ]
    return _clean(" ".join(p for p in pieces if p))


def _article_body(article: ET.Element) -> str:
    """Join the article's paragraph contents into one clean string.

    Only paragraph/content text is taken — never the article <num> — so a
    repealed article (whose only text is its number) comes back effectively
    empty and can be detected and skipped.
    """
    paras = [c for c in article if _localname(c.tag) == "paragraph"]
    if paras:
        texts = []
        for p in paras:
            content = _direct_child(p, "content")
            source = content if content is not None else p
            body = _text_without_footnotes(source)
            if body:
                texts.append(body)
        return "\n".join(texts)
    # Some articles hold content directly (no <paragraph> wrapper).
    content = _direct_child(article, "content")
    return _text_without_footnotes(content) if content is not None else ""


def _is_repealed(text: str) -> bool:
    """Repealed provisions carry only the ellipsis placeholder '…' (no real text)."""
    return not text.replace("…", "").strip()


def convert(xml_path: Path, language: str) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}

    def ancestors(elem: ET.Element):
        node = parents.get(elem)
        while node is not None:
            yield node
            node = parents.get(node)

    records: list[dict] = []
    skipped_repealed = 0

    for article in root.iter(f"{A}article"):
        eid = article.get("eId")
        if not eid:
            continue

        num_elem = _direct_child(article, "num")
        art_num = _text_without_footnotes(num_elem) if num_elem is not None else ""

        # Randtitel: heading of the nearest ancestor level marked "marginal".
        randtitel = ""
        breadcrumb: list[str] = []
        for anc in ancestors(article):
            if anc.tag not in STRUCTURAL_TAGS:
                continue
            heading = _heading_text(anc)
            if not heading:
                continue
            if not randtitel and anc.get(ROLE) == "marginal":
                randtitel = _clean(re.sub(r"^\d+\.?\s*", "", heading))  # drop "2. " prefix
            else:
                breadcrumb.append(heading)
        breadcrumb.reverse()  # top-level first

        body = _article_body(article)
        if _is_repealed(body):
            skipped_repealed += 1
            continue

        if randtitel == "…":
            randtitel = ""
        text = f"{randtitel}\n{body}".strip() if randtitel else body

        records.append(
            {
                "_id": eid,
                "title": f"{art_num} OR".strip() if art_num else eid,
                "text": text,
                "metadata": {
                    "law": "OR",
                    "sr": "220",
                    "article": art_num,
                    "marginal_note": randtitel,
                    "hierarchy": " > ".join(breadcrumb),
                    "language": language,
                },
            }
        )

    print(f"parsed {len(records)} articles ({skipped_repealed} repealed/empty skipped)")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="SR-220 Akoma Ntoso XML -> BEIR corpus.jsonl")
    ap.add_argument("--xml", type=Path, default=Path("data/SR-220-01012026-DE.xml"))
    ap.add_argument("--out", type=Path, default=Path("data/or-corpus.jsonl"))
    ap.add_argument("--language", default="de")
    args = ap.parse_args()

    records = convert(args.xml, args.language)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} docs -> {args.out}")


if __name__ == "__main__":
    main()
