#!/usr/bin/env python3
"""Render source-backed documentation diagrams using only the Python stdlib."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import io
import json
from pathlib import Path
import re
import sys
import unicodedata
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs/graphs"
BUNDLE = "mimir_framework_svg_pack_v3_corrected.zip"
WIDTH, MARGIN, COL_GAP, ROW_GAP = 1200, 44, 68, 100
TONES = {
    "blue": ("#2563eb", "#eff6ff"),
    "teal": ("#0f766e", "#ecfdf5"),
    "purple": ("#7c3aed", "#f5f3ff"),
    "orange": ("#b45309", "#fffbeb"),
    "slate": ("#475569", "#f8fafc"),
    "red": ("#b91c1c", "#fef2f2"),
}


def text_width(text: str, size: float, bold: bool = False) -> float:
    """Conservative DejaVu Sans advance estimate for portable line breaking."""
    units = 0.0
    for character in text:
        if unicodedata.combining(character):
            continue
        if unicodedata.east_asian_width(character) in ("W", "F"):
            units += 1.0
        elif character in "ilI.,:;'`!| ":
            units += 0.34
        elif character in "MW@%&":
            units += 0.98
        elif character.isupper():
            units += 0.76
        else:
            units += 0.64
    return units * size * (1.05 if bold else 1.0)


def wrap(text: str, width: float, size: float, bold: bool = False) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            # Preserve every character of long paths and identifiers.
            pieces: list[str] = []
            chunk = ""
            for character in word:
                if chunk and text_width(chunk + character, size, bold) > width:
                    pieces.append(chunk)
                    chunk = ""
                chunk += character
            pieces.append(chunk)
            for index, piece in enumerate(pieces):
                candidate = current + (" " if current else "") + piece
                if current and (index or text_width(candidate, size, bold) > width):
                    lines.append(current)
                    current = piece
                else:
                    current = candidate
                if index + 1 < len(pieces):
                    lines.append(current)
                    current = ""
        lines.append(current)
    return lines


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def intersects(self, other, padding=0):
        return not (self.right + padding <= other.x or other.right + padding <= self.x
                    or self.bottom + padding <= other.y or other.bottom + padding <= self.y)


def number(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def label(lines, x, y, size=20, color="#334155", bold=False, leading=28):
    parts = [f'<text x="{number(x)}" y="{number(y)}" font-size="{size}" fill="{color}"'
             + (' font-weight="700"' if bold else '') + '>']
    for index, line in enumerate(lines):
        parts.append(f'<tspan x="{number(x)}" dy="{0 if index == 0 else leading}">{escape(line)}</tspan>')
    parts.append('</text>')
    return "\n".join(parts)


def node_text(node, card_width):
    titles = wrap(node["title"], card_width - 44, 22, True)
    lines: list[str] = []
    for line in node.get("lines", []):
        lines.extend(wrap(line, card_width - 44, 19))
    badges = wrap(node["badge"], card_width - 44, 14, True) if node.get("badge") else []
    height = 28 + len(badges) * 20 + len(titles) * 29 + (14 if lines else 0) + len(lines) * 26 + 24
    return titles, lines, badges, max(150, height)


def route(source, target, nodes, columns):
    a, b = source["rect"], target["rect"]
    ar, ac, br, bc = source["row"], source["col"], target["row"], target["col"]
    if ar == br:
        between = any(n["row"] == ar and min(ac, bc) < n["col"] < max(ac, bc) for n in nodes.values())
        if not between:
            return [(a.right, a.cy), (b.x, b.cy)] if ac < bc else [(a.x, a.cy), (b.right, b.cy)]
        gutter = min(a.y, b.y) - 32
        return [(a.cx, a.y), (a.cx, gutter), (b.cx, gutter), (b.cx, b.y)]
    if ac == bc:
        between = any(n["col"] == ac and min(ar, br) < n["row"] < max(ar, br) for n in nodes.values())
        if not between:
            return [(a.cx, a.bottom), (b.cx, b.y)] if ar < br else [(a.cx, a.y), (b.cx, b.bottom)]
    if abs(ar - br) == 1:
        gutter = (a.bottom + b.y) / 2 if ar < br else (b.bottom + a.y) / 2
        # Diagonal branches use side ports. They must not share the center
        # port with the opposite-direction vertical main flow.
        direction = 1 if bc > ac else -1
        sx, tx = a.cx + direction * a.w * .22, b.cx - direction * b.w * .22
        start = (sx, a.bottom if ar < br else a.y)
        end = (tx, b.y if ar < br else b.bottom)
        return [start, (sx, gutter), (tx, gutter), end]
    # Long skip / feedback paths use a side corridor, never the inside of a card.
    corridor = WIDTH - 18 if ac + bc >= columns - 1 else 18
    sy = a.bottom + 30 if ar < br else a.y - 30
    ty = b.y - 30 if ar < br else b.bottom + 30
    direction = 1 if corridor > a.cx else -1
    sx, tx = a.cx + direction * a.w * .22, b.cx + direction * b.w * .22
    return [(sx, a.bottom if ar < br else a.y), (sx, sy), (corridor, sy),
            (corridor, ty), (tx, ty), (tx, b.y if ar < br else b.bottom)]


def compact(points):
    result = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result


def render(spec):
    columns = spec.get("columns", 3)
    card_width = (WIDTH - 2 * MARGIN - (columns - 1) * COL_GAP) / columns
    title = wrap(spec["title"], WIDTH - 2 * MARGIN, 31, True)
    subtitle = wrap(spec["subtitle"], WIDTH - 2 * MARGIN, 21)
    first_row = 50 + len(title) * 39 + len(subtitle) * 28 + 46
    nodes = {}
    heights = {}
    for node in spec["nodes"]:
        titles, lines, badges, height = node_text(node, card_width)
        nodes[node["id"]] = dict(node, titles=titles, body=lines, badges=badges, height=height)
        heights[node["row"]] = max(heights.get(node["row"], 0), height)
    row_y = {}
    y = first_row
    for row in range(max(heights) + 1):
        row_y[row] = y
        y += heights.get(row, 0) + ROW_GAP
    for node in nodes.values():
        node["rect"] = Rect(MARGIN + node["col"] * (card_width + COL_GAP),
                            row_y[node["row"]], card_width, heights[node["row"]])
    content_end = max(node["rect"].bottom for node in nodes.values())
    edges = []
    edge_labels = []
    deferred = []
    for edge in spec.get("edges", []):
        source, target = nodes[edge["from"]], nodes[edge["to"]]
        points = compact(route(source, target, nodes, columns))
        color = TONES[edge.get("tone", "slate")][0]
        path = "M " + " L ".join(f'{number(x)},{number(y)}' for x, y in points)
        style = ' stroke-dasharray="7 6"' if edge.get("style") == "dashed" else ''
        edges.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"'
                     f' stroke-linejoin="round" marker-end="url(#arrow-{edge.get("tone", "slate")})"{style}/>')
        if not edge.get("label"):
            continue
        lines = wrap(edge["label"], 180, 15)
        width = max(text_width(line, 15) for line in lines) + 18
        height = len(lines) * 21 + 8
        candidates = []
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            candidates.append(Rect(cx - width / 2, cy - height / 2, width, height))
        chosen = next((candidate for candidate in candidates
                       if candidate.x >= 8 and candidate.right <= WIDTH - 8
                       and candidate.y >= first_row - 40
                       and not any(candidate.intersects(node["rect"], 7) for node in nodes.values())
                       and not any(candidate.intersects(prior, 5) for prior, _, _ in edge_labels)), None)
        if chosen:
            edge_labels.append((chosen, lines, color))
        else:
            deferred.append(f'{source["title"]} → {target["title"]} : {edge["label"]}')
    notes = list(spec.get("notes", [])) + deferred
    note_lines = []
    for note in notes:
        note_lines.extend(wrap("• " + note, WIDTH - 2 * MARGIN - 44, 18))
    note_y = content_end + 48
    note_height = 32 + len(note_lines) * 26 if note_lines else 0
    sources = "Sources : " + " · ".join(Path(path).name for path in spec["sources"])
    source_lines = wrap(sources, WIDTH - 2 * MARGIN, 14)
    footer_y = note_y + note_height + 30
    total_height = int(footer_y + len(source_lines) * 21 + 35)
    description = spec["description"] + " Sources : " + ", ".join(spec["sources"])
    metadata = json.dumps({"generator": "scripts/docs/generate_diagrams.py", "sources": spec["sources"],
                           "related_docs": spec.get("related_docs", [])}, ensure_ascii=False)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{total_height}"'
             f' viewBox="0 0 {WIDTH} {total_height}" role="img" aria-labelledby="title desc">',
             f'<title id="title">{escape(spec["title"])}</title>',
             f'<desc id="desc">{escape(description)}</desc>',
             f'<metadata>{escape(metadata)}</metadata>', '<defs>']
    for name, (color, _) in TONES.items():
        parts.append(f'<marker id="arrow-{name}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
                     f'<path d="M1,1 L8,4.5 L1,8 Z" fill="{color}"/></marker>')
    parts.extend(['</defs>', '<g font-family="DejaVu Sans, Liberation Sans, Arial, sans-serif">',
                  f'<rect width="{WIDTH}" height="{total_height}" fill="#ffffff"/>',
                  label(title, MARGIN, 47, 31, "#0f172a", True, 39),
                  label(subtitle, MARGIN, 47 + len(title) * 39, 21, "#475569", leading=28)])
    parts.extend(edges)
    for node in nodes.values():
        rect = node["rect"]
        color, background = TONES[node.get("tone", "blue")]
        parts.append(f'<g data-node="{escape(node["id"], quote=True)}">')
        parts.append(f'<rect x="{number(rect.x)}" y="{number(rect.y)}" width="{number(rect.w)}"'
                     f' height="{number(rect.h)}" rx="12" fill="{background}" stroke="{color}" stroke-width="1.4"/>')
        tx, ty = rect.x + 22, rect.y + 31
        if node["badges"]:
            parts.append(label(node["badges"], tx, ty, 14, color, True, 20))
            ty += len(node["badges"]) * 20 + 7
        parts.append(label(node["titles"], tx, ty, 22, "#0f172a", True, 29))
        ty += len(node["titles"]) * 29 + 14
        if node["body"]:
            parts.append(label(node["body"], tx, ty, 19, "#334155", leading=26))
        parts.append('</g>')
    for rect, lines, color in edge_labels:
        parts.append(f'<rect x="{number(rect.x)}" y="{number(rect.y)}" width="{number(rect.w)}"'
                     f' height="{number(rect.h)}" rx="5" fill="#ffffff" stroke="#e2e8f0"/>')
        parts.append(label(lines, rect.x + 9, rect.y + 19, 15, color, leading=21))
    if note_lines:
        parts.append(f'<rect x="{MARGIN}" y="{number(note_y)}" width="{WIDTH - 2 * MARGIN}"'
                     f' height="{number(note_height)}" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
        parts.append(label(note_lines, MARGIN + 22, note_y + 28, 18, "#334155", leading=26))
    parts.append(label(source_lines, MARGIN, footer_y + 5, 14, "#64748b", leading=21))
    parts.extend(['</g>', '</svg>', ''])
    return "\n".join(parts)


def load_specs():
    specs = []
    for path in sorted(SPEC_DIR.glob("diagrams_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: expected an array")
        specs.extend(data)
    seen = set()
    for spec in specs:
        filename = spec["file"]
        if not re.fullmatch(r"\d{2}_[a-z0-9_]+\.svg", filename) or filename in seen:
            raise ValueError(f"Invalid or duplicate diagram filename: {filename}")
        seen.add(filename)
        columns = spec.get("columns", 3)
        if not 1 <= columns <= 4 or not spec["nodes"] or not spec["sources"]:
            raise ValueError(f"{filename}: missing nodes/sources or invalid columns")
        identifiers, positions = set(), set()
        for node in spec["nodes"]:
            position = node["row"], node["col"]
            if (node["id"] in identifiers or position in positions or node["row"] < 0
                    or not 0 <= node["col"] < columns or node.get("tone", "blue") not in TONES):
                raise ValueError(f"{filename}: duplicate or invalid node {node['id']}")
            identifiers.add(node["id"])
            positions.add(position)
        for edge in spec.get("edges", []):
            if (edge["from"] not in identifiers or edge["to"] not in identifiers
                    or edge.get("tone", "slate") not in TONES):
                raise ValueError(f"{filename}: invalid edge {edge}")
        for relative in spec["sources"] + spec.get("related_docs", []):
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or not path.exists():
                raise ValueError(f"{filename}: missing local reference {relative}")
    expected = {path.name for path in OUTPUT.glob("[0-9][0-9]_*.svg")}
    if expected - seen:
        raise ValueError(f"Manifest coverage mismatch: missing={sorted(expected - seen)}")
    return sorted(specs, key=lambda spec: spec["file"])


def bundle_bytes(diagrams):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(diagrams.items()):
            entry = zipfile.ZipInfo("mimir_framework_svg_pack_v3/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content.encode("utf-8"))
        readme = OUTPUT / "README.md"
        if readme.exists():
            entry = zipfile.ZipInfo("mimir_framework_svg_pack_v3/README.md", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, readme.read_bytes())
    return result.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify manifests and generated SVG contents without writing")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="optional preview directory")
    parser.add_argument("--bundle", action="store_true", help="also regenerate/check the existing downloadable SVG bundle")
    args = parser.parse_args()
    diagrams = {spec["file"]: render(spec) for spec in load_specs()}
    stale = []
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in diagrams.items():
        target = args.output_dir / filename
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                stale.append(str(target))
        else:
            target.write_text(content, encoding="utf-8")
    if args.bundle:
        target = args.output_dir / BUNDLE
        content = bundle_bytes(diagrams)
        if args.check:
            if not target.exists() or target.read_bytes() != content:
                stale.append(str(target))
        else:
            target.write_bytes(content)
    if stale:
        print("Stale diagram artifacts:\n" + "\n".join(stale), file=sys.stderr)
        return 1
    print(f"Documentation diagrams {'checked' if args.check else 'generated'}: {len(diagrams)} SVG"
          + (", bundle included" if args.bundle else ""))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"Diagram generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
