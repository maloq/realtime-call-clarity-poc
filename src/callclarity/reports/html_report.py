from __future__ import annotations

import html
import re


_IMAGE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)$")


def markdown_to_basic_html(markdown: str, title: str = "Call Clarity Report") -> str:
    body_lines = []
    for line in markdown.splitlines():
        image = _IMAGE_RE.match(line.strip())
        if image:
            src = html.escape(image.group("src"), quote=True)
            alt = html.escape(image.group("alt"), quote=True)
            body_lines.append(f"<img src=\"{src}\" alt=\"{alt}\">")
        elif line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body_lines.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("|"):
            body_lines.append(f"<pre>{html.escape(line)}</pre>")
        elif line.strip():
            body_lines.append(f"<p>{html.escape(line)}</p>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:32px auto;"
        "line-height:1.45}img{max-width:100%;height:auto}"
        "pre{background:#f5f5f5;padding:6px;overflow-x:auto}</style></head><body>"
        + "\n".join(body_lines)
        + "</body></html>"
    )
