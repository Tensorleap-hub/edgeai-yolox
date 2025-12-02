#!/usr/bin/env python3
"""
Tiny CLI to read Tensorleap docs from the terminal.

Examples:
  python tensorleap_docs_cli.py integration
  python tensorleap_docs_cli.py writing-integration-code
  python tensorleap_docs_cli.py integration-test
"""

import sys
import textwrap
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("This script requires 'requests' and 'beautifulsoup4':")
    print("  pip install requests beautifulsoup4")
    sys.exit(1)


BASE_URL = "https://docs.tensorleap.ai/"


# Mapping simple keywords -> real doc paths
PAGE_MAP = {
    "integration": "tensorleap-integration",
    "tensorleap-integration": "tensorleap-integration",
    "writing-integration-code": "tensorleap-integration/writing-integration-code",
    "integration-test": "tensorleap-integration/integration-test",
    "model-integration": "tensorleap-integration/model-integration",
    "leap-yaml": "tensorleap-integration/leap.yaml",
}


def fetch_page(path: str) -> str:
    url = urljoin(BASE_URL, path)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Some GitBook pages put main content in <main> or specific divs.
    # If we find a main element, focus on that.
    main = soup.find("main")
    if main:
        root = main
    else:
        # Fallback: just use body
        root = soup.body or soup

    lines = []

    def add(text="", indent=0, blank_before=False):
        if not text:
            return
        wrapped = textwrap.fill(text.strip(), width=100,
                                subsequent_indent=" " * indent)
        if blank_before and lines and lines[-1] != "":
            lines.append("")
        lines.append(wrapped)

    # basic traversal: headings, paragraphs, list items, code
    for element in root.descendants:
        if element.name and element.name.startswith("h"):
            # Heading
            level = int(element.name[1]) if element.name[1].isdigit() else 2
            heading_text = element.get_text(separator=" ", strip=True)
            if heading_text:
                prefix = "#" * level + " "
                add(prefix + heading_text, blank_before=True)
        elif element.name == "p":
            text = element.get_text(separator=" ", strip=True)
            if text:
                add(text, blank_before=True)
        elif element.name in ("li",):
            text = element.get_text(separator=" ", strip=True)
            if text:
                add(f"- {text}", indent=2)
        elif element.name in ("pre", "code"):
            # Code blocks: keep them more verbatim
            code_text = element.get_text("\n", strip=True)
            if code_text:
                lines.append("")
                lines.append("```")
                lines.extend(code_text.splitlines())
                lines.append("```")

    # Deduplicate consecutive blank lines
    cleaned = []
    for l in lines:
        if l == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(l)

    return "\n".join(cleaned).strip() + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tensorleap_docs_cli.py integration")
        print("  python tensorleap_docs_cli.py writing-integration-code")
        print("  python tensorleap_docs_cli.py integration-test")
        sys.exit(1)

    key = sys.argv[1].strip().lower()

    if key not in PAGE_MAP:
        print(f"Unknown page key: {key}")
        print("Known keys:")
        for k in sorted(PAGE_MAP.keys()):
            print(" ", k)
        sys.exit(1)

    path = PAGE_MAP[key]

    try:
        html = fetch_page(path)
    except Exception as e:
        print(f"Error fetching page: {e}")
        sys.exit(1)

    text = html_to_text(html)
    print(text)


if __name__ == "__main__":
    main()
