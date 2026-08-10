"""Assemble the shareable HTML report.

``reports/artifact.html`` is the source: it carries placeholder tokens where the
webfonts belong. This script substitutes base64-encoded WOFF2 subsets and writes
``reports/artifact.built.html``, which is fully self-contained — no external
requests, so it renders identically offline and behind a strict content
security policy.

The fonts are EB Garamond subsets under the SIL Open Font License; see
``assets/fonts/LICENSE.txt``. Regenerate the subsets with::

    python -m fontTools.subset EBGaramond12-Regular.otf \\
        --unicodes="U+0020-007E,U+00A0-00FF,U+2010-2027,..." \\
        --layout-features="kern,liga,onum,tnum,frac" --flavor=woff2

Run: ``python scripts/build_artifact.py``
"""

from __future__ import annotations

import base64
from pathlib import Path

from seriea.config import PROJECT_ROOT, REPORTS_DIR

#: Placeholder token in the source HTML -> font file supplying its data.
FONT_SLOTS: dict[str, str] = {
    "__FONT_REGULAR__": "EBGaramond12-Regular.woff2",
    "__FONT_ITALIC__": "EBGaramond12-Italic.woff2",
    "__FONT_BOLD__": "EBGaramond12-Bold.woff2",
}

SOURCE = REPORTS_DIR / "artifact.html"
OUTPUT = REPORTS_DIR / "artifact.built.html"
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"


def build() -> Path:
    """Inline every font and write the self-contained report.

    Returns:
        Path to the built HTML.

    Raises:
        FileNotFoundError: If the source HTML or any font file is missing.
        ValueError: If a placeholder is absent from the source, which would
            silently ship a page with no webfont.
    """
    if not SOURCE.exists():
        raise FileNotFoundError(f"Artifact source not found at {SOURCE}.")

    html = SOURCE.read_text()
    for token, filename in FONT_SLOTS.items():
        if token not in html:
            raise ValueError(f"Placeholder {token} missing from {SOURCE.name}.")

        font_path = FONT_DIR / filename
        if not font_path.exists():
            raise FileNotFoundError(f"Font subset not found at {font_path}.")

        html = html.replace(token, base64.b64encode(font_path.read_bytes()).decode())

    OUTPUT.write_text(html)
    return OUTPUT


def main() -> None:
    """Build the report and print a short summary."""
    output = build()
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
