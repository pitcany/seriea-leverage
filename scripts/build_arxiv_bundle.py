"""Assemble a self-contained arXiv submission tarball.

arXiv compiles each submission in a flat directory with no access to anything
outside it, so a source tree that reads figures from a sibling directory will
build locally and fail on upload. This copies the LaTeX source and every figure
it references into one flat directory, verifies the result compiles *from that
directory alone*, and tars it.

The paper's ``\\graphicspath`` lists both the repository location and ``./``, so
the same source builds in place during development and inside the flat bundle.

Run: ``python scripts/build_arxiv_bundle.py``
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from seriea.config import FIGURES_DIR, PROJECT_ROOT

PAPER_DIR = PROJECT_ROOT / "paper"
SOURCE = PAPER_DIR / "seriea-leverage-arxiv.tex"
BUNDLE_DIR = PAPER_DIR / "arxiv"
TARBALL = PAPER_DIR / "seriea-leverage-arxiv.tar.gz"

#: LaTeX run limit. Two passes resolve references; a third settles page numbers.
_LATEX_PASSES: int = 3


def referenced_figures(source: Path) -> list[str]:
    """List the figure filenames the source includes.

    Args:
        source: Path to the LaTeX file.

    Returns:
        Figure filenames in order of first appearance, deduplicated.

    Raises:
        ValueError: If a reference carries a directory component, which would
            not survive flattening.
    """
    text = source.read_text()
    names: list[str] = []
    for match in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        if "/" in match:
            raise ValueError(
                f"Figure reference {match!r} carries a path; arXiv needs flat names."
            )
        if match not in names:
            names.append(match)
    return names


def build() -> Path:
    """Assemble the bundle, verify it compiles in isolation, and archive it.

    Returns:
        Path to the tarball.

    Raises:
        FileNotFoundError: If the source or any referenced figure is missing.
        RuntimeError: If the flattened bundle fails to compile.
    """
    if not SOURCE.exists():
        raise FileNotFoundError(f"Paper source not found at {SOURCE}.")

    figures = referenced_figures(SOURCE)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)

    shutil.copy2(SOURCE, BUNDLE_DIR / SOURCE.name)
    for name in figures:
        origin = FIGURES_DIR / name
        if not origin.exists():
            raise FileNotFoundError(
                f"Figure {name} referenced by the paper is missing from {FIGURES_DIR}. "
                "Run scripts/paper_supplements.py first."
            )
        shutil.copy2(origin, BUNDLE_DIR / name)

    # Compile in a scratch copy so the verification cannot leave aux files in
    # the bundle we are about to ship.
    with tempfile.TemporaryDirectory() as scratch:
        staging = Path(scratch) / "build"
        shutil.copytree(BUNDLE_DIR, staging)
        for _ in range(_LATEX_PASSES):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", SOURCE.name],
                cwd=staging,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            tail = "\n".join(result.stdout.splitlines()[-25:])
            raise RuntimeError(f"Flat bundle failed to compile:\n{tail}")
        pages = (staging / SOURCE.with_suffix(".pdf").name).exists()
        if not pages:
            raise RuntimeError("Flat bundle produced no PDF.")

    with tarfile.open(TARBALL, "w:gz") as archive:
        for item in sorted(BUNDLE_DIR.iterdir()):
            archive.add(item, arcname=item.name)

    return TARBALL


def main() -> None:
    """Build the bundle and report its contents."""
    tarball = build()
    with tarfile.open(tarball) as archive:
        names = archive.getnames()
    size_kb = tarball.stat().st_size / 1024
    print(f"Wrote {tarball} ({size_kb:.0f} KB)")
    print("Contents (flat, self-contained):")
    for name in names:
        print(f"  {name}")
    print("\nVerified: compiles from the flattened directory alone.")


if __name__ == "__main__":
    main()
