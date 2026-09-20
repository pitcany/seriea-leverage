"""Build verified manuscript sources and a ZIP with the PDF and Sloan abstract."""
from pathlib import Path
import re,shutil,subprocess,tarfile,tempfile,zipfile
from seriea.config import PROJECT_ROOT
PAPER_DIR=PROJECT_ROOT/"paper"
SOURCE=PAPER_DIR/"seriea-leverage-arxiv.tex"
TARBALL=PAPER_DIR/"seriea-leverage-arxiv.tar.gz"
ZIP_BUNDLE=PAPER_DIR/"seriea-sloan-bundle.zip"

def build():
    text=SOURCE.read_text()
    with tempfile.TemporaryDirectory() as scratch:
        staging=Path(scratch)
        for name in re.findall(r"\\input\{([^}]+)\}",text):
            path=PAPER_DIR/name
            shutil.copy2(path,staging/path.name)
            text=text.replace("{"+name+"}","{"+path.name+"}")
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",text):
            path=(PAPER_DIR/name).resolve()
            shutil.copy2(path,staging/path.name)
            text=text.replace("{"+name+"}","{"+path.name+"}")
        (staging/SOURCE.name).write_text(text)
        source_files=sorted(p.name for p in staging.iterdir())
        for _ in range(3):
            result=subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",SOURCE.name],
                cwd=staging,capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError("\n".join(result.stdout.splitlines()[-30:]))
        with tarfile.open(TARBALL,"w:gz") as archive:
            for name in source_files:
                archive.add(staging/name,arcname=name)
        shutil.copy2(PAPER_DIR/"SLOAN_ABSTRACT.md",staging/"SLOAN_ABSTRACT.md")
        title=re.search(r"\\title\{([^}]+)\}",text).group(1)
        (staging/"README.txt").write_text(
            title+"\n"
            "Yannik Pitcan\n\n"
            "This manuscript bundle contains the compiled paper, its complete LaTeX\n"
            "sources, generated tables and figures, and the Sloan abstract.\n\n"
            "Read: seriea-leverage-arxiv.pdf\n"
            "Abstract: SLOAN_ABSTRACT.md\n"
            "Rebuild: latexmk -pdf seriea-leverage-arxiv.tex\n"
            "Keep all files together when compiling. A standard LaTeX installation\n"
            "with the packages declared in the manuscript is required.\n\n"
            "Repository: https://github.com/pitcany/seriea-leverage\n"
            "Revision branch: sloan-scientific-revision\n"
            "The source code and analytical reports are in the repository.\n"
            "This ZIP contains manuscript materials, not the raw match dataset.\n"
            "The manuscript is a draft and includes the author's name.\n"
        )
        zip_files=source_files+[SOURCE.with_suffix(".pdf").name,"SLOAN_ABSTRACT.md","README.txt"]
        with zipfile.ZipFile(ZIP_BUNDLE,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name in sorted(zip_files):
                archive.write(staging/name,arcname=name)
        with zipfile.ZipFile(ZIP_BUNDLE) as archive:
            if archive.testzip() is not None or sorted(archive.namelist())!=sorted(zip_files):
                raise RuntimeError("ZIP bundle integrity check failed.")
        print(f"Verified flat compilation; wrote {TARBALL} and {ZIP_BUNDLE}")
    return TARBALL
if __name__=="__main__":build()
