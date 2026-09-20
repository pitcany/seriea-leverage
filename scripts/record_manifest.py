"""Record exact local inputs, runtime and generated forecast hashes; no raw redistribution."""
from pathlib import Path
import hashlib,json,platform,sys,subprocess
from importlib.metadata import distributions
from seriea.config import PROJECT_ROOT
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    root=PROJECT_ROOT
    raw=[]
    for path in sorted((root/"data/raw").glob("I1_*.csv")):
        season=path.stem.split("_")[1]
        raw.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,
            "sha256":sha(path),"source":f"https://www.football-data.co.uk/mmz4281/{season}/I1.csv"})
    (root/"reports/data_manifest.json").write_text(json.dumps({"snapshot_date":"2026-09-20",
        "redistribution_permission":"not established","files":raw},indent=2))
    artifacts={}
    for directory,pattern in (("reports","*.csv"),("reports","*.parquet"),("src","*.py"),("scripts","*.py"),("tests","*.py"),("paper","*.tex"),("paper","*.pdf"),("paper","*.md"),("paper","*.tar.gz"),("paper","*.zip")):
        for path in sorted((root/directory).rglob(pattern)):
            artifacts[str(path.relative_to(root))]=sha(path)
    for name in ("fiorentina_stakes_settings.json","survival_subgroup_settings.json","abstract_metadata.json","manuscript_values.json","verification.json"):
        path=root/"reports"/name
        artifacts[str(path.relative_to(root))]=sha(path)
    diff=subprocess.run(["git","diff","--binary"],cwd=root,capture_output=True,check=True).stdout
    manifest={"python":sys.version,"platform":platform.platform(),
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        "tracked_diff_sha256":hashlib.sha256(diff).hexdigest(),
        "packages":{d.metadata["Name"]:d.version for d in distributions()},
        "artifacts":artifacts}
    (root/"reports/reproduction_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Recorded {len(raw)} source files and {len(artifacts)} analysis/source files.")
if __name__=="__main__":main()
