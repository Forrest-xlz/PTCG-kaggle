"""Convert replay ZIPs to resumable gzip JSONL training shards."""
from __future__ import annotations
import argparse, gzip, io, json, os, tempfile, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from deck.extract import extract_decks


def _selected(action):
    if isinstance(action, list) and all(isinstance(x, int) for x in action): return action
    return None


def process_archive(job):
    archive, output = Path(job[0]), Path(job[1])
    force, limit = job[2], job[3]
    shard, meta = output / f"{archive.stem}.jsonl.gz", output / f"{archive.stem}.meta.json"
    if shard.exists() and meta.exists() and not force: return {"archive": archive.name, "status": "skipped"}
    output.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(dir=output, suffix=".tmp"); os.close(fd)
    episodes = samples = failed = 0; errors = []
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as out, zipfile.ZipFile(archive) as zf:
            members = [x for x in zf.infolist() if not x.is_dir() and x.filename.endswith(".json")][:limit]
            for member in members:
                try:
                    with zf.open(member) as raw: replay = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
                    decks = extract_decks(replay); episode = replay.get("info", {}).get("EpisodeId", Path(member.filename).stem)
                    for step_index, step in enumerate(replay.get("steps", [])):
                        for player, state in enumerate(step):
                            obs, action = state.get("observation"), _selected(state.get("action"))
                            if not obs or obs.get("current") is None or action is None: continue
                            record = {"episode_id": episode, "date": archive.stem, "step": step_index, "player": player,
                                      "deck": decks[player], "observation": obs, "selected": action}
                            out.write(json.dumps(record, separators=(",", ":")) + "\n"); samples += 1
                    episodes += 1
                except Exception as exc:
                    failed += 1
                    if len(errors) < 100: errors.append({"member": member.filename, "error": str(exc)})
        os.replace(tmp, shard)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    summary = {"archive": archive.name, "status": "written", "episodes": episodes, "samples": samples, "failed": failed, "errors": errors}
    meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=max(1,min(4,(os.cpu_count() or 2)-1))); p.add_argument("--limit-members",type=int); p.add_argument("--force",action="store_true"); n=p.parse_args()
    archives=[n.input] if n.input.is_file() else sorted(n.input.glob("*.zip")); n.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=n.workers) as pool:
        futures=[pool.submit(process_archive,(str(a),str(n.output),n.force,n.limit_members)) for a in archives]
        for f in as_completed(futures): print(json.dumps(f.result(),ensure_ascii=False))
if __name__ == "__main__": main()
