"""Export W&B experiments for matching runs to this repository's branches.

Examples (run from the repository root in your training environment):
    python export_wandb_runs.py
    python export_wandb_runs.py --history
    python export_wandb_runs.py --history --run-id RUN_ID

Uses your existing W&B credentials and server settings. Requires: pip install wandb
The default export directory is outputs/wandb_export/<timestamp>/.
If the SDK transport returns HTML 403, metadata is retried using direct HTTPS.
That fallback exports config and summary only; --history reports an incomplete
export rather than claiming that full curves were downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from itertools import chain
from pathlib import Path


def _is_html_403(error: Exception) -> bool:
    """Distinguish the SDK transport failure from a project authorization error."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message = str(error).lower()
        if "403" in message and ("<!doctype html" in message or "<html" in message):
            return True
        error = error.__cause__ or error.__context__
    return False


def _http_records(project: str, run_ids: list[str]):
    """Read the same W&B GraphQL fields over Python HTTPS, without wandb-core."""
    import requests

    base_url = os.environ.get("WANDB_BASE_URL", "https://api.forge.coreweave.com").rstrip("/")
    auth = requests.utils.get_netrc_auth(base_url)
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        auth = ("api", api_key)
    elif not auth and base_url in {
        "https://api.forge.coreweave.com", "https://forge.coreweave.com/api/wandb"
    }:
        # Official Forge and W&B SaaS share credentials. Never forward a SaaS
        # credential to an arbitrary custom server.
        auth = requests.utils.get_netrc_auth("https://api.wandb.ai")
    if not auth:
        raise RuntimeError("No W&B credentials found for HTTPS export. Run 'wandb login'.")

    entity, project_name = project.split("/", 1)
    query = """
    query ExportRuns($entity: String!, $project: String!, $cursor: String) {
      project(name: $project, entityName: $entity) {
        runs(first: 50, after: $cursor) {
          edges { cursor node {
            name displayName state createdAt commit group tags description
            config summaryMetrics
          } }
          pageInfo { hasNextPage }
        }
      }
    }
    """
    selected = set(run_ids)
    found = set()
    cursor = None
    with requests.Session() as session:
        session.auth = auth
        while True:
            response = session.post(
                base_url + "/graphql",
                json={"query": query, "variables": {
                    "entity": entity, "project": project_name, "cursor": cursor,
                }},
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("errors"):
                raise RuntimeError("; ".join(e["message"] for e in body["errors"]))
            data = (body.get("data") or {}).get("project")
            if data is None:
                raise RuntimeError(f"Project not found or inaccessible: {project}")
            page = data["runs"]
            edges = page["edges"]
            for edge in edges:
                node = edge["node"]
                run_id = node["name"]
                if selected and run_id not in selected:
                    continue
                found.add(run_id)
                raw_config = node.get("config") or {}
                if isinstance(raw_config, str):
                    raw_config = json.loads(raw_config)
                config = {
                    key: value.get("value", value) if isinstance(value, dict) else value
                    for key, value in raw_config.items() if not key.startswith("_")
                }
                summary = node.get("summaryMetrics") or {}
                if isinstance(summary, str):
                    summary = json.loads(summary)
                yield {
                    "id": run_id, "name": node["displayName"],
                    "url": f"https://forge.coreweave.com/wandb/{project}/runs/{run_id}",
                    "project": project, "state": node["state"],
                    "created_at": node["createdAt"], "git_commit": node.get("commit"),
                    "group": node.get("group"), "tags": node.get("tags"),
                    "notes": node.get("description"), "config": config, "summary": summary,
                }, None
            if not page["pageInfo"]["hasNextPage"]:
                break
            if not edges or edges[-1]["cursor"] == cursor:
                raise RuntimeError("W&B pagination did not advance; export is incomplete.")
            cursor = edges[-1]["cursor"]
    if selected - found:
        raise RuntimeError(f"Run IDs not found: {', '.join(sorted(selected - found))}")


def _records(wandb, args):
    try:
        api = wandb.Api(timeout=120)
        if args.run_id:
            runs = (api.run(f"{args.project}/{run_id}") for run_id in dict.fromkeys(args.run_id))
        else:
            runs = iter(api.runs(args.project, per_page=50))
        # Fetch the first page here so lazy SDK transport errors can fall back
        # before any records have been exported (no duplicated runs).
        first = next(runs, None)
    except Exception as exc:
        if not _is_html_403(exc):
            raise
        print("SDK transport returned HTML 403; retrying metadata via HTTPS.", flush=True)
        yield from _http_records(args.project, args.run_id)
        return
    for run in chain(() if first is None else (first,), runs):
        yield {
            "id": run.id, "name": run.name, "url": run.url,
            "project": args.project, "state": run.state,
            "created_at": run.created_at, "git_commit": getattr(run, "commit", None),
            "group": run.group, "tags": run.tags, "notes": run.notes,
            "config": run.config, "summary": run.summary._json_dict,
        }, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", default="xlz1014361885-tokyo/ptcg_bc",
        help="W&B entity/project (default: %(default)s).",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Export directory; defaults to a new timestamped directory.",
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Also export every logged history record, without sampling.",
    )
    parser.add_argument(
        "--run-id", action="append", default=[], metavar="ID",
        help="Export only this run ID; repeat to select multiple runs.",
    )
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        parser.exit(1, "W&B is not installed. Run: python -m pip install wandb\n")

    output = args.output or (
        Path(__file__).resolve().parent / "outputs" / "wandb_export"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "runs.jsonl"
    if index_path.exists():
        parser.exit(1, f"Export already exists: {index_path}\nUse a new --output directory.\n")

    count = 0
    failures = []
    print(f"Project: {args.project}", flush=True)
    print(f"Output: {output}", flush=True)

    try:
        # One JSON object per line preserves completed records if interrupted.
        with index_path.open("x", encoding="utf-8") as index_file:
            for record, run in _records(wandb, args):
                index_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                index_file.flush()
                count += 1
                print(f"[{count}] {record['id']}: {record['name']}", flush=True)

                if args.history:
                    if run is None:
                        failures.append(record["id"])
                        continue
                    history_dir = output / "history"
                    history_dir.mkdir(exist_ok=True)
                    history_path = history_dir / f"{run.id}.jsonl"
                    rows = 0
                    try:
                        with history_path.open("x", encoding="utf-8") as history_file:
                            # No keys filter: metrics logged on different steps
                            # must not be restricted to their shared records.
                            for row in run.scan_history(page_size=500):
                                history_file.write(
                                    json.dumps(row, ensure_ascii=False, default=str) + "\n"
                                )
                                rows += 1
                        print(f"    history: {rows} records", flush=True)
                    except Exception as exc:
                        failures.append(run.id)
                        print(f"    history incomplete for {run.id}: {exc}", flush=True)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Preserved {count} run records in {index_path}")
        return 130
    except Exception as exc:
        print(f"\nExport stopped: {exc}")
        print(f"Preserved {count} run records in {index_path}")
        print("If authentication is required, run 'wandb login' in this environment.")
        return 1

    if failures:
        print(f"\nRun index complete: {index_path}")
        print(f"Incomplete history IDs: {', '.join(failures)}")
        print("Retry those IDs using --history --run-id ID in a new output directory.")
        print("If SDK transport still fails, update the W&B SDK or use another network.")
        return 1

    print(f"\nExport complete: {count} runs in {index_path}")
    if args.history:
        print(f"Full histories: {output / 'history'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
