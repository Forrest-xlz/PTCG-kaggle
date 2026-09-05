"""Select isolation decks, publish a review report, and export audited selections."""
from __future__ import annotations

import base64
import html
import io
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.deck_statistics import (
    CardCatalog, annotate_deck_facts, build_deck_census, load_deck_rows,
    exact_deck_identity,
)
from analysis.deck_selection import (
    audit_deck_archetype_selection, build_archetype_isolation_candidates,
    build_archetype_selection_details, build_deck_isolation_candidates,
    build_top_deck_archetype_candidates, roll_archetype_isolation_selection,
    roll_deck_isolation_selection, roll_top_deck_archetype_selection,
)
from validation.deck_names import parse_top_decks

CONFIG_PATH = PROJECT_ROOT / "cfg" / "select_validation_decks.yaml"
GROUPS = ("deck_isolation", "archetype_isolation", "top_deck_archetype_isolation")


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_settings(path=CONFIG_PATH):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tops = cfg["top_deck_archetype_isolation"]
    if isinstance(tops, dict):
        tops = [dict(tops, name=tops.get("name", tops.get("archetype", "top_deck")))]
    parse_top_decks(tops)
    cfg["top_deck_archetype_isolation"] = tops
    for group in [cfg["deck_isolation"], cfg["archetype_isolation"], *tops]:
        for key in ("min_validation_replays", "min_count", "total_replays_min", "total_replays_max", "roll_id"):
            if type(group.get(key)) is not int or group[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if group["min_count"] < 1 or group["total_replays_max"] < group["total_replays_min"]:
            raise ValueError("Invalid sampling count or replay range")
    for top in tops:
        if not isinstance(top.get("archetype"), str) or not top["archetype"].strip():
            raise ValueError("Each top deck requires an archetype")
    return cfg


def _roll_options(cfg):
    return {k: cfg[k] for k in ("min_count", "total_replays_min", "total_replays_max", "roll_id")}


def _selected(candidates, roll, card_lookup):
    selected = candidates.set_index("deck_id").loc[list(roll.selected)].reset_index()
    selected["card_ids"] = selected["deck_id"].map(card_lookup)
    return selected


def _top_candidates(facts, top, protected_ids):
    candidates = build_top_deck_archetype_candidates(
        facts, target_archetype=top["archetype"],
        top_deck_card_ids=top["card_ids"],
        min_validation_replays=top["min_validation_replays"],
    )
    candidates["is_top_deck"] = candidates["deck_id"].isin(protected_ids)
    candidates["eligible"] &= ~candidates["is_top_deck"]
    candidates.insert(0, "name", top["name"].strip())
    return candidates


def _archetype_exclusions(top_archetypes, selected_decks):
    """Exact-deck holdouts require their archetype to remain in training."""
    return tuple(sorted(set(top_archetypes) | set(selected_decks["deck_archetype"])))


def _audit_error(audit):
    reasons = []
    for key in ("conflicting_archetypes", "missing_deck_archetypes", "missing_deck_card_ids"):
        if audit.get(key):
            reasons.append(f"{key}={audit[key]}")
    for key in ("exact_deck_train_uses", "archetype_train_uses", "top_deck_selected"):
        if audit.get(key):
            reasons.append(f"{key}={audit[key]}")
    if not audit.get("remaining_training_replays", 0):
        reasons.append("no remaining training replays")
    return "Combined isolation audit failed: " + "; ".join(reasons)


def _figure(fig):
    try:
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
        return '<img alt="Selection overview" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode("ascii") + '">'
    finally:
        plt.close(fig)


def _charts(deck_candidates, selected, archetypes):
    charts = []
    fig, ax = plt.subplots(figsize=(10, 5))
    for band, color in (("high", "#D55E00"), ("moderate", "#0072B2"), ("lower", "#009E73")):
        rows = deck_candidates.loc[deck_candidates["eligible"] & deck_candidates["similarity_band"].eq(band)]
        ax.scatter(rows["reference_changed_slots"], rows["replays"], label=band, color=color)
    if not selected.empty:
        ax.scatter(selected["reference_changed_slots"], selected["replays"], marker="*", s=180, color="gold", edgecolor="black", label="Selected")
    ax.set(xlabel="Changed slots to training reference", ylabel="Unique replays", title="Deck isolation candidates")
    ax.legend()
    charts.append(_figure(fig))
    for column, title, subset in (
        ("unique_exact_decks", "Exact decks per archetype", archetypes),
        ("replays", "Eligible archetype replay support", archetypes.loc[archetypes["eligible"]]),
    ):
        rows = subset.sort_values(column)
        fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * .3)))
        ax.barh(rows["deck_archetype"], rows[column])
        ax.set(title=title, xlabel=column)
        charts.append(_figure(fig))
    return charts


def _publish(files, output):
    """Stage bytes before replacing the explicitly owned files."""
    output.mkdir(parents=True, exist_ok=True)
    staging = output / (".selection-staging-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        for name, content in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        for name in files:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / name, destination)
    finally:
        # staging is an explicitly created UUID child of the resolved output.
        shutil.rmtree(staging)


def run(cfg):
    paths = sorted(_path(cfg["inputs"]["deck_data"]).glob("*.decks.csv"))
    if not paths:
        raise FileNotFoundError("No extracted .decks.csv files found")
    print(f"Loading {len(paths)} deck extracts", flush=True)
    facts = load_deck_rows(paths)
    catalog = CardCatalog.from_csv(_path(cfg["inputs"]["card_table"]))
    summary = build_deck_census(facts, catalog)
    facts = annotate_deck_facts(facts, catalog)
    card_lookup = summary.set_index("deck_id")["deck_card_ids_json"]
    tops = cfg["top_deck_archetype_isolation"]
    protected_ids = {exact_deck_identity(t["card_ids"]).deck_id for t in tops}
    excluded_archetypes = tuple(dict.fromkeys(t["archetype"] for t in tops))
    tables, selections, messages = {}, {}, []
    selected_archetype_summary = pd.DataFrame()
    top_candidates, top_selected = [], []
    error = None
    audit = {"valid": False}
    try:
        for top in tops:
            print(f"Selecting deck({top['name']}) variants", flush=True)
            candidates = _top_candidates(facts, top, protected_ids)
            top_candidates.append(candidates)
            tables["top_deck_archetype_candidates"] = pd.concat(top_candidates, ignore_index=True)
            roll = roll_top_deck_archetype_selection(facts, candidates, **_roll_options(top))
            top_selected.append(_selected(candidates, roll, card_lookup))
            messages.append(f"deck({top['name']}): {roll.replay_count} unique replays, {roll.attempts} attempts")
        selections[GROUPS[2]] = pd.concat(top_selected, ignore_index=True)
        deck_cfg = cfg[GROUPS[0]]
        candidates = build_deck_isolation_candidates(facts, min_validation_replays=deck_cfg["min_validation_replays"], excluded_archetypes=excluded_archetypes)
        tables["deck_isolation_candidates"] = candidates
        roll = roll_deck_isolation_selection(facts, candidates, excluded_archetypes=excluded_archetypes, **_roll_options(deck_cfg))
        selections[GROUPS[0]] = _selected(candidates, roll, card_lookup)
        messages.append(f"Deck isolation: {roll.replay_count} unique replays, {roll.attempts} attempts")
        arch_cfg = cfg[GROUPS[1]]
        archetype_exclusions = _archetype_exclusions(
            excluded_archetypes, selections[GROUPS[0]]
        )
        candidates = build_archetype_isolation_candidates(facts, catalog, min_validation_replays=arch_cfg["min_validation_replays"])
        candidates["excluded_by_deck_isolation"] = candidates["deck_archetype"].isin(archetype_exclusions)
        candidates["eligible"] &= ~candidates["excluded_by_deck_isolation"]
        tables["archetype_isolation_candidates"] = candidates
        roll = roll_archetype_isolation_selection(facts, candidates, excluded_archetypes=archetype_exclusions, **_roll_options(arch_cfg))
        selected_arches = tuple(roll.selected)
        selected_archetype_summary = candidates.loc[candidates["deck_archetype"].isin(selected_arches)].copy()
        selections[GROUPS[1]] = build_archetype_selection_details(facts, selected_arches)
        messages.append(f"Archetype isolation: {roll.replay_count} unique replays, {roll.attempts} attempts")
        selected_ids = tuple(set(selections[GROUPS[0]]["deck_id"]) | set(selections[GROUPS[2]]["deck_id"]))
        audit = audit_deck_archetype_selection(facts, selected_ids, selected_arches)
        audit["top_deck_selected"] = bool(protected_ids.intersection(selected_ids))
        audit["valid"] = bool(audit["valid"] and not audit["top_deck_selected"])
        if not audit["valid"]:
            raise ValueError(_audit_error(audit))
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
        audit["valid"] = False
        audit["error"] = error
        audit["diagnostics"] = getattr(exc, "diagnostics", {})

    sections = ['<h1>Validation deck selection</h1>', '<p>' + html.escape(" | ".join(messages)) + '</p>']
    sections.append('<h2>Combined audit</h2>' + pd.DataFrame([audit]).to_html(index=False, escape=True))
    sections.append('<details><summary>Configuration</summary><pre>' + html.escape(yaml.safe_dump(cfg, allow_unicode=True)) + '</pre></details>')
    for name, frame in selections.items():
        sections.append('<h2>' + html.escape(name) + '</h2>' + frame.drop(columns=["card_ids"], errors="ignore").to_html(index=False, escape=True))
    if not selected_archetype_summary.empty:
        sections.append('<h2>Selected archetype summary</h2>' + selected_archetype_summary.to_html(index=False, escape=True))
    if "deck_isolation_candidates" in tables and "archetype_isolation_candidates" in tables:
        sections.extend(_charts(tables["deck_isolation_candidates"], selections.get(GROUPS[0], pd.DataFrame()), tables["archetype_isolation_candidates"]))
    for name, frame in tables.items():
        sections.append('<details><summary>' + html.escape(name) + '</summary>' + frame.to_html(index=False, escape=True) + '</details>')
    document = '<!doctype html><meta charset="utf-8"><title>Validation deck selection</title><style>body{font:14px sans-serif;margin:32px}table{border-collapse:collapse}td,th{padding:6px;white-space:nowrap}img{max-width:100%}details{margin:20px 0}</style>' + '\n'.join(sections)
    report_files = {"selection_report.html": document.encode("utf-8")}
    report_files.update({f"tables/{name}.csv": frame.to_csv(index=False).encode("utf-8-sig") for name, frame in tables.items()})
    report_files["tables/combined_audit.csv"] = pd.DataFrame([audit]).to_csv(index=False).encode("utf-8-sig")
    _publish(report_files, _path(cfg["outputs"]["reports"]))
    if error:
        raise ValueError(error + "; previous selection CSV files were preserved. See selection_report.html")
    _publish({f"{name}_selection.csv": selections[name].to_csv(index=False).encode("utf-8-sig") for name in GROUPS}, _path(cfg["outputs"]["selections"]))
    print("Combined audit passed. Saved selections and selection_report.html", flush=True)
    return audit


if __name__ == "__main__":
    run(load_settings())
