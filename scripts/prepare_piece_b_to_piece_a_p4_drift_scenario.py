"""Prepare the natural Piece B -> Piece A/P4 drift replay scenario.

The generated replay plan keeps a stable Piece B baseline, then injects the
Piece A/P4 rows represented by the strict Casting_class1:2_3 source pattern.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import shutil
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path


SCENARIO_ID = "production_replay_natural_piece_b_to_piece_a_p4_drift"
SCENARIO_TYPE = "production_drift"
MANIFEST_ID = "casting_flux_replay_plan_piece_b_to_piece_a_p4_drift_v001"
DATASET_VERSION = SCENARIO_ID
SCENARIO_VERSION = MANIFEST_ID
REPLAY_ID = SCENARIO_ID
PIECE_B_VIEW_PAIRS = "Casting_class1:1_2|Casting_class1:1_3|Casting_class1:2_3"
PIECE_A_P4_VIEW_PAIRS = "Casting_class1:2_3"
ROI_MODEL_VERSION = "roi_segmenter_v001_fixed"
FEATURE_AE_VERSION = "rd_feature_ae_gated_v001_bootstrap"
DEFAULT_WINDOW_SIZE = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--piece-b-plan",
        type=Path,
        default=Path("data/metadata/casting_flux_replay_plan_piece_b_full_v001.csv"),
    )
    parser.add_argument(
        "--p4-source-manifest",
        type=Path,
        default=Path("data/metadata/casting_piece_events.csv"),
    )
    parser.add_argument(
        "--output-plan",
        type=Path,
        default=Path("data/metadata/casting_flux_replay_plan_piece_b_to_piece_a_p4_drift_v001.csv"),
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=Path(".cache/iqa/visual_checks/piece_a_p4_drift_contact_sheet.html"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(".cache/iqa/source_datasets/hss-iad"),
    )
    parser.add_argument(
        "--materialize-from",
        type=Path,
        default=Path("data/raw/hss-iad"),
        help="Optional local source root used to materialize missing visual-check images into --image-root.",
    )
    parser.add_argument("--suspected-events", type=int, default=30)
    parser.add_argument("--correction-events", type=int, default=30)
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Observation window size used to align the stable baseline and P4 drift blocks.",
    )
    parser.add_argument(
        "--classification-selection-manifest",
        type=Path,
        default=Path("data/validation/classification_selection_piece_b_to_piece_a_p4_drift_v001.csv"),
    )
    parser.add_argument(
        "--critical-defects-per-window",
        type=int,
        default=6,
        help=(
            "Number of defective P4 replay events placed at the start of each "
            "critical drift window. With the default window size 30, 6 events "
            "give a deterministic red-rate degradation proxy without changing thresholds."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), list(reader)


def _stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _phase_for_drift_index(index: int, total: int, *, suspected_events: int, correction_events: int) -> str:
    if index < suspected_events:
        return "drift_piece_a_p4_suspected"
    correction_start = max(suspected_events, total - correction_events)
    if index >= correction_start:
        return "correction_replay"
    return "drift_piece_a_p4_confirmed"


def _split_pipe(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def _p4_source_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = [
        row
        for row in rows
        if row.get("source_class") == "Casting_class1"
        and row.get("view_pairs") == PIECE_A_P4_VIEW_PAIRS
    ]
    output.sort(key=lambda row: (row.get("source_timestamp") or "", row.get("event_id") or ""))
    return output


def _is_good_conforming_row(row: dict[str, str]) -> bool:
    label = str(row.get("label") or "").lower()
    is_defective = str(row.get("is_defective") or "").lower()
    oracle = str(row.get("oracle_verdict") or "").lower()
    return label != "defective" and is_defective != "true" and oracle not in {"defective", "non_conforme"}


def _balanced_piece_b_baseline_rows(rows: list[dict[str, str]], *, window_size: int) -> list[dict[str, str]]:
    """Keep a calm, complete-window Piece B baseline for the drift demo.

    The observation scenario should not start with expected defectives: those are
    useful for gate evaluation, but they make the drift dashboard noisy before
    Piece A/P4 appears. We also trim to a complete observation window so P4
    starts on a clean boundary.
    """

    if window_size <= 0:
        raise ValueError("--window-size must be > 0")
    buckets: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        if not _is_good_conforming_row(row):
            continue
        buckets.setdefault(row.get("view_pairs") or "", []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (row.get("source_timestamp") or "", row.get("sequence_number") or "", row.get("event_id") or ""))
    balanced: list[dict[str, str]] = []
    while any(buckets.values()):
        for view_pair in sorted(buckets):
            bucket = buckets[view_pair]
            if bucket:
                balanced.append(bucket.pop(0))
    complete_count = (len(balanced) // window_size) * window_size
    return balanced[:complete_count]


def _mask_path_for_source_image(row: dict[str, str], relative_path: str) -> str:
    if str(row.get("is_defective") or "").lower() != "true":
        return ""
    path = Path(relative_path.replace("\\", "/"))
    return f"Casting_class1/ground_truth/defective/{path.stem}_mask.png"


def _p4_source_rows_to_replay_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for source_row in source_rows:
        image_ids = _split_pipe(source_row.get("image_ids") or "")
        relative_paths = _split_pipe(source_row.get("relative_paths") or "")
        view_pairs = _split_pipe(source_row.get("view_pairs") or "")
        if not relative_paths:
            continue
        for view_index, relative_path in enumerate(relative_paths, start=1):
            view_pair = view_pairs[view_index - 1] if view_index <= len(view_pairs) else view_pairs[-1]
            image_id = image_ids[view_index - 1] if view_index <= len(image_ids) else ""
            output.append(
                {
                    "simulated_event_id": "",
                    "scenario_id": SCENARIO_ID,
                    "scenario_type": SCENARIO_TYPE,
                    "scenario_phase": "",
                    "is_representative": "False",
                    "sequence_number": "",
                    "scheduled_at": "",
                    "production_date": "",
                    "shift_id": "day_shift",
                    "station_id": "QC-CASTING-01",
                    "lot_id": "",
                    "sequence_in_lot": "",
                    "piece_serial": "",
                    "source_event_id": source_row.get("event_id", ""),
                    "source_event_key": (
                        f"{source_row.get('event_key', '')}#p4v{view_index}"
                        if len(relative_paths) > 1
                        else source_row.get("event_key", "")
                    ),
                    "source_class": "Casting_class1",
                    "source_group_key": source_row.get("group_key", ""),
                    "source_timestamp": source_row.get("source_timestamp", ""),
                    "label": source_row.get("label", ""),
                    "is_defective": source_row.get("is_defective", ""),
                    "expected_review_required": source_row.get("is_defective", ""),
                    "n_images": "1",
                    "source_classes": "Casting_class1",
                    "view_pairs": view_pair,
                    "image_ids": image_id,
                    "relative_paths": relative_path,
                    "event_time": "",
                    "recorded_at": "",
                    "is_simulated": "True",
                    "roi_model_version": ROI_MODEL_VERSION,
                    "feature_ae_version": FEATURE_AE_VERSION,
                    "raw_dataset_id": source_row.get("raw_dataset_id") or "hss_iad_casting_raw_v1",
                    "manifest_id": MANIFEST_ID,
                    "piece_event_id": source_row.get("event_id", ""),
                    "dataset_version": DATASET_VERSION,
                    "replay_id": REPLAY_ID,
                    "validation_id": "",
                    "scenario_version": SCENARIO_VERSION,
                    "gt_mask_paths": _mask_path_for_source_image(source_row, relative_path),
                }
            )
    return output


def _ordered_p4_rows_for_drift_detection(
    rows: list[dict[str, str]],
    *,
    suspected_events: int,
    critical_defects_per_window: int,
) -> list[dict[str, str]]:
    """Shape P4 rows so two complete windows carry the degradation signal.

    The stable Piece B baseline currently has 372 events, so with a window size
    of 30 the first drift window contains 12 Piece B events and 18 P4 events.
    Putting the repeated defective P4 rows at the start of the suspected and
    confirmed blocks keeps the signal in complete windows instead of the final
    partial tail.
    """

    defective_rows = [row for row in rows if str(row.get("is_defective") or "").lower() == "true"]
    good_rows = [row for row in rows if str(row.get("is_defective") or "").lower() != "true"]
    if not defective_rows:
        return rows
    critical_defect_count = critical_defects_per_window * 2
    repeated_defects = [
        dict(defective_rows[index % len(defective_rows)])
        for index in range(critical_defect_count)
    ]
    suspected_defects = repeated_defects[:critical_defects_per_window]
    confirmed_defects = repeated_defects[critical_defects_per_window:]
    suspected_good_count = max(0, suspected_events - len(suspected_defects))
    confirmed_good_count = suspected_good_count
    suspected_rows = [*suspected_defects, *good_rows[:suspected_good_count]]
    confirmed_rows = [
        *confirmed_defects,
        *good_rows[suspected_good_count : suspected_good_count + confirmed_good_count],
    ]
    correction_rows = good_rows[suspected_good_count + confirmed_good_count :]
    return [*suspected_rows, *confirmed_rows, *correction_rows]


def _rewrite_rows(
    rows: list[dict[str, str]],
    *,
    phase: str | None,
    scheduled_start: datetime,
    sequence_offset: int,
    suspected_events: int = 30,
    correction_events: int = 30,
) -> list[dict[str, str]]:
    rewritten = []
    total = len(rows)
    for local_index, row in enumerate(rows):
        output = dict(row)
        sequence_number = sequence_offset + local_index + 1
        scenario_phase = phase or _phase_for_drift_index(
            local_index,
            total,
            suspected_events=suspected_events,
            correction_events=correction_events,
        )
        scheduled_at = scheduled_start + timedelta(minutes=2 * (sequence_number - 1))
        output.update(
            {
                "simulated_event_id": _stable_id("sim_event", SCENARIO_ID, sequence_number, row.get("source_event_id")),
                "scenario_id": SCENARIO_ID,
                "scenario_type": SCENARIO_TYPE,
                "scenario_phase": scenario_phase,
                "sequence_number": str(sequence_number),
                "scheduled_at": scheduled_at.isoformat(timespec="seconds"),
                "production_date": scheduled_at.date().isoformat(),
                "lot_id": _lot_id_for_phase(scenario_phase, sequence_number),
                "piece_serial": _piece_serial_for_phase(scenario_phase, sequence_number, row),
                "event_time": scheduled_at.isoformat(timespec="seconds"),
                "recorded_at": scheduled_at.isoformat(timespec="seconds"),
                "is_simulated": "True",
                "manifest_id": MANIFEST_ID,
                "dataset_version": DATASET_VERSION,
                "replay_id": REPLAY_ID,
                "validation_id": "",
                "scenario_version": SCENARIO_VERSION,
            }
        )
        rewritten.append(output)
    return rewritten


def _lot_id_for_phase(phase: str, sequence_number: int) -> str:
    prefix = {
        "stable_baseline_piece_b": "IQA-PB-P4DRIFT-STABLE",
        "drift_piece_a_p4_suspected": "IQA-PA-P4DRIFT-SUSP",
        "drift_piece_a_p4_confirmed": "IQA-PA-P4DRIFT-CONF",
        "correction_replay": "IQA-PA-P4DRIFT-CORR",
    }.get(phase, "IQA-P4DRIFT")
    return f"{prefix}-L{((sequence_number - 1) // 50) + 1:03d}"


def _piece_serial_for_phase(phase: str, sequence_number: int, row: dict[str, str]) -> str:
    source = row.get("piece_serial") or row.get("source_group_key") or row.get("source_event_id") or str(sequence_number)
    suffix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    return f"{phase.upper().replace('-', '_')}-P{sequence_number:04d}-{suffix}"


def build_plan(args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    piece_b_fields, piece_b_rows = _read_csv(args.piece_b_plan)
    source_fields, source_rows = _read_csv(args.p4_source_manifest)
    fieldnames = list(
        OrderedDict.fromkeys([*piece_b_fields, *source_fields, "gt_mask_paths"])
    )

    p4_rows = _ordered_p4_rows_for_drift_detection(
        _p4_source_rows_to_replay_rows(_p4_source_rows(source_rows)),
        suspected_events=args.suspected_events,
        critical_defects_per_window=args.critical_defects_per_window,
    )
    stable_source_rows = _balanced_piece_b_baseline_rows(piece_b_rows, window_size=args.window_size)
    if not stable_source_rows:
        raise ValueError(f"No Piece B baseline rows found in {args.piece_b_plan}")
    if not p4_rows:
        raise ValueError(f"No Casting_class1 P4 rows found in {args.p4_source_manifest}")

    scheduled_start = datetime.fromisoformat("2026-06-27T08:00:00")
    stable_rows = _rewrite_rows(
        stable_source_rows,
        phase="stable_baseline_piece_b",
        scheduled_start=scheduled_start,
        sequence_offset=0,
    )
    drift_start = scheduled_start + timedelta(minutes=2 * len(stable_rows))
    drift_rows_out = _rewrite_rows(
        p4_rows,
        phase=None,
        scheduled_start=drift_start,
        sequence_offset=len(stable_rows),
        suspected_events=args.suspected_events,
        correction_events=args.correction_events,
    )
    return fieldnames, [*stable_rows, *drift_rows_out]


def write_classification_selection_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    selection_rows = [
        row
        for row in rows
        if row.get("scenario_phase") in {"drift_piece_a_p4_suspected", "drift_piece_a_p4_confirmed"}
    ]
    if not selection_rows:
        raise ValueError("classification selection manifest would be empty")
    output_fields = list(OrderedDict.fromkeys([*fieldnames, "validation_role"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        for row in selection_rows:
            output = dict(row)
            output["validation_role"] = "classification_selection_piece_a_p4"
            output["validation_id"] = path.stem
            writer.writerow(output)


def write_plan(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def materialize_visual_images(rows: list[dict[str, str]], *, image_root: Path, source_root: Path) -> dict[str, int]:
    copied = 0
    missing = 0
    if not source_root.is_dir():
        return {"copied": copied, "missing": missing}
    for row in rows:
        if row.get("source_class") != "Casting_class1" or row.get("scenario_phase") == "stable_baseline_piece_b":
            continue
        for relative_path in [part for part in (row.get("relative_paths") or "").split("|") if part]:
            rel = Path(relative_path.replace("\\", "/"))
            destination = image_root / rel
            if destination.is_file():
                continue
            source = source_root / rel
            if not source.is_file():
                missing += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    return {"copied": copied, "missing": missing}


def write_contact_sheet(path: Path, image_root: Path, rows: list[dict[str, str]], *, max_pieces: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        if row.get("source_class") != "Casting_class1" or row.get("scenario_phase") == "stable_baseline_piece_b":
            continue
        key = row.get("source_group_key") or row.get("piece_event_id") or row.get("source_event_id") or ""
        grouped.setdefault(key, []).append(row)
    pieces = list(grouped.items())[:max_pieces]
    image_root_rel = Path("../source_datasets/hss-iad")
    lines = [
        '<!doctype html><html><head><meta charset="utf-8">',
        "<title>Piece A P4 drift visual check</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:18px;background:#f6f7f9;color:#111}",
        "h1{font-size:22px;margin:0 0 8px}",
        ".summary{margin:0 0 16px;padding:10px;background:#fff;border:1px solid #ddd}",
        "table{border-collapse:collapse;width:100%;background:#fff}",
        "th,td{border:1px solid #ddd;vertical-align:top;padding:6px}",
        "th{position:sticky;top:0;background:#eef2f6;z-index:1}",
        "td.meta{font-size:12px;width:230px}",
        "td img{width:220px;max-width:100%;height:auto;display:block;background:#ddd}",
        ".cap{font-size:11px;line-height:1.25;margin-top:4px}",
        ".cap span{color:#555}",
        ".defective{background:#fff0f0}",
        ".good{background:#f0fff4}",
        ".missing{background:#ffe8a3;font-weight:bold;color:#7a4b00}",
        "</style></head><body>",
        "<h1>Piece A / P4 - verification visuelle drift</h1>",
        (
            '<div class="summary">'
            f"Scenario: {SCENARIO_ID}<br>"
            f"Pieces affichees: {len(pieces)} | Source: Casting_class1 hors triplet Piece B<br>"
            "P4/piece A n'est pas force defective: les labels source sont conserves."
            "</div>"
        ),
        "<table><thead><tr><th>Piece event</th><th>Images P4 / Piece A</th></tr></thead><tbody>",
    ]
    for index, (group_key, group_rows) in enumerate(pieces, start=1):
        label = group_rows[0].get("label") or "unknown"
        css_class = "defective" if label == "defective" else "good"
        lines.append(
            f'<tr class="{css_class}"><td class="meta">#{index}<br>'
            f"{html.escape(group_key)}<br>"
            f"label={html.escape(label)}<br>"
            f"phase={html.escape(group_rows[0].get('scenario_phase') or '')}</td><td>"
        )
        for row in group_rows:
            for relative_path in [part for part in (row.get("relative_paths") or "").split("|") if part]:
                rel = Path(relative_path.replace("\\", "/"))
                image_path = image_root / rel
                src = (image_root_rel / rel).as_posix()
                caption = html.escape(relative_path)
                if image_path.exists():
                    lines.append(f'<img src="{src}" loading="lazy"><div class="cap"><span>{caption}</span></div>')
                else:
                    lines.append(f'<div class="missing">missing<br><span>{caption}</span></div>')
        lines.append("</td></tr>")
    lines.extend(["</tbody></table></body></html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    fieldnames, rows = build_plan(args)
    write_plan(args.output_plan, fieldnames, rows)
    write_classification_selection_manifest(args.classification_selection_manifest, fieldnames, rows)
    materialized = materialize_visual_images(rows, image_root=args.image_root, source_root=args.materialize_from)
    write_contact_sheet(args.contact_sheet, args.image_root, rows)
    phase_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        phase_counts[row["scenario_phase"]] = phase_counts.get(row["scenario_phase"], 0) + 1
        if row.get("source_class") == "Casting_class1" and row.get("scenario_phase") != "stable_baseline_piece_b":
            key = f"{row.get('label')}:{row.get('is_defective')}"
            label_counts[key] = label_counts.get(key, 0) + 1
    print(
        {
            "output_plan": str(args.output_plan),
            "classification_selection_manifest": str(args.classification_selection_manifest),
            "contact_sheet": str(args.contact_sheet),
            "materialized_visual_images": materialized,
            "rows": len(rows),
            "phase_counts": phase_counts,
            "p4_label_counts": label_counts,
        }
    )


if __name__ == "__main__":
    main()
