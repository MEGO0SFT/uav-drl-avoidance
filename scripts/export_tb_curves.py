"""Export TensorBoard tfevents scalars to plain CSV.

The utility reads locally available ``events.out.tfevents.*`` files and writes
one CSV per run.

The tfevents format is parsed by hand (TFRecord framing + protobuf wire format)
so that neither ``tensorboard`` nor ``protobuf`` is required.

Usage
-----
    python scripts/export_tb_curves.py \
        RL_scripts/body-axis_v2/logs \
        --out tmp/tb_curves

Only runs listed in ``RUN_LABELS`` are exported. Add your own entry to export a
new run.
"""
from __future__ import annotations

import argparse
import csv
import os
import struct
import sys

# run directory -> (output label, description). The label equals the directory name, so a CSV is
# traceable back to its TensorBoard run at a glance.
RUN_LABELS = {
    "2026-03-18_paper-curriculum-6M": (
        "2026-03-18_paper-curriculum-6M",
        "PAPER - curriculum group - 6.0 M steps - ent 0.05->0.005"),
    "2026-04-13_paper-control-6M": (
        "2026-04-13_paper-control-6M",
        "PAPER - no-curriculum control group - 6.0 M steps - ent 0.05->0.005"),
    "2026-04-27_paper-sac-6M": (
        "2026-04-27_paper-sac-6M",
        "PAPER - off-policy control - 6.1 M steps - ent auto"),
    "2026-03-11_pilot-5M": (
        "2026-03-11_pilot-5M",
        "pilot - 5.12 M steps - ent 0.03->0 - in-training success peaked at 0.89"),
    "2026-02-07_pilot-5M": (
        "2026-02-07_pilot-5M",
        "pilot - 5.12 M steps - ent 0.03->0 - in-training success peaked at 0.73"),
    "2025-12-23_pilot-4M": (
        "2025-12-23_pilot-4M",
        "pilot - earliest full run - 4.05 M steps - ent 0.03->0.001"),
}

TAGS = (
    "train/success_rate",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/approx_kl",
    "train/value_loss",
    "train/std",
    "train/ent_coef",
)


# ── minimal tfevents reader ──────────────────────────────────────────────────
def _varint(buf, i):
    out, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _fields(buf):
    out, i, n = [], 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        out.append((fno, wt, v))
    return out


def _records(path):
    with open(path, "rb") as fh:
        data = fh.read()
    pos, n = 0, len(data)
    while pos + 12 <= n:
        ln = struct.unpack("<Q", data[pos:pos + 8])[0]
        pos += 12
        if pos + ln + 4 > n:
            return
        yield data[pos:pos + ln]
        pos += ln + 4


def iter_scalars(path):
    """Yield ``(step, tag, value)`` for every scalar in a tfevents file."""
    for payload in _records(path):
        try:
            ev = _fields(payload)
        except Exception:
            continue
        step, summary = 0, None
        for fno, wt, v in ev:
            if fno == 2:
                step = v
            elif fno == 5:
                summary = v
        if summary is None:
            continue
        try:
            values = _fields(summary)
        except Exception:
            continue
        for fno, wt, v in values:
            if fno != 1:
                continue
            tag = val = None
            try:
                vf = _fields(v)
            except Exception:
                continue
            for f2, w2, v2 in vf:
                if f2 == 1:
                    tag = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 5:
                    val = struct.unpack("<f", v2)[0]
                elif f2 == 8:  # tensor-based scalar summary
                    try:
                        for f3, w3, v3 in _fields(v2):
                            if f3 == 5:
                                val = struct.unpack("<f", (v3[:4] if w3 == 2 else v3))[0]
                    except Exception:
                        pass
            if tag is not None and val is not None:
                yield step, tag, val


def find_event_file(run_dir):
    for name in sorted(os.listdir(run_dir)):
        if name.startswith("events.out.tfevents"):
            return os.path.join(run_dir, name)
    return None


def export(logs_root, out_dir, tags=TAGS, max_rows=5000):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for run, (label, desc) in RUN_LABELS.items():
        run_dir = os.path.join(logs_root, run)
        if not os.path.isdir(run_dir):
            print(f"  skip {run} (absent)", file=sys.stderr)
            continue
        event_file = find_event_file(run_dir)
        if event_file is None:
            print(f"  skip {run} (no tfevents)", file=sys.stderr)
            continue

        series = {t: [] for t in tags}
        for step, tag, val in iter_scalars(event_file):
            if tag in series:
                series[tag].append((step, val))

        # one CSV per run, one column per tag, joined on the logged step index
        steps = sorted({s for pts in series.values() for s, _ in pts})
        # Decimate: off-policy runs log every gradient step (~280 k points for
        # SAC), which is pointless detail for a curve and bloats the repository.
        if len(steps) > max_rows:
            stride = len(steps) // max_rows + 1
            kept = steps[::stride]
            if steps[-1] != kept[-1]:
                kept.append(steps[-1])
            steps = kept
        lookup = {t: dict(pts) for t, pts in series.items()}
        path = os.path.join(out_dir, f"{label}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["# run", label])
            w.writerow(["# description", desc])
            w.writerow(["step", *tags])
            for s in steps:
                w.writerow([s, *[lookup[t].get(s, "") for t in tags]])
        n = sum(len(v) for v in series.values())
        print(f"  {label:<18} {n:7d} points -> {len(steps):5d} rows -> {path}")
        written.append(path)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs_root", help="directory containing PPO_*/SAC_* run folders")
    ap.add_argument("--out", default="tmp/tb_curves", help="local-only output directory")
    args = ap.parse_args(argv)
    print(f"exporting from {args.logs_root}")
    files = export(args.logs_root, args.out)
    print(f"done: {len(files)} CSV file(s) in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
