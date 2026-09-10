"""Per-run output directories.

Every execution writes into its own timestamped folder, so results from
different runs never overwrite each other and a figure can always be traced back
to the run that produced it:

    results/
      runs/
        20260809-231455_analysis/
          figures/          plots
          tables/           CSVs
          manifest.json     what produced this, and from what
        20260809-234012_analysis/
          ...
      cache/                expensive, reusable across runs
      fingerprint_corpus.csv

WHAT DOES *NOT* GO IN A RUN FOLDER
----------------------------------
Two kinds of file are deliberately shared rather than copied per run:

  * `fingerprint_corpus.csv` -- an accumulating input, built over hours and
    resumable. It is data the runs consume, not output they produce.
  * `cache/` -- expensive intermediates such as the multi-seed attack-impact
    grid, which costs 18 federated runs. Copying it per run would mean
    recomputing it per run.

Each run's manifest records the corpus row count and content hash it used, so a
run is still reproducible without duplicating gigabytes.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

RESULTS_ROOT = Path("results")
RUNS_ROOT = RESULTS_ROOT / "runs"
CACHE_ROOT = RESULTS_ROOT / "cache"


def new_run_dir(label: str = "run", root: Path | str = RUNS_ROOT) -> Path:
    """Create and return `results/runs/<timestamp>_<label>/`.

    The timestamp leads so that a plain alphabetical listing is also
    chronological -- no sorting by mtime, which breaks the moment files are
    copied or synced.
    """
    root = Path(root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{_slug(label)}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run(label: str, df, root: Path | str = RUNS_ROOT, **manifest_fields) -> Path:
    """Save a sweep result under a NAMED, timestamped run, and as the latest.

    Writes two things:

      results/runs/<timestamp>_<label>/<label>.csv   the permanent record
      results/runs/<timestamp>_<label>/manifest.json what produced it
      results/<label>.csv                            the latest, for the notebook

    Both are needed and they do different jobs. The timestamped copy means a
    result is never silently overwritten by a later run, so a number quoted in
    the write-up can always be traced to the exact run that produced it. The
    flat copy gives the notebook a stable path to load, so it does not have to
    guess which run is current.

    The flat file is a POINTER, not an archive: it is overwritten every run by
    design. Anything cited should cite the timestamped folder.
    """
    import pandas as pd  # noqa: F401  (df is already a DataFrame; import kept local)

    run_dir = new_run_dir(label, root=root)
    csv_path = run_dir / f"{label}.csv"
    df.to_csv(csv_path, index=False)

    write_manifest(run_dir, label=label, rows=len(df),
                   columns=list(df.columns),
                   digest=file_digest(csv_path), **manifest_fields)

    flat = Path(root).parent / f"{label}.csv"
    flat.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flat, index=False)

    print(f"[run] {label}: {len(df)} rows -> {run_dir}")
    print(f"[run] latest copy -> {flat}")
    return run_dir


def cache_dir() -> Path:
    """Shared directory for expensive intermediates reused across runs."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return CACHE_ROOT


# --------------------------------------------------------------------------- #
# The sweep row schema
# --------------------------------------------------------------------------- #
# THE GOVERNING RULE: every row carries the COMPLETE experimental state that
# produced it, not just the parameter its sweep happened to vary.
#
# The alternative -- recording only the varied factor and putting the rest in a
# manifest -- is what the earlier result files did, and it fails three ways:
#
#   * a row is meaningless outside its folder. `0,fedavg,control,0.0,0.8305`
#     does not say how many rounds, epochs or groups produced it.
#   * sweeps cannot be pooled, because each writes different columns.
#   * a guard can only check what the file contains, so a file produced at
#     rounds=8 is silently accepted as rounds=6.
#
# HANDOFF.md records four separate incidents in this project of stale results
# being pooled with fresh ones. A campaign that varies eight parameters
# multiplies that risk by eight, so the schema is fixed here and every sweep
# writes into it. Missing measurements are NaN; columns never disappear.

PROVENANCE_COLUMNS = [
    "run_id", "sweep", "factor", "git_commit", "git_dirty", "code_digest",
    "timestamp", "secs", "device", "torch_version",
]

# The experimental state. Two rows with identical values here are the same
# experiment, which is what makes control pairing a groupby rather than a
# hand-written lookup.
CONFIG_COLUMNS = [
    "dataset", "arch", "partition", "num_clients", "n_groups", "concentration",
    "overlap", "max_train", "warmup", "rounds", "epochs", "n_attackers",
    "attacker_ids", "mechanism", "beta", "payload", "payload_scale",
    "aggregator", "scope", "recursive", "max_clusters", "eps1", "eps2", "seed",
    # Which of the other groups the attacker aims at, as a RANK (0..K-2) over the
    # other groups in sorted order, not as a group id. The id is seed-dependent,
    # because the attacker's home is wherever client 0 landed and group
    # assignment is shuffled per seed, so an id cannot be enumerated when the
    # cell list is built. Rank can. The resolved id is recorded as a measurement.
    # Rank 0 is what every experiment before 10 did, so those rows backfill to 0.
    "target_rank",
]

CONDITION_COLUMNS = ["condition", "is_control"]

MEASUREMENT_COLUMNS = [
    "infiltration", "post_split_rounds", "split_round", "ari",
    "target_resemblance", "to_target", "to_home", "honest_within",
    "honest_cross", "final_accuracy", "victim_accuracy",
    # Which group `target_rank` actually resolved to this seed, so a row says
    # what it aimed at without needing the bundle rebuilt to find out.
    "target_group",
    # The two predictors experiment 10 tests against each other.
    #   target_spread   Jensen-Shannon divergence from the target's mean profile
    #                   to uniform. LOW means diffuse, and the hypothesis is that
    #                   low is EASIER for the attacker, so this one runs opposite
    #                   to every other "low is better" column here.
    #   home_to_target  Jensen-Shannon divergence from the home profile to the
    #                   target profile. The confound: a target that resembles the
    #                   attacker's own group is easier for reasons that have
    #                   nothing to do with how diffuse it is.
    "target_spread", "home_to_target",
    # The architecture channel. `declared_arch` is the field the attacker
    # falsifies and `true_arch` is what it really runs; they differ only for a
    # spoofer. `shape_rejected_rounds` counts how often the free weight-shape
    # check stopped that client, which must be recorded separately from any
    # fingerprint result: pooling the two credits a defence with the shape
    # check's work.
    "declared_arch", "true_arch", "shape_rejected_rounds",
]

SWEEP_COLUMNS = (PROVENANCE_COLUMNS + CONFIG_COLUMNS
                 + CONDITION_COLUMNS + MEASUREMENT_COLUMNS)

# Columns that identify one experimental cell, ignoring the condition. A
# treatment row and its control differ ONLY in the condition block, so grouping
# on these pairs them automatically -- and a group with no control row is a
# treatment measured against nothing, which is the failure worth catching.
#
# The exclusions are not cosmetic:
#   beta, payload   these ARE the condition. The control is beta=0 with no
#                   payload, so leaving them in put every control and its own
#                   treatment into different groups and made the check report
#                   that every cell lacked a control.
#   eps1, eps2      tuned per seed from the warmup, so they differ between two
#                   runs of the same cell without the cell being different.
_CONDITION_DEFINING = ("beta", "payload", "eps1", "eps2")
CELL_KEY = [c for c in CONFIG_COLUMNS if c not in _CONDITION_DEFINING]


def blank_row(**values) -> dict:
    """A schema-complete row: every column present, unset ones None.

    Building rows through this rather than as ad-hoc dicts is what keeps the
    sweeps concatenable. A sweep that forgets a column gets None rather than a
    ragged frame that only fails later, at concat time, far from the cause.
    """
    unknown = set(values) - set(SWEEP_COLUMNS)
    if unknown:
        raise KeyError(f"not in the sweep schema: {sorted(unknown)}. "
                       f"Add it to run_context.SWEEP_COLUMNS deliberately, "
                       f"rather than letting one sweep grow its own column.")
    row = {c: None for c in SWEEP_COLUMNS}
    row.update(values)
    return row


def git_state(repo: Path | str = ".") -> dict:
    """Current commit and whether the tree is dirty.

    Recorded per row because the code changes mid-campaign: the three K>2 fixes
    land between sweeps, and without this there is no way to tell afterwards
    which rows were produced before them.
    """
    import subprocess

    def _run(*args):
        try:
            out = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run("rev-parse", "--short", "HEAD")
    status = _run("status", "--porcelain")
    return {"git_commit": commit,
            "git_dirty": None if status is None else bool(status.strip())}


def code_digest(files=("attacks.py", "clustering.py", "fl_loop.py",
                       "data.py", "aggregation.py", "lab_data.py",
                       "sweeps.py", "analysis.py")) -> str:
    """Hash of the modules that actually determine an experiment's behaviour.

    Complements `git_commit`: the tree is dirty for most of this project's
    life, so the commit alone does not pin the code. This does.

    `sweeps.py` is in the list because it is the driver, and leaving it out was
    a real gap: when the attacker-selection bug in `sweeps.run_cell` was fixed
    mid-campaign, the digest did not change, so rows produced before and after
    the fix were indistinguishable by it. Anything that decides what an
    experiment DOES belongs here, not only the libraries it calls.
    """
    h = hashlib.sha256()
    for name in sorted(files):
        p = Path(name)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# experiments/ -- one folder per experiment, everything it produced inside it
# --------------------------------------------------------------------------- #
# The older layout scattered a single experiment across four places: its rows in
# results/runs/<timestamp>_<label>/, a flat copy at results/<label>.csv, a
# pooled copy in results/, and its figures in docs/figures/. Finding everything
# one experiment produced meant knowing all four conventions.
#
#     experiments/
#       README.md              index of every experiment
#       01_structure/
#         README.md            what it asked, what it found
#         data/structure.csv
#         graphs/*.png
#         manifest.json        provenance and the grid that was swept
#         log.txt              per-cell trace, including [sattler] tuning lines
#
# The timestamped folders existed so re-runs could not overwrite each other.
# That job now belongs to the DATA: every row carries `run_id`, `git_commit`,
# `code_digest` and `timestamp`, so two runs remain distinguishable after they
# land in the same file. Row-level provenance is what makes the simpler folder
# layout safe, and it is strictly more useful -- a row keeps its provenance when
# it is copied out, which a folder name does not.

EXPERIMENTS_ROOT = Path("experiments")


def experiment_dir(name: str, root: Path | str = EXPERIMENTS_ROOT) -> Path:
    """`experiments/<name>/` with its `data/` and `graphs/` subfolders."""
    d = Path(root) / _slug(name)
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "graphs").mkdir(parents=True, exist_ok=True)
    return d


def experiment_csv(name: str, root: Path | str = EXPERIMENTS_ROOT) -> Path:
    """Canonical data path: `experiments/<name>/data/results.csv`.

    Fixed filename rather than one derived from the experiment, because the
    folder name already says what the experiment was. Two conventions naming the
    same thing is one more than needed, and it is the folder name that a person
    reads when scanning the directory.
    """
    return Path(root) / _slug(name) / "data" / "results.csv"


def save_experiment(name: str, df, log_text: str = "", **manifest_fields) -> Path:
    """Write an experiment's rows, manifest and log into its own folder."""
    d = experiment_dir(name)
    csv_path = experiment_csv(name)
    df.to_csv(csv_path, index=False)

    write_manifest(d, label=name, rows=len(df), columns=list(df.columns),
                   digest=file_digest(csv_path), **manifest_fields)
    if log_text:
        (d / "log.txt").write_text(log_text, encoding="utf-8")

    print(f"[experiment] {name}: {len(df)} rows -> {csv_path}")
    return d


class GuardError(AssertionError):
    """A results file did not match what the caller expected of it."""


def load_experiment(name: str, expect: Optional[dict] = None,
                    require_control: bool = True,
                    root: Path | str = EXPERIMENTS_ROOT):
    """Guarded load of `experiments/<name>/data/results.csv`. See `load_run`.

    `01_structure_envelope` does no training and carries no condition columns, so
    it needs `require_control=False`.
    """
    return load_run(experiment_csv(name, root), expect=expect,
                    require_control=require_control, path_is_full=True)


def load_run(label: str, expect: Optional[dict] = None,
             require_control: bool = True, root: Path | str = RESULTS_ROOT,
             path_is_full: bool = False):
    """Load a results CSV and REFUSE it unless it is what was asked for.

    Replaces the hand-written per-cell guards. Those check the condition set and
    the seed count, which catches a wrong treatment but happily accepts a file
    produced at a different number of rounds, because the old files did not
    record the number of rounds. This checks the configuration itself.

    `expect` pins configuration columns, e.g. `dict(n_groups=2, rounds=6)`.
    Every row must match every pinned value, and any pinned column that VARIES
    is refused too -- a file containing two settings is not the file the caller
    thinks it is, whichever setting it wanted.

    `require_control` demands that every experimental cell has a control row.
    A treatment whose control never ran produces a number with nothing to read
    it against, which is the shape of this project's oldest mistake. It fails
    closed: a file with no `is_control` column, or with none of the cell-key
    columns, is refused rather than quietly skipping the check.

    Raises GuardError naming the specific mismatch, rather than returning
    something plausible.
    """
    import pandas as pd

    path = Path(label) if path_is_full else Path(root) / f"{label}.csv"
    if not path.exists():
        raise GuardError(f"{path} not present. Run `python sweeps.py <name>` to produce it.")

    df = pd.read_csv(path)
    problems = []

    for key, want in (expect or {}).items():
        if key not in df.columns:
            problems.append(f"{key!r} pinned but not a column in the file")
            continue
        found = set(df[key].dropna().unique().tolist())
        if len(found) > 1:
            problems.append(f"{key!r} varies across the file ({sorted(found)}), "
                            f"but was pinned to {want!r}")
        elif found and next(iter(found)) != want:
            problems.append(f"{key!r} is {next(iter(found))!r}, expected {want!r}")

    if require_control:
        # Fails CLOSED. The earlier version ran this check only `if "is_control"
        # in df.columns`, so a file without the column sailed through as though
        # it had passed, which is the one place this project's guards did not
        # behave like guards. A file that cannot be checked is refused, and a
        # caller that genuinely has no conditions says so with
        # `require_control=False` rather than being let through silently.
        if "is_control" not in df.columns:
            problems.append(
                "'is_control' is not a column, so control pairing cannot be "
                "checked. Pass require_control=False if this file genuinely has "
                "no treatment/control structure, e.g. an untrained sweep")
        else:
            keys = [c for c in CELL_KEY if c in df.columns]
            if not keys:
                problems.append(
                    f"none of the cell-key columns {CELL_KEY} are present, so "
                    f"treatment rows cannot be paired with their controls")
            else:
                has_ctrl = df.groupby(keys, dropna=False)["is_control"].any()
                missing = int((~has_ctrl).sum())
                if missing:
                    problems.append(f"{missing} experimental cell(s) have no control row")

    if problems:
        raise GuardError(f"{path} refused:\n  - " + "\n  - ".join(problems))
    return df


def build_campaign(root: Path | str = EXPERIMENTS_ROOT, out: str = "campaign.csv"):
    """Concatenate every schema-conforming experiment into one master table.

    DERIVED, never hand-edited, so it cannot drift from its sources. This is the
    single artefact that answers "show me all the evidence": one long table where
    every row is self-describing, so any cut can be taken with a groupby and no
    row depends on the folder it came from.
    """
    import pandas as pd

    root = Path(root)
    frames, skipped, ignored, other_channel = [], [], [], []
    # ONLY `results.csv`. `experiment_csv` fixes that name as the canonical one, and
    # globbing `*.csv` pooled anything else a folder happened to contain: on 31 Aug
    # 2026 that would have swept 72 rows from `m3_aggregators/retracted_*.csv`, two
    # runs kept ON PURPOSE as evidence of a correction and marked do-not-cite, into
    # the master table alongside valid data. They are schema-conforming, so nothing
    # downstream would have objected. Auxiliary files are listed rather than silently
    # dropped, so a genuine second data file cannot disappear here unnoticed.
    for p in sorted(root.glob("*/data/*.csv")):
        if p.name == out:
            continue
        if p.name != "results.csv":
            ignored.append(f"{p.parent.parent.name}/{p.name}")
            continue
        # PART A ONLY, by folder-naming convention. The distribution campaign uses
        # numeric prefixes (`01_`..`10_`); the architecture channel uses `m1_`/`m3_`
        # precisely so the two cannot be confused. They are NOT pooled, for a
        # measurement reason rather than a tidiness one: architecture rows carry
        # NaN for `ari`, `separation` and `target_resemblance`, because
        # cross-architecture cosine is undefined. A groupby over a mixed table
        # would average incommensurable rows and report a number for it. The
        # architecture channel is analysed from its own folders and its own
        # documents; see ARCHITECTURE_CHANNEL.md.
        if not p.parent.parent.name[0].isdigit():
            other_channel.append(f"{p.parent.parent.name}/{p.name}")
            continue
        d = pd.read_csv(p)
        if set(SWEEP_COLUMNS).issubset(d.columns):
            frames.append(d[SWEEP_COLUMNS])
        else:
            skipped.append((f"{p.parent.parent.name}/{p.name}",
                            sorted(set(SWEEP_COLUMNS) - set(d.columns))[:4]))

    if not frames:
        print("[campaign] no schema-conforming experiments found")
        return None

    master = pd.concat(frames, ignore_index=True)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / out
    master.to_csv(dest, index=False)
    print(f"[campaign] {len(master)} rows from {len(frames)} experiments -> {dest}")
    if ignored:
        print(f"[campaign] ignored {len(ignored)} non-canonical file(s) in data/: "
              f"{', '.join(ignored)}")
    if other_channel:
        print(f"[campaign] excluded {len(other_channel)} architecture-channel "
              f"experiment(s), which are not pooled by design: "
              f"{', '.join(other_channel)}")
    for name, missing in skipped:
        print(f"[campaign] skipped {name} (not schema-conforming): missing {missing}")
    return master


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(text).strip().lower()]
    return "".join(keep).strip("-") or "run"


def file_digest(path: Path | str, chunk: int = 1 << 20) -> Optional[str]:
    """Short content hash, so a manifest can pin which corpus a run consumed."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def write_manifest(run_dir: Path | str, **fields) -> Path:
    """Record what produced this run. Written last, so its presence also marks
    the run as having completed rather than died halfway."""
    run_dir = Path(run_dir)
    manifest = {
        "run": run_dir.name,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        **fields,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def latest_run(root: Path | str = RUNS_ROOT, completed_only: bool = True) -> Optional[Path]:
    """Most recent run directory, or None. `completed_only` skips runs with no
    manifest -- i.e. ones that crashed before finishing."""
    root = Path(root)
    if not root.exists():
        return None
    runs = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
    for d in runs:
        if not completed_only or (d / "manifest.json").exists():
            return d
    return None


# --------------------------------------------------------------------------- #
# Shared notebook helpers
# --------------------------------------------------------------------------- #
# These lived in cell 1 of arch-spoofing-analysis.ipynb. They are here so the
# distribution notebook uses the same bootstrap, the same validated palette and
# the same tolerant corpus reader, rather than a second copy that quietly drifts
# out of step with the first.

# Categorical palette: blue / orange / aqua / violet. This is the four-colour set
# that passes the all-pairs colour-vision checks on a light surface (worst CVD
# dE 9.2, worst normal-vision dE 16.3). The conventional fourth slot, yellow,
# fails beside orange at 13.7, which matters wherever a scatter puts every pair
# of series next to each other.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D"]     # secondary encoding, so identity is never colour alone
INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def use_report_style():
    """Matplotlib defaults for figures headed into a printed LNCS report.

    Commits to a light surface and paints it explicitly rather than inheriting
    the Jupyter theme; a dark-themed notebook otherwise exports figures with
    invisible axes.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 150,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": SECOND, "ytick.labelcolor": SECOND,
        "font.family": "sans-serif", "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6,
    })
    return mpl


def boot_ci(values, n_boot: int = 2000, alpha: float = 0.05, rng=None):
    """Percentile bootstrap confidence interval, returning (mean, lo, hi).

    Used instead of mean +/- std because seed counts here are small and the
    statistics are not remotely normal.
    """
    import numpy as np

    rng = rng or np.random.default_rng(0)
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (float("nan"),) * 3
    if len(v) == 1:
        return (float(v[0]),) * 3
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def read_corpus(path, bool_cols=(), required=()):
    """Read a corpus CSV that a generator may still be appending to.

    `on_bad_lines="skip"` because the corpus is designed to be analysed while
    generation is running; a read landing mid-write sees a truncated final line,
    and skipping it costs one row out of hundreds. A partial line can also parse
    as a structurally valid row with missing fields, which skipping does not
    catch, so `required` columns are dropped on NaN as well.
    """
    import pandas as pd

    df = pd.read_csv(path, on_bad_lines="skip")
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1"])
    if required:
        before = len(df)
        df = df.dropna(subset=[c for c in required if c in df.columns])
        if len(df) < before:
            print(f"[corpus] dropped {before - len(df)} incomplete row(s) "
                  f"(generator is probably still running)")
    return df


def parse_vec(s):
    """';'-joined float string -> array. Corpora store variable-width vectors as
    text so one CSV can hold datasets with 7, 10 and 62 classes."""
    import numpy as np

    if not isinstance(s, str) or not s:
        return np.array([])
    return np.array([float(x) for x in s.split(";")])


def show(df, title=None, save_as=None, tabdir=None):
    """Print a table beside every figure, optionally saving it.

    Three of the palette slots sit below 3:1 contrast on white, and the colour
    guidance requires relief (a visible table or direct labels) wherever that
    happens, so every figure gets its numbers printed alongside.
    """
    import pandas as pd

    if title:
        print(f"\n{title}")

    # Print the index when it CARRIES INFORMATION, i.e. anything other than the
    # default RangeIndex. Unconditionally passing index=False silently strips
    # row labels off pivot tables: an aggregator comparison printed six rows of
    # numbers with no way to tell which aggregator each belonged to, which is
    # worse than useless in a results notebook because it still looks like a
    # table. A named or non-integer index is a label, not decoration.
    keep_index = not isinstance(df.index, pd.RangeIndex)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=keep_index))
    if save_as and tabdir is not None:
        Path(tabdir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(tabdir) / f"{save_as}.csv", index=keep_index)


# --------------------------------------------------------------------------- #
# Analysis gates
# --------------------------------------------------------------------------- #
# Each of these exists because this project once published a plausible-looking
# wrong number (HANDOFF section 6). They are the only sanctioned route to the
# thing they compute; inlining the calculation elsewhere is how the wrong number
# came back the previous three times.

def assert_grid_matches(df, grid: dict, expected_rows: int | None = None):
    """Refuse a corpus holding anything outside the declared grid.

    Four separate incidents came from a corpus that silently accumulated rows
    from different configurations: two generators appending at once, a key
    missing a column, config edits that appeared not to take effect, and 36
    scratch rows from verification runs pooled into every aggregate.

    `grid` maps a column name to the values the config cell declares. Raises
    rather than warns: an analysis that runs on a contaminated corpus still
    produces numbers, and numbers get copied into reports.
    """
    problems = []
    for column, allowed in grid.items():
        if column not in df.columns:
            problems.append(f"corpus has no column '{column}'")
            continue
        allowed_set = {str(v) for v in allowed}
        extra = {str(v) for v in df[column].unique()} - allowed_set
        if extra:
            problems.append(
                f"column '{column}' holds {sorted(extra)}, which the config does "
                f"not declare (declared: {sorted(allowed_set)})")

    if expected_rows is not None and len(df) != expected_rows:
        problems.append(
            f"corpus has {len(df)} rows, the grid implies {expected_rows}: "
            f"generation is incomplete, or the file holds rows from another grid")

    if problems:
        raise ValueError("corpus does not match the declared grid, refusing to "
                         "analyse:\n  " + "\n  ".join(problems) +
                         "\n\nDelete the corpus and regenerate, or fix the config.")
    return True


def assert_control_present(grid: dict, dial: str, control_value=0.0):
    """Refuse a grid with no control condition on the attack's strength dial.

    The last completed distribution run has no `lam=0` control, because QUICK
    mode collapsed LAMBDAS to a single value. Without it, "infiltration is
    caused by the imitation" is an unsupported claim, and it was recorded as a
    finding anyway. A dial with no zero is not an experiment.
    """
    values = [float(v) for v in grid.get(dial, [])]
    if not values:
        raise ValueError(f"grid declares no values for the dial '{dial}'")
    if not any(abs(v - control_value) < 1e-12 for v in values):
        raise ValueError(
            f"grid for '{dial}' is {values} and contains no control at "
            f"{control_value}. Every claim that the attack CAUSES an effect "
            f"needs it. Add {control_value} before generating.")
    return True


def detection_table(df, flagged_col: str = "flagged",
                    attacker_col: str = "is_attacker", seed_col: str = "seed"):
    """Detection rate on attackers AND false-alarm rate on honest clients.

    Returns both, or raises. There is no path through this module that reports
    one without the other. A detection rate means nothing on its own: the
    architecture study once measured false alarms in-sample, got 0.000, and
    every detection number looked like signal until leave-one-seed-out fixed it
    and CHANGED THE CONCLUSION.

    `verdict` is derived from the two rates, never hand-written.
    """
    for c in (flagged_col, attacker_col):
        if c not in df.columns:
            raise KeyError(f"detection_table needs column '{c}'")

    atk = df[df[attacker_col].astype(bool)]
    hon = df[~df[attacker_col].astype(bool)]
    if len(atk) == 0 or len(hon) == 0:
        raise ValueError(f"need both attacker and honest rows to report a "
                         f"detection rate; got {len(atk)} attacker, {len(hon)} honest")

    tp = int(atk[flagged_col].astype(bool).sum())
    fp = int(hon[flagged_col].astype(bool).sum())
    det, far = tp / len(atk), fp / len(hon)

    if det <= far:
        verdict = (f"the attacker is flagged no more often than an innocent "
                   f"client ({det:.1%} against {far:.1%}): below noise")
    else:
        verdict = f"detection {det:.1%} against a {far:.1%} false-alarm rate"

    return {"detection_rate": det, "false_alarm_rate": far,
            "attacker_rows": len(atk), "honest_rows": len(hon),
            "flagged_attackers": tp, "flagged_honest": fp,
            "n_seeds": int(df[seed_col].nunique()) if seed_col in df.columns else None,
            "verdict": verdict}


def per_seed_breakdown(df, value_col: str, level_col: str, seed_col: str = "seed"):
    """Mean of `value_col` per (level, seed), and a warning when it is not constant.

    Infiltration once read 0.500 for two knowledge levels. The breakdown was
    seed 0 at 1.0 and seed 1 at 0.0 for both: nothing was half-working, one seed
    worked and one did not. A mean over few seeds can be a seed effect rather
    than a rate, and the only way to see that is to look.
    """
    import numpy as np

    table = df.groupby([level_col, seed_col])[value_col].mean().unstack(seed_col)
    unstable = []
    for level, row in table.iterrows():
        vals = row.dropna().to_numpy(dtype=float)
        if len(vals) > 1 and float(np.ptp(vals)) > 1e-9:
            unstable.append((level, float(vals.min()), float(vals.max())))

    if unstable:
        print(f"[warn] '{value_col}' is not constant across seeds:")
        for level, lo, hi in unstable:
            print(f"       {level_col}={level}: {lo:.3f} to {hi:.3f} across seeds. "
                  f"Treat the mean as a seed effect until more seeds say otherwise.")
    return table


def list_runs(root: Path | str = RUNS_ROOT) -> list[dict]:
    """All runs, newest first, with their manifest fields where available."""
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for d in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        entry = {"run": d.name, "path": str(d),
                 "figures": len(list((d / "figures").glob("*.png"))),
                 "complete": (d / "manifest.json").exists()}
        if entry["complete"]:
            try:
                entry.update(json.loads((d / "manifest.json").read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        out.append(entry)
    return out
