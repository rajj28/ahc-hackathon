"""Dataset acquisition, verification and merge across the five Drive mirrors.

STATUS AS OF THE LOCAL COPY: complete. 3,173/3,173 train + 34/34 test files present, none
zero-byte, none undecodable, no orphans, 13.9 GB. The other four mirrors are redundant copies
of the same pack, so nothing here needs to run unless one of these happens:

  - the organizers publish a corrected ground_truth.csv or additional footage mid-event
  - a mirror turns out to be a SHARD rather than a copy (verify_mirror decides this, cheaply)
  - local files get corrupted

Design rule: this module NEVER overwrites an existing good file and never edits the canonical
tree in place. It stages, diffs, reports, and merges only what is genuinely new. A dataset
tool that clobbers during a seven-hour event is worse than no tool.

    python -m ahc.ingest verify                       # local coverage report
    python -m ahc.ingest probe --mirror 2             # copy or shard? (CSVs only, ~KB)
    python -m ahc.ingest pull  --mirror 2 --to staging/m2
    python -m ahc.ingest merge --from staging/m2 --dry-run
    python -m ahc.ingest merge --from staging/m2
"""
from __future__ import annotations


def verify(root: str = ".") -> dict:
    """Coverage + integrity of the canonical tree. Run this FIRST, always.

    For each of the 12 train class folders and test/: join ground_truth.csv to videos.csv,
    resolve every `filename`, and report present / missing / zero-byte / orphan counts plus
    total bytes. Then decode-spot-check a sample per class (cv2 frame count > 0).

    -> {'complete': bool, 'missing': [...], 'zero': [...], 'orphans': [...], 'bytes': int}

    A `filename` in videos.csv uses forward slashes ("videos/TR01350.mp4") and must be
    os.sep-normalized before joining. Paths are relative to the class folder, not the root.
    """
    raise NotImplementedError


def probe(mirror: int, cfg) -> dict:
    """Is this mirror a COPY of what we have, or a SHARD with different videos?

    Pulls only the CSVs (a few KB, no video), unions their video_id sets, and diffs against
    local. This is the cheap question that decides whether a full pull is worth minutes of a
    seven-hour budget.

    -> {'kind': 'copy' | 'shard' | 'superset', 'only_remote': [...], 'only_local': [...],
        'csv_differs': bool}

    `csv_differs` compares the ground-truth ROWS for shared video_ids, not just the id sets —
    a corrected annotation is the one realistic reason to care about another mirror, and it
    would leave the id sets identical.
    """
    raise NotImplementedError


def pull(mirror: int, dest: str, cfg, csv_only: bool = False) -> None:
    """Fetch a mirror into a staging directory. Never writes to train/ or test/.

    Transport, in order of preference:
      1. rclone with a configured Drive remote — the only option that handles folders with
         hundreds of files reliably, and it resumes.
      2. Manual browser download. This is what produced the local copy: Drive zips a large
         folder into multi-part archives. Point `--from` at the folder of parts and let
         `merge` do the rest.
      3. gdown --folder. CONVENIENT BUT LIMITED: it silently caps at 50 files per folder,
         and train/<class>/videos/ holds up to 973. Acceptable for CSVs only; do NOT trust
         it for video pulls. If used, verify() afterwards will catch the shortfall.

    Resumable: skip any file already staged with a matching size.
    """
    raise NotImplementedError


def merge(staging: str, root: str = ".", dry_run: bool = True) -> dict:
    """Merge staged content into the canonical tree. Additive and idempotent.

    Rules, in force even when dry_run is False:
      - a video_id already present locally and non-zero-byte is NEVER overwritten
      - a new video_id is copied in, and its row appended to the right ground_truth.csv and
        videos.csv, preserving column order and the existing schema exactly
        (train CSVs have NO `level` column; test's does — do not homogenize them)
      - a CHANGED ground-truth row for an existing video_id is reported and NOT applied
        silently; it needs a human decision, so print a diff and require --accept-csv-changes
      - anything unresolvable is left in staging and named in the report

    -> {'added': [...], 'skipped': [...], 'conflicts': [...]}

    Always run with --dry-run first and read the report. After any real merge, re-run
    verify(), and invalidate the affected embedding cache entries: a merged video whose
    .npz already exists would otherwise keep stale features.
    """
    raise NotImplementedError


def unpack_parts(parts_dir: str, dest: str) -> None:
    """Reassemble Drive's multi-part browser download.

    Drive splits a large folder into `<name>-001.zip`, `<name>-002.zip`, … Each part is an
    independent archive holding a subset of the tree, NOT a split of one archive, so each is
    extracted separately into the same destination and the union forms the whole.
    On PowerShell 5.1 use `Expand-Archive -Force`; the .NET
    ZipFile.ExtractToDirectory(src, dst, True) overload does not exist there and throws on the
    boolean argument.
    """
    raise NotImplementedError
