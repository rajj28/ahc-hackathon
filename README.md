# AHC Video Anomaly Detection

Frozen vision encoder + small trained temporal heads for real-time video anomaly detection.
See `ARCHITECTURE.md` for the design, `CODEX.md` for implementation contracts, and
`tasks/BACKLOG.md` for the build order.

## Run

    .venv\Scripts\python.exe -m ahc.features --split train --encoder siglip2   # cache
    .venv\Scripts\python.exe -m ahc.train clip
    .venv\Scripts\python.exe -m ahc.train temporal
    .venv\Scripts\python.exe -m ahc.predict --manifest manifest.json --out runs/sub.json
    .venv\Scripts\python.exe -m ahc.validate runs/sub.json --manifest manifest.json

## Layout

    configs/default.yaml   every tunable; no magic numbers in code
    src/ahc/               package
    cache/{encoder}/       embedding cache (.npz per video)
    runs/                  checkpoints, submissions, threshold sweeps
    train/ test/           dataset (not committed)
