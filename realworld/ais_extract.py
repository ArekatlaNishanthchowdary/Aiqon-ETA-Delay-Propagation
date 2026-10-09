"""Stream DMA daily AIS zips (aisdk-YYYY-MM-DD.zip) into compact per-day files: realworld/data/days/<date>.csv.gz.

Keeps Class A transponders of commercial ship types, one row (the last) per MMSI per 5-minute bin, so a 3 GB / 17M-message day
shrinks to ~150k rows. Every AIS position row carries the vessel's CURRENT declared Destination and ETA, so no separate
static-message join is needed. Idempotent (skips existing days unless --force).
Usage: python realworld/ais_extract.py <zip-or-folder> [more ...] [--force]
The input zips are treated as untrusted data and are only read, never executed."""
import argparse, zipfile, sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "days"
COMMERCIAL = ["Cargo", "Tanker", "Passenger", "Other"]
COLS = ["# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude", "Navigational status", "SOG", "IMO", "Name",
        "Ship type", "Length", "Draught", "Destination", "ETA"]
KEEP = ["ts", "MMSI", "Latitude", "Longitude", "Navigational status", "SOG", "IMO", "Ship type", "Length", "Draught", "Destination", "ETA"]


def extract(zpath: Path, force=False) -> Path | None:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{zpath.name}: expected exactly one CSV inside, found {z.namelist()}")
        day = "-".join(Path(names[0]).stem.split("-")[-3:])
        out = OUT / f"{day}.csv.gz"
        if out.exists() and not force:
            print(f"skip {day} (exists)"); return out
        parts = []
        with z.open(names[0]) as f:
            header = pd.read_csv(f, nrows=0).columns
            missing = [c for c in COLS if c not in header]
            if missing:
                raise ValueError(f"{zpath.name}: unexpected AIS schema, missing columns {missing}")
        with z.open(names[0]) as f:
            for ch in pd.read_csv(f, usecols=COLS, chunksize=1_000_000, low_memory=False):
                ch = ch[(ch["Type of mobile"] == "Class A") & ch["Ship type"].isin(COMMERCIAL)].copy()
                ch["ts"] = pd.to_datetime(ch["# Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
                ch = ch.dropna(subset=["ts", "Latitude", "Longitude"])
                ch["bin"] = ch.ts.dt.floor("5min")
                parts.append(ch.drop_duplicates(["MMSI", "bin"], keep="last")[KEEP + ["bin"]])
    df = pd.concat(parts).sort_values("ts").drop_duplicates(["MMSI", "bin"], keep="last").drop(columns="bin")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"{day}: {len(df):,} rows, {df.MMSI.nunique()} vessels -> {out.name}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    zips = sorted(z for p in map(Path, a.paths) for z in ([p] if p.is_file() else p.glob("aisdk-*.zip")))
    if not zips:
        sys.exit("no aisdk-*.zip files found")
    for z in zips:
        extract(z, a.force)
