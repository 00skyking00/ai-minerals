"""One-time CSV → Parquet conversion for the Lawley 2022 datacube.

The published CSV is 7.2 GB. Loaded naively, pandas inflates it to
~25 GB (object dtypes everywhere). WSL has 16 GB of RAM.

Streams the CSV in chunks and writes each chunk as a parquet row
group via pyarrow ParquetWriter. Peak memory is ~one chunk worth.

Output:
  data/raw/lawley2022/datacube.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path("/home/sky/src/learning/ai-minerals/data/raw/lawley2022")
CSV_PATH = DATA_DIR / "2021_Table04_Datacube.csv"
PARQUET_PATH = DATA_DIR / "datacube.parquet"
CHUNK_ROWS = 200_000

# Both the Geology_Dictionary_* and Training_* columns use the strings
# "Present" / "Absent" / NaN in the raw CSV. Cast them all the same way
# (== "Present" rather than astype(bool), which would make "Absent"
# truthy because it's a non-empty string).
PRESENT_ABSENT_COLS = [
    "Geology_Dictionary_Alkalic", "Geology_Dictionary_Anatectic",
    "Geology_Dictionary_Calcareous", "Geology_Dictionary_Carbonaceous",
    "Geology_Dictionary_Cherty", "Geology_Dictionary_CoarseClastic",
    "Geology_Dictionary_Evaporitic", "Geology_Dictionary_Felsic",
    "Geology_Dictionary_FineClastic", "Geology_Dictionary_Gneissose",
    "Geology_Dictionary_Igneous", "Geology_Dictionary_Intermediate",
    "Geology_Dictionary_Pegmatitic", "Geology_Dictionary_RedBed",
    "Geology_Dictionary_Schistose", "Geology_Dictionary_Sedimentary",
    "Geology_Dictionary_UltramaficMafic",
    "Training_MVT_Deposit", "Training_MVT_Occurrence",
    "Training_CD_Deposit", "Training_CD_Occurrence",
]


def downcast_chunk(df: pd.DataFrame) -> pd.DataFrame:
    for col in PRESENT_ABSENT_COLS:
        if col in df.columns:
            df[col] = (df[col] == "Present")
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def main() -> None:
    print(f"=== CSV → Parquet streaming conversion ===", flush=True)
    print(f"input : {CSV_PATH}", flush=True)
    print(f"output: {PARQUET_PATH}", flush=True)
    assert CSV_PATH.exists(), f"missing {CSV_PATH}"

    t0 = time.time()
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    schema: pa.Schema | None = None

    for i, chunk in enumerate(pd.read_csv(
        CSV_PATH, encoding="latin-1", chunksize=CHUNK_ROWS, low_memory=False,
    )):
        chunk = downcast_chunk(chunk)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(
                PARQUET_PATH, schema,
                compression="snappy",
                use_dictionary=True,
            )
        else:
            # Align to the first chunk's schema to avoid drift in
            # rarely-seen string columns.
            table = table.cast(schema, safe=False)
        writer.write_table(table)
        n_rows += len(chunk)
        print(f"  chunk {i:>3}: +{len(chunk):,} rows  "
              f"total {n_rows:>9,}  ({(time.time()-t0)/60:>5.1f} min)",
              flush=True)

    if writer is not None:
        writer.close()

    file_mb = PARQUET_PATH.stat().st_size / 1e6
    print(f"\nwrote {PARQUET_PATH} ({file_mb:.0f} MB on disk)", flush=True)
    print(f"total: {n_rows:,} rows, wall {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
