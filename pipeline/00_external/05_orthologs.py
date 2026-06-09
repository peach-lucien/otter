"""Align mouse and human gene expression matrices via mouse-human orthologs.

Reads the gene lists produced by 02c_mouse_genes_v2.py and 04_human_genes.py and
finds the mouse-gene ↔ human-gene ortholog pairs. Restricts both matrices to
the orthologous gene set in the same order, so direct comparison is possible.

Sources of orthology, in priority order:
  1. Local NCBI Homologene file (small, ships with this repo for reproducibility)
  2. The `pyhomologene` package if installed
  3. Mygene.info HTTP API (live lookup, slower but always current)

Output:
  data_external/orthologs.csv                   ortholog table
  data_external/mouse_genes_aligned.npy         (1864, n_orth) restricted
  data_external/human_genes_aligned.npy         (2094, n_orth) restricted
  data_external/orthologs_meta.json             alignment summary
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_external"


def lookup_orthologs_via_mygene(human_symbols: list[str]) -> pd.DataFrame:
    """For each human gene symbol, look up the mouse ortholog via mygene.info."""
    import requests
    rows = []
    print(f"querying mygene.info for {len(human_symbols)} human symbols (batched)...")
    batch_size = 200
    for k in range(0, len(human_symbols), batch_size):
        batch = human_symbols[k:k + batch_size]
        try:
            resp = requests.post(
                "https://mygene.info/v3/query",
                data={
                    "q":      ",".join(batch),
                    "scopes": "symbol",
                    "fields": "symbol,entrezgene,homologene",
                    "species": "human",
                },
                timeout=30,
            )
            resp.raise_for_status()
            for hit in resp.json():
                if "symbol" not in hit: continue
                rows.append({
                    "human_symbol": hit["symbol"],
                    "human_entrez": hit.get("entrezgene"),
                    "homologene":   (hit.get("homologene") or {}).get("id"),
                })
        except Exception as e:
            print(f"  batch {k}: {e}")
        time.sleep(0.5)

    # Now look up the mouse member of each homologene group
    hg_ids = sorted({r["homologene"] for r in rows if r.get("homologene")})
    print(f"resolving mouse orthologs for {len(hg_ids)} homologene groups...")
    hg_to_mouse = {}
    for k in range(0, len(hg_ids), batch_size):
        batch = hg_ids[k:k + batch_size]
        try:
            resp = requests.post(
                "https://mygene.info/v3/query",
                data={
                    "q":      ",".join(map(str, batch)),
                    "scopes": "homologene.id",
                    "fields": "symbol,entrezgene,homologene,taxid",
                    "species": "mouse",
                },
                timeout=30,
            )
            resp.raise_for_status()
            for hit in resp.json():
                hg_id = (hit.get("homologene") or {}).get("id")
                if not hg_id or "symbol" not in hit: continue
                hg_to_mouse[hg_id] = {
                    "mouse_symbol": hit["symbol"],
                    "mouse_entrez": hit.get("entrezgene"),
                }
        except Exception as e:
            print(f"  batch {k}: {e}")
        time.sleep(0.5)

    out = []
    for r in rows:
        hg = r.get("homologene")
        if hg in hg_to_mouse:
            out.append({**r, **hg_to_mouse[hg]})
    return pd.DataFrame(out)


def main():
    mouse_meta = pd.read_csv(OUT / "mouse_gene_list.csv")
    human_meta = pd.read_csv(OUT / "human_gene_list.csv")
    print(f"mouse genes: {len(mouse_meta)}, human genes: {len(human_meta)}")
    if len(mouse_meta) == 0:
        print("\nERROR: mouse_gene_list.csv is empty. Did 02c_mouse_genes_v2.py finish?")
        print("Re-run: PYTHONPATH=src python pipeline/00_external/02c_mouse_genes_v2.py")
        sys.exit(1)
    if len(human_meta) == 0:
        print("ERROR: human_gene_list.csv is empty. Run 04_human_genes.py first.")
        sys.exit(1)

    # 1. Try local Homologene CSV bundled with the repo, if present
    local_homologene = ROOT / "scripts" / "external" / "homologene.csv"
    if local_homologene.exists():
        print(f"using local homologene table at {local_homologene}")
        ortho = pd.read_csv(local_homologene)
    else:
        print("no local homologene; using mygene.info live lookup")
        ortho = lookup_orthologs_via_mygene(human_meta["gene_symbol"].dropna().unique().tolist())

    # 2. Match each mouse_gene_list row to a homologene group
    if "mouse_symbol" in ortho.columns:
        merged_mouse = mouse_meta.merge(
            ortho[["mouse_symbol", "homologene"]],
            left_on="gene_symbol", right_on="mouse_symbol", how="inner",
        )
    elif "mouse_entrez" in ortho.columns and "entrez_id" in mouse_meta.columns:
        merged_mouse = mouse_meta.merge(
            ortho[["mouse_entrez", "homologene"]],
            left_on="entrez_id", right_on="mouse_entrez", how="inner",
        )
    else:
        raise RuntimeError("ortho table has neither mouse_symbol nor mouse_entrez")

    merged_human = human_meta.merge(
        ortho[["human_symbol", "homologene"]],
        left_on="gene_symbol", right_on="human_symbol", how="inner",
    )

    # 3. Inner-join on homologene
    pairs = merged_mouse.merge(
        merged_human, on="homologene", suffixes=("_m", "_h"),
    )
    print(f"ortholog pairs: {len(pairs)}")

    # 4. Restrict the two expression matrices to the matched genes ----------
    mouse_expr = np.load(OUT / "mouse_genes.npy")     # (1864, n_mouse)
    human_expr = np.load(OUT / "human_genes.npy")     # (2094, n_human)

    mouse_idx = []; human_idx = []
    for _, row in pairs.iterrows():
        mi = mouse_meta.index[mouse_meta["gene_symbol"] == row["gene_symbol_m"]]
        hi = human_meta.index[human_meta["gene_symbol"] == row["gene_symbol_h"]]
        if len(mi) and len(hi):
            mouse_idx.append(int(mi[0]))
            human_idx.append(int(hi[0]))
    mouse_idx = np.asarray(mouse_idx, dtype=np.int64)
    human_idx = np.asarray(human_idx, dtype=np.int64)
    if len(mouse_idx) == 0:
        print("ERROR: no ortholog pairs matched both gene lists.")
        print("This usually means the gene symbols in the mouse and human lists")
        print("are in different formats (mouse uses Camelcase 'Cux1', human uses")
        print("UPPERCASE 'CUX1'). Inspect ortholog table:")
        print(pairs.head())
        sys.exit(1)

    mouse_aligned = mouse_expr[:, mouse_idx]
    human_aligned = human_expr[:, human_idx]
    np.save(OUT / "mouse_genes_aligned.npy", mouse_aligned.astype(np.float32))
    np.save(OUT / "human_genes_aligned.npy", human_aligned.astype(np.float32))

    pairs.to_csv(OUT / "orthologs.csv", index=False)
    info = {
        "n_pairs":        int(len(mouse_idx)),
        "mouse_aligned_shape": list(mouse_aligned.shape),
        "human_aligned_shape": list(human_aligned.shape),
    }
    (OUT / "orthologs_meta.json").write_text(json.dumps(info, indent=2))
    print(f"\nsaved → {OUT / 'orthologs.csv'}")
    print(f"        {OUT / 'mouse_genes_aligned.npy'}  {mouse_aligned.shape}")
    print(f"        {OUT / 'human_genes_aligned.npy'}  {human_aligned.shape}")


if __name__ == "__main__":
    main()
