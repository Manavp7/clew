"""Splink (Fellegi-Sunter) linkage backend for entity resolution.

Splink computes probabilistic name-match scores; we keep CIK as an authoritative
anchor and cluster with union-find (same contract as :mod:`resolver`). Fixed m/u
weights are specified directly rather than EM-trained, which is the robust choice
for the Phase-1 cold-start data volume. As the corpus grows, switch to
``estimate_u_using_random_sampling`` + EM training for sharper weights.
"""

from __future__ import annotations

from clew.resolve.normalize import normalized
from clew.resolve.resolver import Cluster, Record, _pick_canonical, _UnionFind

MATCH_THRESHOLD = 0.9
# Below this many records, EM training is unstable -> use fixed Fellegi-Sunter weights.
EM_MIN_RECORDS = 500


def resolve_records_splink(records: list[Record], *, train: bool = True) -> list[Cluster]:
    import pandas as pd
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    if not records:
        return []

    rows = [
        {
            "unique_id": r.rid,
            "name_norm": normalized(r.name),
            "etype": r.entity_type,
            "cik": r.cik,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)

    name_cmp = cl.JaroWinklerAtThresholds("name_norm", [0.92, 0.85]).configure(
        m_probabilities=[0.7, 0.2, 0.07, 0.03],
        u_probabilities=[0.0001, 0.001, 0.02, 0.979],
    )
    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("substr(name_norm, 1, 4)"),
            block_on("cik"),
        ],
        comparisons=[name_cmp],
        probability_two_random_records_match=0.01,
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker(df, settings, db_api=DuckDBAPI())

    # EM training only when there is enough data; otherwise keep the fixed weights
    # (Splink's EM cold-starts poorly on small corpora — see P3.5 rationale).
    if train and len(records) >= EM_MIN_RECORDS:
        try:
            from splink import block_on as _block_on

            linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
            linker.training.estimate_parameters_using_expectation_maximisation(
                _block_on("substr(name_norm, 1, 4)")
            )
        except Exception as exc:  # noqa: BLE001 - fall back to fixed weights
            print(f"  ! Splink EM training failed, using fixed weights: {exc}")

    preds = linker.inference.predict(threshold_match_probability=0.5).as_pandas_dataframe()

    uf = _UnionFind(len(records))
    pos = {r.rid: i for i, r in enumerate(records)}

    # Splink name links (respecting CIK authority).
    for _, row in preds.iterrows():
        if row["match_probability"] < MATCH_THRESHOLD:
            continue
        i, j = pos[int(row["unique_id_l"])], pos[int(row["unique_id_r"])]
        ci, cj = records[i].cik, records[j].cik
        if ci and cj and ci != cj:
            continue
        uf.union(i, j)

    # CIK anchor.
    by_cik: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        if r.cik:
            by_cik.setdefault(r.cik, []).append(i)
    for idxs in by_cik.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    groups: dict[int, list[int]] = {}
    for i in range(len(records)):
        groups.setdefault(uf.find(i), []).append(i)

    clusters: list[Cluster] = []
    for idxs in groups.values():
        members = [records[i] for i in idxs]
        names = [m.name for m in members]
        canonical_name = _pick_canonical(names)
        ciks = {m.cik for m in members if m.cik}
        cik = next(iter(ciks)) if ciks else None
        clusters.append(
            Cluster(
                canonical_name=canonical_name,
                entity_type=members[0].entity_type,
                cik=cik,
                aliases=sorted({n for n in names if n != canonical_name}),
                members=members,
                confidence=1.0 if cik else MATCH_THRESHOLD,
            )
        )
    return clusters
