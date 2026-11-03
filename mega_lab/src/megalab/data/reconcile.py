"""Reconciliação entre fontes por votação.

Para cada concurso, cada fonte "vota" no conjunto de 6 dezenas que publica.

Status do concurso na base canônica:

- ``A``: 3 ou mais fontes concordam e nenhuma discorda;
- ``B``: exatamente 2 fontes concordam e nenhuma discorda;
- ``C``: uma única fonte (não verificado);
- ``X``: há divergência entre fontes (o conjunto majoritário é mantido apenas
  se tiver maioria estrita com pelo menos 2 votos; caso contrário a linha é
  marcada ``resolvido=False`` e excluída das análises).

A data e a ordem de extração são reconciliadas da mesma forma, com status
próprios.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from megalab.data.sources import SOURCES

B_COLS = [f"b{i}" for i in range(1, 7)]
O_COLS = [f"o{i}" for i in range(1, 7)]


def reconcile(extracts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    votes: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    dates: dict[int, list[tuple[str, str]]] = defaultdict(list)
    orders: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)

    for sid, df in extracts.items():
        for row in df.itertuples(index=False):
            c = int(row.concurso)
            nums = tuple(int(getattr(row, b)) for b in B_COLS if not pd.isna(getattr(row, b)))
            votes[c].append((sid, nums))
            if isinstance(row.data, str) and row.data:
                dates[c].append((sid, row.data))
            o = [getattr(row, oc) for oc in O_COLS]
            if not any(pd.isna(v) for v in o):
                orders[c].append((sid, tuple(int(v) for v in o)))

    records, conflicts = [], []
    for c in sorted(votes):
        tally = Counter(nums for _, nums in votes[c])
        (winner, n_win), *rest = tally.most_common()
        conflict = len(tally) > 1
        resolved = (not conflict) or (n_win >= 2 and n_win > rest[0][1])
        supporters = [sid for sid, nums in votes[c] if nums == winner]
        origins = {SOURCES[s].origin for s in supporters if s in SOURCES}
        if conflict:
            status = "X"
            for sid, nums in votes[c]:
                conflicts.append({"concurso": c, "campo": "dezenas", "fonte": sid,
                                  "valor": " ".join(map(str, nums)),
                                  "majoritario": " ".join(map(str, winner))})
        else:
            status = "A" if n_win >= 3 else ("B" if n_win == 2 else "C")

        # Data
        dtally = Counter(d for _, d in dates[c])
        if not dtally:
            data, status_data = "", "ausente"
        else:
            (data, nd), *drest = dtally.most_common()
            if len(dtally) == 1:
                status_data = "ok" if nd >= 2 else "fonte_unica"
            else:
                status_data = "conflito"
                for sid, d in dates[c]:
                    conflicts.append({"concurso": c, "campo": "data", "fonte": sid,
                                      "valor": d, "majoritario": data})

        # Ordem de extração (somente fontes cujo conjunto confere com o majoritário)
        valid_orders = [(sid, o) for sid, o in orders[c] if tuple(sorted(o)) == winner]
        otally = Counter(o for _, o in valid_orders)
        order: tuple[int, ...] | None = None
        if not otally:
            status_ordem = "ausente"
        elif len(otally) == 1:
            order = next(iter(otally))
            status_ordem = "ok" if len(valid_orders) >= 2 else "fonte_unica"
        else:
            status_ordem = "conflito"
            for sid, o in valid_orders:
                conflicts.append({"concurso": c, "campo": "ordem", "fonte": sid,
                                  "valor": " ".join(map(str, o)), "majoritario": ""})

        rec = {"concurso": c, "data": data}
        for i, v in enumerate(winner, start=1):
            rec[f"b{i}"] = v
        for i in range(6):
            rec[f"o{i + 1}"] = order[i] if order else pd.NA
        rec.update({
            "status": status, "resolvido": bool(resolved), "n_fontes": n_win,
            "n_origens": len(origins), "fontes": ";".join(sorted(supporters)),
            "status_data": status_data, "status_ordem": status_ordem,
        })
        records.append(rec)

    canon = pd.DataFrame(records)
    for col in O_COLS:
        canon[col] = canon[col].astype("Int64")
    conf = pd.DataFrame(conflicts, columns=["concurso", "campo", "fonte", "valor", "majoritario"])
    return canon, conf
