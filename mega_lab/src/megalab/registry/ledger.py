"""Livro-razão append-only de previsões e avaliações, encadeado por SHA-256.

Cada linha de ``registry/ledger.jsonl``::

    {"seq": 7, "tipo": "previsao", "registrado_em": "...Z", "conteudo": {...},
     "hash_anterior": "<64 hex>", "hash": "<64 hex>"}

``hash = sha256(hash_anterior + json_canonico({seq, tipo, registrado_em, conteudo}))``.
Alterar, remover ou reordenar qualquer linha quebra a cadeia, o que
``verify`` detecta. O carimbo de tempo externo é o commit/push no GitHub.

Regras impostas:

1. Previsão só para o **próximo** concurso (último da base + 1).
2. Um único registro de previsão por concurso — nada de registrar de novo
   até "acertar".
3. Avaliação é uma entrada nova, que referencia o hash da previsão.
4. Previsões registradas depois do horário do sorteio (data do concurso,
   20h de Brasília = 23h UTC) são marcadas como tardias e excluídas das
   estatísticas prospectivas.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from megalab import config
from megalab.combinatorics import set_log_likelihood_ratio
from megalab.models.base import Model, ticket_from_proba

GENESIS = "0" * 64
DRAW_CUTOFF_UTC_HOUR = 23  # 20h de Brasília


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(prev: str, payload: dict) -> str:
    return hashlib.sha256((prev + canonical_json(payload)).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def code_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=config.ROOT, check=True).stdout.strip()
    except Exception:  # pragma: no cover - fora de um repositório git
        return "desconhecido"


class Ledger:
    def __init__(self, path: Path = config.LEDGER_PATH):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, tipo: str, conteudo: dict, registrado_em: str | None = None) -> dict:
        entries = self.entries()
        prev = entries[-1]["hash"] if entries else GENESIS
        payload = {"seq": len(entries) + 1, "tipo": tipo,
                   "registrado_em": registrado_em or utc_now(), "conteudo": conteudo}
        rec = {**payload, "hash_anterior": prev, "hash": _hash(prev, payload)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(canonical_json(rec) + "\n")
        return rec

    def verify(self) -> tuple[bool, list[str]]:
        problems, prev = [], GENESIS
        for i, e in enumerate(self.entries(), start=1):
            payload = {k: e[k] for k in ("seq", "tipo", "registrado_em", "conteudo")}
            if e["seq"] != i:
                problems.append(f"linha {i}: seq {e['seq']} fora de ordem")
            if e["hash_anterior"] != prev:
                problems.append(f"linha {i}: hash_anterior não confere")
            if _hash(e["hash_anterior"], payload) != e["hash"]:
                problems.append(f"linha {i}: hash não confere (conteúdo alterado)")
            prev = e["hash"]
        return (not problems), problems

    def predictions(self) -> dict[int, dict]:
        return {e["conteudo"]["concurso_alvo"]: e for e in self.entries() if e["tipo"] == "previsao"}

    def evaluations(self) -> dict[int, dict]:
        return {e["conteudo"]["concurso"]: e for e in self.entries() if e["tipo"] == "avaliacao"}


def register_predictions(ledger: Ledger, draws: np.ndarray, last_concurso: int, base_sha256: str,
                         models: list[Model], hedge_weights: dict[str, float] | None = None,
                         seed: int = config.SEED, registrado_em: str | None = None) -> dict:
    ok, problems = ledger.verify()
    if not ok:
        raise RuntimeError("Livro-razão corrompido: " + "; ".join(problems))
    target = int(last_concurso) + 1
    if target in ledger.predictions():
        raise ValueError(f"Já existe previsão registrada para o concurso {target}.")
    if len(draws) != last_concurso:
        raise ValueError("A base precisa ser contígua de 1 até o último concurso.")
    hist = np.asarray(draws).copy()
    hist.setflags(write=False)
    preds = {}
    probs = {}
    for m in models:
        p = m.predict_proba(hist)
        probs[m.name] = p
    if hedge_weights:
        members = [k for k in hedge_weights if k in probs]
        w = np.array([hedge_weights[k] for k in members])
        w = w / w.sum()
        from megalab.models.base import normalize
        probs["hedge"] = normalize(sum(wi * probs[k] for wi, k in zip(w, members)))
    for name, p in probs.items():
        rng = np.random.default_rng([seed, target, zlib.crc32(name.encode())])
        if name == "aleatorio":
            ticket = np.sort(rng.choice(60, 6, replace=False) + 1)
        else:
            ticket = ticket_from_proba(p, rng)
        preds[name] = {"bilhete": [int(x) for x in ticket], "p": [round(float(x), 6) for x in p]}
    content = {
        "concurso_alvo": target, "base_ultimo_concurso": int(last_concurso),
        "base_sha256": base_sha256, "codigo_commit": code_commit(),
        "versao": __import__("megalab").__version__, "semente": seed,
        "pesos_hedge": {k: round(float(v), 6) for k, v in (hedge_weights or {}).items()},
        "previsoes": preds,
    }
    return ledger.append("previsao", content, registrado_em=registrado_em)


def _is_late(registered: str, draw_date: str) -> bool:
    if not draw_date:
        return False
    reg = datetime.strptime(registered, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    cutoff = datetime.strptime(draw_date, "%Y-%m-%d").replace(
        hour=DRAW_CUTOFF_UTC_HOUR, tzinfo=timezone.utc)
    return reg >= cutoff


def score(ledger: Ledger, results: dict[int, tuple[list[int], str]]) -> list[dict]:
    """Avalia previsões cujo resultado (dezenas, data) está disponível e ainda não avaliado."""
    done = ledger.evaluations()
    new = []
    for target, entry in sorted(ledger.predictions().items()):
        if target in done or target not in results:
            continue
        nums, draw_date = results[target]
        drawn = np.array(sorted(nums))
        per_model = {}
        for name, pr in entry["conteudo"]["previsoes"].items():
            p = np.array(pr["p"])
            llr = float(set_log_likelihood_ratio(p[None, :], drawn[None, :] - 1)[0])
            per_model[name] = {"acertos": len(set(pr["bilhete"]) & set(nums)),
                               "bits": llr / np.log(2), "lr": float(np.exp(llr))}
        content = {"concurso": target, "resultado": [int(x) for x in drawn], "data_sorteio": draw_date,
                   "previsao_seq": entry["seq"], "previsao_hash": entry["hash"],
                   "registro_tardio": _is_late(entry["registrado_em"], draw_date),
                   "por_modelo": per_model}
        new.append(ledger.append("avaliacao", content))
    return new


def prospective_summary(ledger: Ledger) -> dict:
    """Acumulado prospectivo por modelo (exclui registros tardios)."""
    evals = [e for e in ledger.evaluations().values() if not e["conteudo"]["registro_tardio"]]
    evals.sort(key=lambda e: e["conteudo"]["concurso"])
    summary: dict[str, dict] = {}
    for e in evals:
        for name, r in e["conteudo"]["por_modelo"].items():
            s = summary.setdefault(name, {"concursos": 0, "acertos": 0, "bits": 0.0, "log10_evalor": 0.0})
            s["concursos"] += 1
            s["acertos"] += r["acertos"]
            s["bits"] += r["bits"]
            s["log10_evalor"] += float(np.log10((1 - config.EVALUE_MIX) + config.EVALUE_MIX * r["lr"]))
    thr = np.log10(config.PROSPECTIVE_EVALUE_THRESHOLD)
    for s in summary.values():
        s["acertos_esperado"] = 0.6 * s["concursos"]
        s["supera_limiar"] = bool(s["log10_evalor"] >= thr)
    return {"concursos_avaliados": len(evals), "limiar_evalor": config.PROSPECTIVE_EVALUE_THRESHOLD,
            "modelos": summary}
