"""Fontes de dados: registro, download, verificação de hash e normalização.

Cada fonte é descrita por um :class:`Source` com URL fixada (commit), hash
SHA-256 esperado do arquivo bruto, licença e "origem" (de onde a fonte
obteve os dados). Um parser converte o arquivo bruto para o formato
normalizado::

    concurso,data,b1,b2,b3,b4,b5,b6,o1,o2,o3,o4,o5,o6

- ``b1..b6``: dezenas em ordem crescente;
- ``o1..o6``: dezenas na ordem de extração (vazio quando a fonte não tem);
- ``data``: AAAA-MM-DD (vazio quando a fonte não tem).

Somente fatos (número do concurso, data, dezenas) são extraídos — nenhum
código das fontes é reutilizado.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from megalab.config import RAW_CACHE_DIR, SOURCES_DIR

NORMALIZED_COLUMNS = (
    ["concurso", "data"]
    + [f"b{i}" for i in range(1, 7)]
    + [f"o{i}" for i in range(1, 7)]
)

RAW_GH = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class Source:
    id: str
    description: str
    url: str
    filename: str
    sha256: str
    license: str
    origin: str               # de onde a própria fonte obteve os dados
    has_order: bool
    parser: Callable[[Path], pd.DataFrame] = field(repr=False, compare=False)
    retrieved: str = "2026-11-03"
    live: bool = False        # URL de branch, atualizável por `megalab data update`


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _br_date(s: str) -> str:
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""
    return datetime.strptime(s, "%d/%m/%Y").strftime("%Y-%m-%d")


def _iso_date(s: str) -> str:
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return ""
    return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")


def _frame(rows: list[tuple[int, str, list[int], list[int] | None]]) -> pd.DataFrame:
    """Monta o DataFrame normalizado a partir de (concurso, data, dezenas, ordem)."""
    out = []
    for concurso, data, numbers, order in rows:
        nums = [int(x) for x in numbers]
        rec = {"concurso": int(concurso), "data": data}
        for i, v in enumerate(sorted(nums), start=1):
            rec[f"b{i}"] = v
        for i in range(1, 7):
            rec[f"o{i}"] = (int(order[i - 1]) if order is not None else pd.NA)
        if len(nums) != 6:
            # Mantém a linha para que a verificação de integridade a acuse.
            rec["_n"] = len(nums)
        out.append(rec)
    df = pd.DataFrame(out)
    for c in NORMALIZED_COLUMNS:
        if c not in df:
            df[c] = pd.NA
    df = df[NORMALIZED_COLUMNS].sort_values("concurso").reset_index(drop=True)
    for c in [f"o{i}" for i in range(1, 7)] + [f"b{i}" for i in range(1, 7)]:
        df[c] = df[c].astype("Int64")
    return df


# ----------------------------------------------------------------------------
# Parsers (um por formato)
# ----------------------------------------------------------------------------

def parse_caixa_xlsx(path: Path) -> pd.DataFrame:
    x = pd.read_excel(path)
    rows = []
    for _, r in x.iterrows():
        rows.append((r["Concurso"], _br_date(r["Data do Sorteio"]),
                     [r[f"Bola{i}"] for i in range(1, 7)], None))
    return _frame(rows)


def parse_caixa_xlsx_prizes(path: Path) -> pd.DataFrame:
    """Dados de premiação da planilha oficial (usados no teste de fatores)."""
    x = pd.read_excel(path)

    def brl(v) -> float:
        s = str(v).replace("R$", "").replace(".", "").replace(",", ".").strip()
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return pd.DataFrame({
        "concurso": x["Concurso"].astype(int),
        "ganhadores_6": x["Ganhadores 6 acertos"].astype(int),
        "rateio_6": x["Rateio 6 acertos"].map(brl),
        "ganhadores_5": x["Ganhadores 5 acertos"].astype(int),
        "rateio_5": x["Rateio 5 acertos"].map(brl),
        "ganhadores_4": x["Ganhadores 4 acertos"].astype(int),
        "rateio_4": x["Rateio 4 acertos"].map(brl),
        "acumulado_6": x["Acumulado 6 acertos"].map(brl),
        "arrecadacao": x["Arrecadação Total"].map(brl),
        "estimativa_premio": x["Estimativa prêmio"].map(brl),
    })


def parse_asloterias_csv(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    rows = [(r["Concurso"], _br_date(r["Data"]),
             [r[f"bola {i}"] for i in range(1, 7)], None) for _, r in x.iterrows()]
    return _frame(rows)


def parse_megaloterias_rpkg(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    rows = [(r["concurso"], _iso_date(r["data"]),
             [r[f"dezena_{i}"] for i in range(1, 7)], None) for _, r in x.iterrows()]
    return _frame(rows)


def parse_salles_json(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [(d["concurso"], _br_date(d["data"]), d["dezenas"], None) for d in data]
    return _frame(rows)


def parse_osjunnior_csv(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    rows = [(r["concurso"], _iso_date(r["data"]),
             [r[f"n{i}"] for i in range(1, 7)], None) for _, r in x.iterrows()]
    return _frame(rows)


def parse_megadata_order_csv(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, encoding="utf-8-sig")
    rows = []
    for _, r in x.iterrows():
        order = [int(r[f"bola {i}"]) for i in range(1, 7)]
        rows.append((r["Concurso"], _br_date(r["Data"]), order, order))
    return _frame(rows)


def parse_guilhermeasn_json(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for k, v in data.items():
        order = [int(s) for s in v]
        rows.append((int(k), "", order, order))
    return _frame(rows)


def parse_lotolab_json(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [(d["concurso"], _br_date(d["data"]), [int(s) for s in d["dezenas"]], None)
            for d in data]
    return _frame(rows)


def parse_rust_spider_json(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [(c[0], _iso_date(c[1]), c[2], None) for c in data["concursos"]]
    return _frame(rows)


# ----------------------------------------------------------------------------
# Registro das fontes
# ----------------------------------------------------------------------------

SOURCES: dict[str, Source] = {s.id: s for s in [
    Source(
        id="caixa_planilha_oficial",
        description="Planilha 'Mega-Sena.xlsx' exportada de loterias.caixa.gov.br (via luizvalle/mega_sena_analysis)",
        url=f"{RAW_GH}/dcshodai29-web/mega_sena_analysis/1524922573aceeaa0f6d6ac60b4bab6f7d4a04e7/data/Mega-Sena.xlsx",
        filename="Mega-Sena.xlsx",
        sha256="0a0a8b49a3eed74dad481ee0c4aefc17505953a50405a2a4338df81061a1beae",
        license="dados públicos da Caixa; repositório MIT",
        origin="caixa",
        has_order=False,
        parser=parse_caixa_xlsx,
    ),
    Source(
        id="asloterias_csv",
        description="CSV baixado de asloterias.com.br (luizvalle/mega_sena_analysis)",
        url=f"{RAW_GH}/dcshodai29-web/mega_sena_analysis/1524922573aceeaa0f6d6ac60b4bab6f7d4a04e7/data/mega_sena_draw_results.csv",
        filename="mega_sena_draw_results.csv",
        sha256="14627b18ebb7875b60490361dd7038fddfc4f30ce23d4d801211cea515abae70",
        license="repositório MIT",
        origin="asloterias",
        has_order=False,
        parser=parse_asloterias_csv,
    ),
    Source(
        id="megaloterias_rpkg",
        description="Pacote R loteria (damarals → McGuyverBR), raspagem de megaloterias.com.br",
        url=f"{RAW_GH}/dcshodai29-web/loteria/9ef993aca29ed97523d9a75793c93c53a20f9193/inst/extdata/megasena.csv",
        filename="megasena.csv",
        sha256="baf8f83c39925165e32434cbdfd871e4defb6f37d94bb31e75a38115d923486e",
        license="MIT",
        origin="megaloterias",
        has_order=False,
        parser=parse_megaloterias_rpkg,
    ),
    Source(
        id="caixa_api_salles",
        description="Exportação JSON da API oficial da Caixa (salles97/MegaSena)",
        url=f"{RAW_GH}/dcshodai29-web/MegaSena/d7a8f6b76cae0c432dfd35e80b1ba1190f8626a9/sorteios.json",
        filename="sorteios.json",
        sha256="3c8c3136db0af31d4673cdc6d5786743309e212dc367631db1cf187949f30e82",
        license="sem licença (apenas fatos extraídos)",
        origin="caixa",
        has_order=False,
        parser=parse_salles_json,
    ),
    Source(
        id="osjunnior_csv",
        description="CSV de 506 concursos (OsJunnior/mega_sena)",
        url=f"{RAW_GH}/dcshodai29-web/mega_sena/026c687eb1ef52a570cb75a4d8ca1a9c05f38d21/dados/megasena.csv",
        filename="megasena.csv",
        sha256="9af9cfe60f3b6b985e4386bdb1a874548703d9f1f8715ce4a0a4f3b29bc24ccf",
        license="sem licença (apenas fatos extraídos)",
        origin="desconhecida",
        has_order=False,
        parser=parse_osjunnior_csv,
    ),
    Source(
        id="asloterias_ordem",
        description="CSV com ordem de extração, origem asloterias.com.br (RoninXK/megadata-megasena)",
        url=f"{RAW_GH}/dcshodai29-web/megadata-megasena/5531606841944a3ec4e38de97dacf10cc1bdc631/data/processed/mega_sena_tratada.csv",
        filename="mega_sena_tratada.csv",
        sha256="81005c0af2fde4ae50203be01f63a866f6ff9518d79aa634610f4bca7d9bdaf3",
        license="sem licença (apenas fatos extraídos)",
        origin="asloterias",
        has_order=True,
        parser=parse_megadata_order_csv,
    ),
    Source(
        id="guilhermeasn_ordem",
        description="JSON com ordem de extração (guilhermeasn/loteria.json)",
        url=f"{RAW_GH}/guilhermeasn/loteria.json/master/data/megasena.json",
        filename="megasena.json",
        sha256="07506c8b18b77365935b6191565c81b002a1baa6c7313480c517cd84d71f6355",
        license="não informada (apenas fatos extraídos)",
        origin="desconhecida",
        has_order=True,
        parser=parse_guilhermeasn_json,
    ),
    Source(
        id="lotolab_api",
        description="JSON coletado da API da Caixa (faelw/lotolab-megasena)",
        url=f"{RAW_GH}/faelw/lotolab-megasena/main/dados_loterias/megasena_todos.json",
        filename="megasena_todos.json",
        sha256="6300be447e4c9c7b07c662dcf53e3a7977b532e8e2b606e277a0b9e3e4f21125",
        license="sem licença (apenas fatos extraídos)",
        origin="caixa",
        has_order=False,
        parser=parse_lotolab_json,
        live=True,
    ),
    Source(
        id="rust_spider",
        description="JSON de histórico (edxgar48/rust_spider_dados)",
        url=f"{RAW_GH}/edxgar48/rust_spider_dados/main/v1/megasena/historico.json",
        filename="historico.json",
        sha256="b6d04243852e0685e171ca2426027f4ef17460797e5fef78db73bf8b4e921505",
        license="não informada (apenas fatos extraídos)",
        origin="desconhecida",
        has_order=False,
        parser=parse_rust_spider_json,
        live=True,
    ),
]}


# Extratos sem arquivo bruto fixo (gerados por `megalab data update` / `add-result`)
EXTRA_ORIGINS = {"caixa_api_live": "caixa", "manual": "manual"}


def origin_of(source_id: str) -> str:
    if source_id in SOURCES:
        return SOURCES[source_id].origin
    return EXTRA_ORIGINS.get(source_id, "desconhecida")


# ----------------------------------------------------------------------------
# Download / extração
# ----------------------------------------------------------------------------

def raw_path(src: Source) -> Path:
    return RAW_CACHE_DIR / src.id / src.filename


def _download(url: str, dest: Path, timeout: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "megalab/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def manifest_hashes() -> dict[str, str]:
    p = SOURCES_DIR / "MANIFEST.json"
    if not p.exists():
        return {}
    return {f["id"]: f["sha256_bruto"] for f in json.loads(p.read_text(encoding="utf-8"))["fontes"]}


def fetch(src: Source, local_copy: Path | None = None, timeout: int = 60,
          refresh: bool = False) -> Path:
    """Obtém o arquivo bruto (cópia local ou download) e confere o SHA-256.

    Fontes fixadas em commit precisam bater com o hash registrado. Fontes
    ``live`` (URL de branch) podem ser re-baixadas com ``refresh=True``; o novo
    hash fica registrado no MANIFEST e passa a ser o esperado.
    """
    dest = raw_path(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if refresh and src.live:
        _download(src.url, dest, timeout)
        return dest
    if not dest.exists():
        if local_copy is not None and Path(local_copy).exists():
            shutil.copyfile(local_copy, dest)
        else:
            _download(src.url, dest, timeout)
    got = sha256_file(dest)
    expected = manifest_hashes().get(src.id, src.sha256) if src.live else src.sha256
    if got != expected and got != src.sha256:
        raise ValueError(
            f"SHA-256 divergente para {src.id}: esperado {src.sha256}, obtido {got}. "
            "A fonte mudou desde o registro; revise antes de atualizar o hash."
        )
    return dest


def extract(src: Source, raw: Path) -> pd.DataFrame:
    df = src.parser(raw)
    df.insert(0, "fonte", src.id)
    return df


def extract_path(source_id: str) -> Path:
    return SOURCES_DIR / f"{source_id}.csv"


def write_manifest(extracts: dict[str, pd.DataFrame], raw_hashes: dict[str, str] | None = None,
                   retrieved: dict[str, str] | None = None) -> dict:
    raw_hashes = raw_hashes or {}
    retrieved = retrieved or {}
    manifest = {"gerado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "fontes": []}
    for sid, df in extracts.items():
        s = SOURCES[sid]
        manifest["fontes"].append({
            "id": s.id, "descricao": s.description, "url": s.url,
            "sha256_bruto": raw_hashes.get(sid, s.sha256),
            "licenca": s.license, "origem": s.origin, "tem_ordem": s.has_order,
            "obtido_em": retrieved.get(sid, s.retrieved), "concursos": int(len(df)),
            "primeiro": int(df["concurso"].min()), "ultimo": int(df["concurso"].max()),
            "sha256_extrato": sha256_file(extract_path(sid)),
        })
    (SOURCES_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_extracts() -> dict[str, pd.DataFrame]:
    """Todos os extratos em ``data/sources/*.csv`` (fontes registradas + extras)."""
    out = {}
    for p in sorted(SOURCES_DIR.glob("*.csv")):
        df = pd.read_csv(p, dtype={"data": str}, keep_default_na=False, na_values=[""])
        df["data"] = df["data"].fillna("")
        for c in [f"o{i}" for i in range(1, 7)] + [f"b{i}" for i in range(1, 7)]:
            df[c] = df[c].astype("Int64")
        out[p.stem] = df
    return out


def write_extract(source_id: str, df: pd.DataFrame) -> Path:
    """Grava um extrato normalizado (colunas padrão, ordenado por concurso)."""
    df = df.copy()
    df["fonte"] = source_id
    df = df[["fonte"] + NORMALIZED_COLUMNS].drop_duplicates("concurso", keep="last")
    df = df.sort_values("concurso")
    path = extract_path(source_id)
    df.to_csv(path, index=False)
    return path
