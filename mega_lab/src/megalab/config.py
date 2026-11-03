"""Constantes e parâmetros pré-registrados do MEGA LAB.

Qualquer alteração aqui altera o protocolo (docs/03_protocolo_cientifico.md)
e deve ser feita em commit próprio, com justificativa.
"""

from __future__ import annotations

from pathlib import Path

# --- Regras do jogo ---------------------------------------------------------
N_NUMBERS = 60          # dezenas possíveis (1..60)
K_DRAWN = 6             # dezenas sorteadas por concurso
P_NUMBER = K_DRAWN / N_NUMBERS  # probabilidade marginal de cada dezena = 0,1

# --- Caminhos -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"
CANONICAL_DIR = DATA_DIR / "canonical"
RAW_CACHE_DIR = DATA_DIR / "raw_cache"
CANONICAL_CSV = CANONICAL_DIR / "megasena.csv"
PRIZES_CSV = CANONICAL_DIR / "premiacao.csv"
REGISTRY_DIR = ROOT / "registry"
LEDGER_PATH = REGISTRY_DIR / "ledger.jsonl"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Períodos pré-registrados ----------------------------------------------------
DISCOVERY = (1, 2000)            # Família 1: descoberta
CONFIRMATION_START = 2001        # Família 1: confirmação (até o último verificado)
BACKTEST_WARMUP_END = 100
BACKTEST_PERIODS = {
    "desenvolvimento": (101, 2000),
    "validacao": (2001, 2600),
    "holdout": (2601, None),     # None = último concurso verificado
}

# --- Estatística -------------------------------------------------------------------
ALPHA = 0.05
MC_REPLICATES = 1000             # B para p-valores de Monte Carlo
PERMUTATIONS = 1000              # B para testes de permutação
SEED = 20261103                  # semente mestre (data do pré-registro)

# --- Modelos -------------------------------------------------------------------------
HEURISTIC_SHRINK = 0.5           # encolhimento em direção a 0,1
EVALUE_MIX = 0.5                 # mistura com o uniforme no e-valor
ML_REFIT_EVERY = 100
DIRICHLET_PRIOR = 50.0
N_MODELS_FAMILY2 = 12            # M01..M12 (sem contar a linha de base)
PROSPECTIVE_EVALUE_THRESHOLD = N_MODELS_FAMILY2 / ALPHA  # = 240
