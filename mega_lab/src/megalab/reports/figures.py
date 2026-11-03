"""Figuras do relatório (matplotlib, PNG estático).

Convenções: uma cor de destaque (azul) para a série principal, cinzas para
referência/nula, grade recessiva, um único eixo y por gráfico, legenda quando
há mais de uma série.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BAND = "#d9d8d3"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "legend.frameon": False,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def frequencies(scan: list[dict], n: int, path: Path):
    """Desvio de cada dezena em relação ao esperado (barras a partir de zero)."""
    _style()
    f = np.array([r["frequencia"] for r in scan])
    E = n * 0.1
    sd = np.sqrt(n * 0.1 * 0.9)
    dev = f - E
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(1, 61)
    ax.axhspan(-1.96 * sd, 1.96 * sd, color=BAND, alpha=0.6, lw=0,
               label="faixa de 95 % sob o acaso (por dezena)")
    ax.bar(x, dev, width=0.7, color=BLUE, label="observado − esperado")
    ax.axhline(0, color=INK2, lw=1)
    lim = max(np.abs(dev).max(), 3 * sd) * 1.15
    ax.set_ylim(-lim, lim)
    ax.set_xlim(0.3, 60.7)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    ax.set_xlabel("dezena")
    ax.set_ylabel(f"vezes sorteada − {E:.0f}")
    ax.set_title(f"Desvio de cada dezena em relação ao esperado ({n} concursos, esperado {E:.0f})")
    for i in np.argsort(np.abs(dev))[-4:]:
        off = 3 if dev[i] > 0 else -10
        ax.annotate(f"{i + 1:02d}: {f[i]}", (x[i], dev[i]), xytext=(0, off), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    _save(fig, path)


def hypothesis_pvalues(hyps: list[dict], path: Path, alpha: float = 0.05):
    _style()
    ids = [h["id"] for h in hyps][::-1]
    pdisc = np.array([h["p_descoberta"] or np.nan for h in hyps][::-1], dtype=float)
    pconf = np.array([h["p_confirmacao"] or np.nan for h in hyps][::-1], dtype=float)
    y = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.axvline(-np.log10(alpha), color=INK2, lw=1, ls="--")
    ax.axvline(-np.log10(alpha / len(ids)), color=MUTED, lw=1, ls=":")
    ax.text(-np.log10(alpha), len(ids) - 0.3, " α = 0,05", color=INK2, fontsize=8, va="bottom")
    ax.text(-np.log10(alpha / len(ids)), len(ids) - 0.3, " Bonferroni", color=MUTED, fontsize=8, va="bottom")
    ax.scatter(-np.log10(pdisc), y + 0.15, s=36, color=BLUE, label="descoberta (1–2000)", zorder=3,
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(-np.log10(pconf), y - 0.15, s=36, color=ORANGE, marker="s", label="confirmação (2001–)",
               zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(ids)
    ax.set_xlabel("−log10(p)   (mais à direita = mais anômalo)")
    ax.set_title("Hipóteses pré-registradas: p-valores de Monte Carlo")
    ax.set_ylim(-0.7, len(ids) + 0.4)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def null_distributions(null: dict, observed: dict, labels: dict, keys: list[str], path: Path):
    _style()
    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows))
    for ax, k in zip(np.ravel(axes), keys):
        v = np.asarray(null[k], dtype=float)
        v = v[np.isfinite(v)]
        ax.hist(v, bins=40, color=BAND, edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(observed[k], color=BLUE, lw=2)
        p = (1 + (v >= observed[k]).sum()) / (1 + len(v))
        ax.set_title(f"{k}  p = {p:.3f}".replace(".", ","), fontsize=10)
        ax.set_xlabel(labels.get(k, ""), fontsize=8)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in np.ravel(axes)[len(keys):]:
        ax.axis("off")
    fig.suptitle("Dados reais (linha azul) contra 1.000 históricos simulados perfeitamente justos (cinza)",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def power_curve(power: list[dict], n: int, path: Path):
    _style()
    w = [r["peso"] for r in power]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhline(0.8, color=MUTED, lw=1, ls=":")
    ax.text(w[0], 0.81, "80 % de poder", color=MUTED, fontsize=8)
    ax.plot(w, [r["poder_H02"] for r in power], color=BLUE, lw=2, marker="o", ms=6, label="H02 (max |z|)")
    ax.plot(w, [r["poder_H01"] for r in power], color=ORANGE, lw=2, marker="s", ms=6, label="H01 (χ² global)")
    ax.set_xlabel("peso relativo de UMA bola viciada (1,0 = justa)")
    ax.set_ylabel("probabilidade de detecção")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Que viés físico a base de {n} concursos consegue detectar?")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def backtest_hits(period: dict, path: Path, title: str):
    _style()
    models = list(period["modelos"])
    mean = np.array([period["modelos"][m]["acertos_media"] for m in models])
    order = np.argsort(mean)
    n = period["n"]
    rp = period.get("jogadores_aleatorios")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(models))
    if rp:
        ax.axvspan(rp["p2_5"] / n, rp["p97_5"] / n, color=BAND, alpha=0.7, lw=0,
                   label="95 % de 1.000 jogadores aleatórios")
    ax.axvline(0.6, color=INK2, lw=1, ls="--", label="esperado sob o acaso = 0,6")
    ax.scatter(mean[order], y, s=40, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5,
               label="modelo")
    lo = min(mean.min(), rp["p2_5"] / n if rp else mean.min())
    hi = max(mean.max(), rp["p97_5"] / n if rp else mean.max())
    pad = (hi - lo) * 0.25
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_yticks(y)
    ax.set_yticklabels([models[i] for i in order])
    ax.set_xlabel("acertos médios por concurso (bilhete de 6 dezenas)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    _save(fig, path)


def evalue_paths(frame, path: Path, thresholds=(20, 240), mix: float = 0.5, floor: float = -12):
    _style()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    below = []
    for m, g in frame.groupby("modelo"):
        if m == "aleatorio":
            continue
        lr = np.power(2.0, g["bits"].to_numpy())
        ev = np.cumsum(np.log10((1 - mix) + mix * lr))
        c = g["concurso"].to_numpy()
        hl = m == "hedge"
        ax.plot(c, np.maximum(ev, floor - 1), color=BLUE if hl else MUTED, lw=2 if hl else 1,
                alpha=1 if hl else 0.7, zorder=3 if hl else 2)
        if hl:
            ax.annotate("hedge", (c[-1], ev[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center")
        if ev.min() < floor:
            below.append(m)
    for t, lab in zip(thresholds, ("e = 20  (α = 5 %, um modelo)", "e = 240  (α = 5 %, 12 modelos)")):
        ax.axhline(np.log10(t), color=INK2, lw=1, ls="--")
        ax.text(frame["concurso"].min(), np.log10(t) + 0.08, lab, color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(floor, 3.2)
    ax.set_xlabel("concurso")
    ax.set_ylabel("log10 do e-valor acumulado")
    ax.set_title("Evidência acumulada contra o acaso (cinza: cada modelo; azul: hedge)")
    if below:
        ax.text(frame["concurso"].min(), floor + 0.3, "abaixo do quadro: " + ", ".join(below),
                color=INK2, fontsize=8)
    _save(fig, path)


def changepoint(profile: list[float], concursos_cut: np.ndarray, null_p95: float, best_cut: int, path: Path):
    _style()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axhline(null_p95, color=INK2, lw=1, ls="--")
    ax.text(concursos_cut[0], null_p95 * 1.01, "95 % do máximo sob o acaso", color=INK2, fontsize=8, va="bottom")
    ax.plot(concursos_cut, profile, color=BLUE, lw=2)
    ax.set_xlabel("concurso de corte (antes × depois)")
    ax.set_ylabel("χ² de homogeneidade")
    ax.set_title(f"Varredura de mudança de regime — maior contraste no concurso {best_cut}")
    _save(fig, path)
