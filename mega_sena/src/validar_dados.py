"""
validar_dados.py

Executa todas as checagens de qualidade exigidas antes de qualquer
análise. Nada aqui "conserta" dados silenciosamente: cada problema é
reportado explicitamente em um relatório (dict), e uma versão corrigida
só é gerada quando isso é seguro e documentado (ex: remover uma linha
exatamente duplicada é seguro; "adivinhar" uma dezena faltante não é).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

COLUNAS_DEZENAS = ["n1", "n2", "n3", "n4", "n5", "n6"]
DEZENA_MIN, DEZENA_MAX = 1, 60


@dataclass
class RelatorioValidacao:
    total_linhas: int = 0
    ok: bool = True
    problemas: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    concursos_faltantes: list[int] = field(default_factory=list)
    linhas_duplicadas: int = 0
    concursos_duplicados: list[int] = field(default_factory=list)
    valores_ausentes_por_coluna: dict[str, int] = field(default_factory=dict)
    linhas_com_dezena_fora_do_intervalo: int = 0
    linhas_com_dezena_repetida: int = 0
    linhas_com_datas_invalidas: int = 0
    fora_de_ordem_cronologica: bool = False

    def registrar_problema(self, msg: str) -> None:
        self.problemas.append(msg)
        self.ok = False

    def registrar_aviso(self, msg: str) -> None:
        self.avisos.append(msg)

    def para_texto(self) -> str:
        linhas = [
            "# Relatório de validação dos dados",
            "",
            f"Total de linhas lidas: {self.total_linhas}",
            f"Status geral: {'OK (sem problemas bloqueantes)' if self.ok else 'PROBLEMAS ENCONTRADOS'}",
            "",
        ]
        if self.problemas:
            linhas.append("## Problemas (bloqueantes ou que exigem atenção)")
            linhas += [f"- {p}" for p in self.problemas]
            linhas.append("")
        if self.avisos:
            linhas.append("## Avisos (não bloqueiam a análise, mas o usuário deve saber)")
            linhas += [f"- {a}" for a in self.avisos]
            linhas.append("")
        linhas.append("## Detalhes")
        linhas.append(f"- Linhas duplicadas (idênticas): {self.linhas_duplicadas}")
        linhas.append(f"- Concursos duplicados (mesmo número, dados diferentes ou não): {self.concursos_duplicados}")
        linhas.append(f"- Concursos faltantes na sequência: {len(self.concursos_faltantes)} "
                       f"(exemplos: {self.concursos_faltantes[:15]})")
        linhas.append(f"- Valores ausentes por coluna: {self.valores_ausentes_por_coluna}")
        linhas.append(f"- Linhas com dezena fora de 1-60: {self.linhas_com_dezena_fora_do_intervalo}")
        linhas.append(f"- Linhas com dezena repetida no mesmo concurso: {self.linhas_com_dezena_repetida}")
        linhas.append(f"- Linhas com data inválida: {self.linhas_com_datas_invalidas}")
        linhas.append(f"- Fora de ordem cronológica (precisou reordenar): {self.fora_de_ordem_cronologica}")
        return "\n".join(linhas)


def validar(df: pd.DataFrame) -> RelatorioValidacao:
    """Roda todas as checagens pedidas e devolve um relatório estruturado."""
    rel = RelatorioValidacao(total_linhas=len(df))

    # 1. valores ausentes
    for col in ["concurso", "data"] + COLUNAS_DEZENAS:
        n_na = int(df[col].isna().sum())
        if n_na:
            rel.valores_ausentes_por_coluna[col] = n_na
            rel.registrar_problema(f"Coluna '{col}' tem {n_na} valor(es) ausente(s).")

    # 2. datas inválidas (viraram NaT na conversão)
    n_data_invalida = int(df["data"].isna().sum())
    rel.linhas_com_datas_invalidas = n_data_invalida
    if n_data_invalida:
        rel.registrar_problema(f"{n_data_invalida} linha(s) com data que não pôde ser interpretada.")

    # 3. dezenas fora do intervalo 1-60
    fora_intervalo = pd.Series(False, index=df.index)
    for c in COLUNAS_DEZENAS:
        fora_intervalo |= df[c].apply(lambda v: pd.notna(v) and not (DEZENA_MIN <= v <= DEZENA_MAX))
    rel.linhas_com_dezena_fora_do_intervalo = int(fora_intervalo.sum())
    if rel.linhas_com_dezena_fora_do_intervalo:
        rel.registrar_problema(
            f"{rel.linhas_com_dezena_fora_do_intervalo} linha(s) com dezena fora do intervalo 1-{DEZENA_MAX}."
        )

    # 4. dezenas repetidas dentro do mesmo concurso
    def tem_repetida(row):
        vals = [row[c] for c in COLUNAS_DEZENAS if pd.notna(row[c])]
        return len(vals) != len(set(vals))

    repetida_mask = df.apply(tem_repetida, axis=1)
    rel.linhas_com_dezena_repetida = int(repetida_mask.sum())
    if rel.linhas_com_dezena_repetida:
        rel.registrar_problema(
            f"{rel.linhas_com_dezena_repetida} linha(s) com dezena repetida dentro do mesmo concurso."
        )

    # 5. cada concurso deve ter exatamente 6 dezenas não nulas -- já coberto por
    #    'valores ausentes' acima, mas deixamos explícito:
    qtd_dezenas_validas = df[COLUNAS_DEZENAS].notna().sum(axis=1)
    linhas_sem_6 = int((qtd_dezenas_validas != 6).sum())
    if linhas_sem_6:
        rel.registrar_problema(f"{linhas_sem_6} linha(s) não têm exatamente 6 dezenas preenchidas.")

    # 6. linhas duplicadas (idênticas em todas as colunas obrigatórias)
    cols_check = ["concurso", "data"] + COLUNAS_DEZENAS
    dup_mask = df.duplicated(subset=cols_check, keep=False)
    rel.linhas_duplicadas = int(dup_mask.sum())
    if rel.linhas_duplicadas:
        rel.registrar_aviso(
            f"{rel.linhas_duplicadas} linha(s) totalmente duplicadas encontradas "
            f"(mesmo concurso, data e dezenas). Podem ser removidas com segurança."
        )

    # 7. concursos duplicados (mesmo número em mais de uma linha, mesmo que dados difiram)
    contagem_concurso = df["concurso"].value_counts()
    concursos_dup = contagem_concurso[contagem_concurso > 1].index.dropna().astype(int).tolist()
    rel.concursos_duplicados = sorted(concursos_dup)
    if concursos_dup:
        rel.registrar_problema(
            f"{len(concursos_dup)} número(s) de concurso aparecem mais de uma vez: {concursos_dup[:20]}"
        )

    # 8. concursos faltantes na sequência
    concursos_validos = df["concurso"].dropna().astype(int)
    if len(concursos_validos):
        c_min, c_max = concursos_validos.min(), concursos_validos.max()
        esperados = set(range(c_min, c_max + 1))
        existentes = set(concursos_validos.tolist())
        faltantes = sorted(esperados - existentes)
        rel.concursos_faltantes = faltantes
        if faltantes:
            rel.registrar_aviso(
                f"{len(faltantes)} concurso(s) estão faltando na sequência {c_min}-{c_max}. "
                f"Isso é ESPERADO se o CSV não cobre o histórico completo, mas deve ser "
                f"confirmado contra a fonte oficial."
            )

    # 9. ordem cronológica
    datas_validas = df["data"].dropna()
    if not datas_validas.is_monotonic_increasing:
        rel.fora_de_ordem_cronologica = True
        rel.registrar_aviso(
            "Os dados não estão em ordem cronológica crescente. "
            "O pipeline reordena automaticamente antes de qualquer análise "
            "(nenhuma informação futura é usada para calcular estatísticas passadas)."
        )

    return rel


def limpar_com_seguranca(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica apenas correções seguras e documentadas:
    - remove linhas totalmente duplicadas;
    - reordena por concurso/data.

    NÃO remove concursos com problemas de conteúdo (dezena fora do
    intervalo, repetida, etc.) -- isso exige decisão humana, então essas
    linhas continuam no DataFrame retornado, mas o relatório de
    validação já apontou onde estão.
    """
    df = df.drop_duplicates(subset=["concurso", "data"] + COLUNAS_DEZENAS, keep="first")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df
