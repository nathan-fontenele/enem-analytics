"""Carregamento, métricas e análises estatísticas (pandas / numpy / statsmodels)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

# paleta de referência (dataviz): azul, laranja, verde-água, amarelo
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#dbe8f9", "#9dc0ee", "#5b97e0", "#2a78d6", "#1a4f93"]

REGIAO = {**{u: "Norte" for u in "AC AP AM PA RO RR TO".split()},
          **{u: "Nordeste" for u in "AL BA CE MA PB PE PI RN SE".split()},
          **{u: "Centro-Oeste" for u in "DF GO MT MS".split()},
          **{u: "Sudeste" for u in "ES MG RJ SP".split()},
          **{u: "Sul" for u in "PR RS SC".split()}}
COR = {0: "Não declarado", 1: "Branca", 2: "Preta", 3: "Parda", 4: "Amarela", 5: "Indígena", 6: "Sem info"}
ST = {0: "n/d", 1: "Já concluiu o EM", 2: "Conclui em 2º ano corrente", 3: "Conclui depois", 4: "Não cursa EM", -1: "n/d (2024)"}
ST_SHORT = {1: "Egresso", 2: "Concluinte do ano", 3: "Cursando (conclui depois)", 4: "Sem EM"}
RENDA_ORD = ["Sem renda", "Até 1 SM", "1 a 2 SM", "2 a 3 SM", "3 a 5 SM", "5 a 10 SM", "Mais de 10 SM"]
RENDA_MID = {"Sem renda": 0, "Até 1 SM": .5, "1 a 2 SM": 1.5, "2 a 3 SM": 2.5, "3 a 5 SM": 4, "5 a 10 SM": 7.5, "Mais de 10 SM": 14}
ESC_ORD = ["Até fund. incompleto", "Fund. completo / médio incompleto", "Médio completo", "Superior", "Não sabe"]
SM = {2020: 1045, 2021: 1100, 2022: 1212, 2023: 1320, 2024: 1412}
NOTA_NOME = {"CN": "Ciências da Natureza", "CH": "Ciências Humanas", "LC": "Linguagens", "MT": "Matemática",
             "RED": "Redação", "GERAL": "Média geral"}
COMP_NOME = {1: "C1 Norma culta", 2: "C2 Tema/tipo textual", 3: "C3 Argumentação", 4: "C4 Coesão", 5: "C5 Proposta de intervenção"}


def faixa_label(f: int) -> str:
    if f <= 2: return "Até 17 anos"
    if f == 3: return "18 anos"
    if f <= 5: return "19-20 anos"
    if f <= 10: return "21-25 anos"
    if f == 11: return "26-30 anos"
    return "31+ anos"
FAIXA_ORD = ["Até 17 anos", "18 anos", "19-20 anos", "21-25 anos", "26-30 anos", "31+ anos"]


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


def have_data() -> bool:
    return (DATA / "cubo_resultados.parquet").exists() and (DATA / "red_discordancia.parquet").exists()


# ------------------------------------------------------------- métricas
def mean_of(d: pd.DataFrame, k: str) -> pd.Series:
    return d[f"sum_{k}"] / d[f"cnt_{k}"]


def sd_of(d: pd.DataFrame, k: str) -> pd.Series:
    n, s, ss = d[f"cnt_{k}"], d[f"sum_{k}"], d[f"ssq_{k}"]
    return np.sqrt(((ss - s * s / n) / (n - 1)).clip(lower=0))


def summarize(res: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Agrega o cubo de resultados e devolve médias, desvios e taxas."""
    num = [c for c in res.columns if c not in ("ano", "uf", "dep", "renda", "st")]
    g = res.groupby(by, observed=True)[num].sum().reset_index()
    g["abst_d1"] = 1 - g.pres_d1 / g.n_insc
    g["comp_2dias"] = g.pres_all / g.n_insc
    for k in ("CN", "CH", "LC", "MT", "RED", "GERAL"):
        g["m_" + k] = mean_of(g, k)
        g["sd_" + k] = sd_of(g, k)
    g["red_p700"] = g.red_ge700 / g.cnt_RED
    g["red_p900"] = g.red_ge900 / g.cnt_RED
    g["red_p1000"] = g.red_1000 / g.cnt_RED
    g["red_pzero"] = g.red_zero / g.cnt_RED
    g["ger_p600"] = g.ger_ge600 / g.cnt_GERAL
    g["ger_p700"] = g.ger_ge700 / g.cnt_GERAL
    for c in range(1, 6):
        g[f"m_c{c}"] = g[f"sumc{c}"] / g[f"cntc{c}"]
    return g


def pct(x, d=1):
    return f"{x*100:.{d}f}%".replace(".", ",")


def num_br(x, d=0):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def short(x):
    x = float(x)
    if abs(x) >= 1e9: return f"{x/1e9:.1f} bi".replace(".", ",")
    if abs(x) >= 1e6: return f"{x/1e6:.2f} mi".replace(".", ",")
    if abs(x) >= 1e3: return f"{x/1e3:.0f} mil"
    return f"{x:.0f}"


def brl(x):
    return "R$ " + short(x) if abs(x) >= 1e3 else f"R$ {x:,.0f}"


# ------------------------------------------------------- statsmodels
@st.cache_data(show_spinner="Ajustando modelos (statsmodels)…")
def fit_ols_nota() -> dict:
    """OLS: média geral ~ renda, escolaridade dos pais, conectividade, rede etc. (2020-2023, presentes, sem treineiros)."""
    d = load("amostra_regressao").copy()
    d["idade_f"] = d.faixa.map(faixa_label)
    d = d.dropna(subset=["renda_sm", "esc_mae_ord", "esc_pai_ord"])
    d = d[d.escola_auto.isin(["Pública", "Privada"])]
    d["privada"] = (d.escola_auto == "Privada").astype(int)
    d["mulher"] = (d.sexo == "F").astype(int)
    d["cor_g"] = d.cor.map({1: "Branca", 2: "Preta", 3: "Parda"}).fillna("Outras")
    d["log_renda"] = np.log1p(d.renda_sm)
    f = ("n_GERAL ~ log_renda + esc_mae_ord + esc_pai_ord + internet + computador + privada + mulher "
         "+ C(cor_g, Treatment('Branca')) + C(idade_f, Treatment('18 anos')) + C(st) + C(ano)")
    m = smf.ols(f, data=d).fit(cov_type="HC1")
    # logit: probabilidade de redação >= 700
    d["red700"] = (d.n_RED >= 700).astype(int)
    f2 = ("red700 ~ log_renda + esc_mae_ord + esc_pai_ord + internet + computador + privada + mulher "
          "+ C(cor_g, Treatment('Branca')) + C(st) + C(ano)")
    lg = smf.logit(f2, data=d.sample(min(len(d), 250_000), random_state=1)).fit(disp=0)
    # contribuição marginal de cada bloco (queda do R² ao retirar)
    blocos = {"Renda familiar": "log_renda", "Escolaridade dos pais": "esc_mae_ord + esc_pai_ord",
              "Conectividade (internet/PC)": "internet + computador", "Rede privada": "privada"}
    base_terms = ["log_renda", "esc_mae_ord", "esc_pai_ord", "internet", "computador", "privada", "mulher",
                  "C(cor_g, Treatment('Branca'))", "C(idade_f, Treatment('18 anos'))", "C(st)", "C(ano)"]
    r2_full = m.rsquared
    sub = d.sample(min(len(d), 150_000), random_state=2)
    r2_blocks = {}
    full_sub = smf.ols("n_GERAL ~ " + " + ".join(base_terms), data=sub).fit().rsquared
    for nome, termos in blocos.items():
        drop = {t.strip() for t in termos.split("+")}
        rest = [t for t in base_terms if t not in drop]
        r2_blocks[nome] = full_sub - smf.ols("n_GERAL ~ " + " + ".join(rest), data=sub).fit().rsquared
    return {"ols": m, "logit": lg, "n": int(m.nobs), "r2": r2_full, "r2_blocks": r2_blocks,
            "mean_nota": float(d.n_GERAL.mean())}


def trend_fit(x: np.ndarray, y: np.ndarray, new_x: list[int], alpha=0.2):
    """Tendência linear (OLS) com intervalo de previsão. n pequeno => usar como cenário, não como previsão."""
    X = sm.add_constant(np.asarray(x, float))
    r = sm.OLS(np.asarray(y, float), X).fit()
    Xn = sm.add_constant(np.asarray(new_x, float), has_constant="add")
    pr = r.get_prediction(Xn).summary_frame(alpha=alpha)
    return r, pr


def cagr(a: float, b: float, years: int) -> float:
    return (b / a) ** (1 / years) - 1 if a > 0 and years > 0 else np.nan


def welch_d(m1, s1, n1, m2, s2, n2):
    """Diferença de médias, estatística t de Welch e d de Cohen a partir de resumos."""
    diff = m1 - m2
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return diff, diff / se, diff / sp


# mapa de blocos (tile map) do Brasil: (coluna, linha)
TILE = {"RR": (2, 0), "AP": (4, 0), "AM": (1, 1), "PA": (3, 1), "MA": (4, 1), "CE": (5, 1), "RN": (6, 1),
        "AC": (0, 2), "RO": (1, 2), "MT": (2, 2), "TO": (3, 2), "PI": (4, 2), "PE": (5, 2), "PB": (6, 2),
        "MS": (2, 3), "GO": (3, 3), "BA": (4, 3), "AL": (5, 3), "SE": (6, 3),
        "DF": (3, 4), "MG": (4, 4), "ES": (5, 4), "PR": (2, 5), "SP": (3, 5), "RJ": (4, 5), "SC": (2, 6), "RS": (2, 7)}
