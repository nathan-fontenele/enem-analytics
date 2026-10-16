"""Dashboard ENEM 2020-2025 para análise de mercado de uma startup de educação.

Rodar:  .venv/bin/streamlit run app/dashboard.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

import lib
from lib import (C1, C2, C3, C4, COMP_NOME, COR, FAIXA_ORD, NOTA_NOME, REGIAO, RENDA_ORD, SEQ, TILE,
                 brl, cagr, faixa_label, load, num_br, pct, short, summarize, trend_fit)

st.set_page_config(page_title="Análise ENEM - 2020 - 2025", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:3.2rem;padding-bottom:3rem;max-width:1480px}
/* abas: texto simples com barra azul inferior na guia ativa (Streamlit >= 1.5x usa role="tab") */
[data-testid="stTabs"] [role="tablist"]{gap:8px !important;padding:0 !important;background:transparent !important;
  border:0 !important;border-bottom:1px solid rgba(128,128,128,.28) !important;box-shadow:none !important;flex-wrap:wrap !important}
[data-testid="stTabs"] [role="tablist"]::after,[data-testid="stTabs"] [role="tablist"]::before{display:none !important}
[data-testid="stTabs"] [role="tab"]{height:40px !important;padding:0 16px !important;margin:0 !important;border-radius:0 !important;
  background:transparent !important;border:0 !important;box-shadow:none !important;white-space:nowrap;transition:box-shadow .15s ease}
[data-testid="stTabs"] [role="tab"]::after,[data-testid="stTabs"] [role="tab"]::before{display:none !important}
[data-testid="stTabs"] [role="tab"] p{font-size:.95rem !important;font-weight:600 !important;opacity:.7}
[data-testid="stTabs"] [role="tab"]:hover{background:transparent !important;box-shadow:inset 0 -3px 0 rgba(42,120,214,.4) !important}
[data-testid="stTabs"] [role="tab"]:hover p{opacity:1}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{background:transparent !important;box-shadow:inset 0 -3px 0 #2a78d6 !important}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p{color:#2a78d6 !important;opacity:1}
[data-testid="stTabs"] [role="tabpanel"]{padding-top:1rem}
/* hero */
.hero{display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;
  padding:20px 26px;margin-bottom:14px;border-radius:18px;border:1px solid rgba(128,128,128,.25);
  background:linear-gradient(120deg,rgba(42,120,214,.16),rgba(27,175,122,.08) 60%,rgba(235,104,52,.07))}
.hero-t{font-size:1.75rem;font-weight:750;letter-spacing:-.01em}
.hero-s{opacity:.75;margin-top:2px}
.tag{display:inline-block;padding:3px 12px;margin-left:6px;border-radius:99px;font-size:.8rem;font-weight:600;
  background:rgba(42,120,214,.16);border:1px solid rgba(42,120,214,.3)}
.tag.warn{background:rgba(237,161,0,.15);border-color:rgba(237,161,0,.4)}
/* seções */
.sec{display:flex;align-items:center;gap:10px;margin:26px 0 8px 0}
.sec b{font-size:1.25rem}
.sec i{flex:1;height:1px;background:rgba(128,128,128,.28)}
.note{opacity:.68;font-size:.84rem}
.insight{padding:12px 16px;border-radius:12px;border-left:4px solid #2a78d6;background:rgba(42,120,214,.09);margin:6px 0;font-size:.95rem}
.insight.o{border-left-color:#eb6834;background:rgba(235,104,52,.09)}
.insight.g{border-left-color:#1baf7a;background:rgba(27,175,122,.09)}
/* slides */
.slide{border:1px solid rgba(128,128,128,.28);border-radius:20px;padding:30px 38px;min-height:440px;
  background:linear-gradient(135deg,rgba(42,120,214,.10),rgba(27,175,122,.05))}
.slide h2{margin:0;font-size:2.1rem;letter-spacing:-.01em}
.slide .sub{opacity:.7;margin:2px 0 22px 0;font-size:1.05rem}
.big{font-size:2.7rem;font-weight:800;line-height:1.05;color:#2a78d6}
.biglbl{opacity:.78;font-size:.92rem;margin:2px 0 12px 0;max-width:260px}
[data-testid="stMetric"]{border-radius:14px}
section[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.2)}
</style>""", unsafe_allow_html=True)

if not lib.have_data():
    st.error("Dados processados não encontrados. Rode primeiro:  `.venv/bin/python app/etl.py`")
    st.stop()


# ============================================================ helpers visuais
def style(fig, h=340, legend=True):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=10, b=6), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), showlegend=legend, separators=",.", hoverlabel=dict(font_size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font=dict(size=12)),
                      title=None)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(128,128,128,.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_traces(marker_cornerradius=5, selector=dict(type="bar"))
    return fig


def chart(fig, title, sub=None, h=340, legend=True, insight=None):
    with st.container(border=True):
        st.markdown(f"**{title}**" + (f"  \n<span class='note'>{sub}</span>" if sub else ""), unsafe_allow_html=True)
        st.plotly_chart(style(fig, h, legend), width="stretch")
        if insight:
            st.markdown(f"<div class='insight'>{insight}</div>", unsafe_allow_html=True)


def section(txt):
    st.markdown(f"<div class='sec'><b>{txt}</b><i></i></div>", unsafe_allow_html=True)


def tile_map(vals: pd.Series, fmt, h=470, colorscale=SEQ):
    """Mapa de blocos do Brasil (um bloco por UF) — sem depender de arquivos externos."""
    nx, ny = 7, 8
    z = np.full((ny, nx), np.nan)
    txt = [[""] * nx for _ in range(ny)]
    hov = [[""] * nx for _ in range(ny)]
    for uf, (x, y) in TILE.items():
        v = vals.get(uf, np.nan)
        if pd.notna(v):
            z[y][x] = v
            txt[y][x] = f"<b>{uf}</b><br>{fmt(v)}"
            hov[y][x] = f"<b>{uf}</b> · {REGIAO[uf]}<br>{fmt(v)}"
    fig = go.Figure(go.Heatmap(z=z, text=txt, texttemplate="%{text}", hovertext=hov, hoverinfo="text", colorscale=colorscale,
                               showscale=False, xgap=4, ygap=4, textfont=dict(size=12)))
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    return fig


def dec(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


# ===================================================== dados e filtros
RES, SOC, FAM, DEM, MUN = (load(n) for n in ("cubo_resultados", "cubo_socio", "cubo_familia", "cubo_demo", "cubo_municipio"))
ANOS = sorted(int(a) for a in SOC.ano.unique())
ANOS_RES = sorted(int(a) for a in RES.ano.unique())
UFS = sorted(u for u in SOC.uf.unique() if u != "n/d")
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def _reset():
    st.session_state["f_ano"] = ANOS[-1]
    st.session_state["f_reg"] = REGIOES
    st.session_state["f_uf"] = []


_KEEP_KEYS = ['pp_sc_zero', 'pp_sc_ano', 'pp_sc_x', 'pp_sc_y', 'pp_sc_min', 'anos_dor', 'esc_lista', 'esc_min', 'ex20', 'excl_tr', 'f_ano', 'f_reg', 'f_uf', 'map_geral', 'map_merc', 'mp_ano', 'mp_min', 'mp_nivel', 'mp_quad', 'mp_rede', 'mp_visao', 'mun_ano', 'mun_busca', 'mun_min', 'mun_minp', 'pp_ano', 'pp_det', 'pp_hist', 'renda_min', 'serie_grupo', 'serie_modo', 'serie_ufs', 'sim_anos', 'sim_cac', 'sim_conv', 'sim_g', 'sim_margem', 'sim_meses', 'sim_ticket', 'so_net', 'p1', 'p2', 'p3', 'p4', 'p5', 'serie_met_Mercado', 'serie_met_Desempenho', 'serie_met_Perfil']
# Streamlit descarta o estado de widgets que não foram renderizados na execução; como só a aba aberta roda,
# reatribuímos os valores para preservar os filtros ao trocar de aba.
for _k in _KEEP_KEYS:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

st.session_state.setdefault("f_ano", ANOS[-1])
st.session_state.setdefault("f_reg", REGIOES)
st.session_state.setdefault("f_uf", [])

with st.sidebar:
    st.markdown("### Análise ENEM - 2020 - 2025")
    st.markdown("**Período**")
    ano_ref = st.segmented_control("Ano de referência", ANOS, key="f_ano", required=True)
    cmp_opts = [a for a in ANOS if a != ano_ref]
    ano_cmp = st.selectbox("Comparar com", cmp_opts, index=cmp_opts.index(ano_ref - 1) if (ano_ref - 1) in cmp_opts else 0)
    st.markdown("**Geografia**")
    reg_sel = st.pills("Região", REGIOES, selection_mode="multi", key="f_reg") or REGIOES
    uf_opts = [u for u in UFS if REGIAO[u] in reg_sel]
    uf_pick = st.multiselect("UF (opcional)", uf_opts, key="f_uf", placeholder="Todas as UFs das regiões")
    ufs = uf_pick or uf_opts
    st.button("Limpar filtros", on_click=_reset, width="stretch")
    st.divider()
    st.caption("Fonte: microdados INEP/ENEM (+ competências da redação). Valores em R$ da aba Possibilidades são **hipóteses editáveis**.")

res = RES[RES.uf.isin(ufs)]
soc = SOC[SOC.uf.isin(ufs)]
ANO_RES = ano_ref if ano_ref in ANOS_RES else max([a for a in ANOS_RES if a <= ano_ref] or ANOS_RES)


# ---------------------------------------------------------- tabela anual
def _year_table(ufs_t: tuple) -> pd.DataFrame:
    s = SOC[SOC.uf.isin(ufs_t)]
    tot = s.groupby("ano").n_insc.sum()
    out = pd.DataFrame(index=tot.index)
    out["inscritos"] = tot
    r = summarize(RES[RES.uf.isin(ufs_t)], ["ano"]).set_index("ano")
    rd = pd.DataFrame(index=r.index)
    rd["compareceu_d1"] = r.pres_d1
    rd["comparecimento_d1"] = 1 - r.abst_d1
    rd["comparecimento_2dias"] = r.comp_2dias
    rd["presentes_2dias"] = r.pres_all
    for k in ("CN", "CH", "LC", "MT", "RED", "GERAL"):
        rd["nota_" + k] = r["m_" + k]
    for k in ("red_p700", "red_p900", "red_p1000", "red_pzero", "ger_p600", "ger_p700"):
        rd[k] = r[k]
    for c in range(1, 6):
        rd[f"comp{c}"] = r[f"m_c{c}"]
    out = out.join(rd)
    g = s.assign(w_net=s.n_insc * s.internet)
    out["treineiros"] = s[s.treineiro == 1].groupby("ano").n_insc.sum() / tot
    out["internet"] = g.groupby("ano").w_net.sum() / tot
    esc = s[s.escola_auto.isin(["Pública", "Privada", "Mista"])]
    out["pct_publica"] = esc[esc.escola_auto == "Pública"].groupby("ano").n_insc.sum() / esc.groupby("ano").n_insc.sum()
    out["pct_egressos"] = s[s.st == 1].groupby("ano").n_insc.sum() / tot
    rr = s[s.renda != "Sem info"]
    rt = rr.groupby("ano").n_insc.sum()
    out["renda_ate2sm"] = rr[rr.renda.isin(["Sem renda", "Até 1 SM", "1 a 2 SM"])].groupby("ano").n_insc.sum() / rt
    out["renda_acima3sm"] = rr[rr.renda.isin(["3 a 5 SM", "5 a 10 SM", "Mais de 10 SM"])].groupby("ano").n_insc.sum() / rt
    out["sam_internet_nt"] = s[(s.treineiro == 0) & (s.internet == 1)].groupby("ano").n_insc.sum()
    return out


@st.cache_data(show_spinner=False)
def year_table(ufs_t: tuple) -> pd.DataFrame:
    return _year_table(ufs_t)


@st.cache_data(show_spinner="Montando painel por UF…")
def uf_panel() -> pd.DataFrame:
    return pd.concat([_year_table((u,)).assign(uf=u).reset_index() for u in UFS], ignore_index=True)


YT = year_table(tuple(ufs))
R = YT.loc[ano_ref]
RR = YT.loc[ANO_RES]
COMP2 = YT.comparecimento_2dias.dropna().iloc[-1]  # último comparecimento observado (p/ anos sem resultados)
PANEL = uf_panel()

# ---------------------------------------------------------------- hero
sem_res = ano_ref not in ANOS_RES
st.markdown(f"""
<div class='hero'><div>
  <div class='hero-t'>Análise ENEM - {ANOS[0]} - {ANOS[-1]}</div>
  <div class='hero-s'>Mercado, desempenho e oportunidades · microdados INEP {ANOS[0]}–{ANOS[-1]}</div></div>
  <div><span class='tag'>Ano {ano_ref}</span><span class='tag'>{len(ufs)} UFs</span><span class='tag'>{short(R.inscritos)} inscritos</span>
  {"<span class='tag warn'>notas/presença: último ano publicado = " + str(ANO_RES) + "</span>" if sem_res else ""}</div></div>
""", unsafe_allow_html=True)


def kpi(col, label, fmt, results=False, pp=False, neutral=False, slot=None):
    y = ANO_RES if results else ano_ref
    slot = slot or st
    v = YT.loc[y, col] if y in YT.index else np.nan
    lab = label + (f" · {y}" if results and y != ano_ref else "")
    if pd.isna(v):
        slot.metric(lab, "n/d", border=True)
        return
    cy = ano_cmp if (not results or y == ano_ref) else y - 1
    c = YT.loc[cy, col] if cy in YT.index else np.nan
    d = None
    if pd.notna(c) and c:
        d = (f"{(v - c) * 100:+.1f}".replace(".", ",") + " p.p.") if pp else f"{(v - c) / abs(c) * 100:+.1f}%".replace(".", ",")
    s = YT[col].dropna().tolist()
    slot.metric(lab, fmt(v), d, delta_color="off" if neutral else "normal", border=True,
                chart_data=s if len(s) > 1 else None, chart_type="area" if len(s) > 1 else "line",
                help=f"Comparação com {cy}" if d else None)


def segmentos(s: pd.DataFrame) -> pd.DataFrame:
    """Segmentos mutuamente exclusivos (renda não informada é tratada como 'até 2 SM', de forma conservadora)."""
    s = s.assign(w_net=s.n_insc * s.internet, alta=~s.renda.isin(["Sem renda", "Até 1 SM", "1 a 2 SM", "Sem info"]))
    g = s.groupby(["st", "escola_auto", "treineiro", "alta"], as_index=False, observed=True)[["n_insc", "w_net"]].sum()

    def seg(r):
        if r.treineiro == 1 or r.st == 3:
            return "Treineiros e alunos de 1º/2º ano"
        if r.st == 1:
            return "Egressos · renda > 2 SM" if r.alta else "Egressos · renda ≤ 2 SM"
        if r.st == 2:
            if r.escola_auto == "Privada":
                return "Concluintes · escola privada"
            return "Concluintes pública · renda > 2 SM" if r.alta else "Concluintes pública · renda ≤ 2 SM"
        return "Sem EM / outros"

    g["segmento"] = g.apply(seg, axis=1)
    out = g.groupby("segmento", as_index=False)[["n_insc", "w_net"]].sum()
    out["internet"] = out.w_net / out.n_insc
    return out.sort_values("n_insc", ascending=False)


D = dict(sim_ticket=39.9, sim_meses=6, sim_conv=0.12, sim_g=0.0, sim_cac=45.0, sim_margem=0.7, sim_anos=1.3,
         p1=0.3, p2=0.8, p3=1.5, p4=2.5, p5=3.5)


def P(k):
    return st.session_state.get(k, D[k])


_names = ["Geral", "Mercado", "Possibilidades", "(removida)", "Consolidado", "Série Histórica", "Pública × Privada",
          "Municípios", "Aluno × Escola", "Mapa do Brasil"]
_display = [0, 9, 1, 2, 4, 5, 6, 7, 8]  # ordem de exibição; o índice lógico 3 (antiga aba de pitch) não é exibido
_t = st.tabs([_names[i] for i in _display], on_change="rerun")
tabs = [None] * len(_names)
for _pos, _i in enumerate(_display):
    tabs[_i] = _t[_pos]


def _active(n):
    """Só a aba aberta é renderizada (as demais não rodam: a página fica bem mais rápida)."""
    return tabs[n].open or st.session_state.get("_force_all", False)

# ============================================================== GERAL
with tabs[0]:
    if _active(0):
        c = st.columns(6)
        kpi("inscritos", "Inscritos", short, slot=c[0])
        kpi("sam_internet_nt", "SAM digital", short, slot=c[1])
        kpi("comparecimento_d1", "Comparecimento 1º dia", pct, results=True, pp=True, slot=c[2])
        kpi("nota_GERAL", "Nota média geral", lambda x: num_br(x, 0), results=True, slot=c[3])
        kpi("red_p700", "Redação ≥ 700", pct, results=True, pp=True, slot=c[4])
        kpi("internet", "Internet em casa", pct, pp=True, slot=c[5])
        c = st.columns(6)
        kpi("pct_egressos", "Egressos", pct, pp=True, neutral=True, slot=c[0])
        kpi("pct_publica", "Escola pública", pct, pp=True, neutral=True, slot=c[1])
        kpi("treineiros", "Treineiros", pct, pp=True, neutral=True, slot=c[2])
        kpi("renda_ate2sm", "Renda ≤ 2 SM", pct, pp=True, neutral=True, slot=c[3])
        kpi("renda_acima3sm", "Renda > 3 SM", pct, pp=True, neutral=True, slot=c[4])
        kpi("red_pzero", "Redação zerada", pct, results=True, pp=True, neutral=True, slot=c[5])

        section("Evolução e desempenho")
        a, b = st.columns([3, 2])
        with a:
            d = YT.reset_index()[["ano", "inscritos"]]
            d["yoy"] = d.inscritos.pct_change()
            lbl = [short(v) + (f"<br><span style='font-size:11px'>{y*100:+.0f}%</span>".replace(".", ",") if pd.notna(y) else "")
                   for v, y in zip(d.inscritos, d.yoy)]
            fig = go.Figure(go.Bar(x=d.ano.astype(str), y=d.inscritos, text=lbl, textposition="outside", cliponaxis=False,
                                   marker_color=[C2 if a_ == ano_ref else C1 for a_ in d.ano],
                                   hovertemplate="%{x}: %{y:,.0f} inscritos<extra></extra>"))
            fig.update_yaxes(visible=False, range=[0, d.inscritos.max() * 1.18])
            cg = cagr(YT.inscritos.loc[2021], YT.inscritos.iloc[-1], ANOS[-1] - 2021)
            chart(fig, "Inscritos por ano", "Barra laranja = ano de referência; % = variação sobre o ano anterior", h=340,
                  legend=False, insight=f"<b>CAGR 2021→{ANOS[-1]}: {pct(cg)} ao ano.</b> 2020 foi atípico (pandemia).")
        with b:
            z = YT[[f"nota_{k}" for k in ("CN", "CH", "LC", "MT", "RED", "GERAL")]].T.dropna(axis=1, how="all")
            z.index = [NOTA_NOME[i.split("_")[1]] for i in z.index]
            fig = px.imshow(z.round(0), text_auto=".0f", aspect="auto", color_continuous_scale=SEQ)
            fig.update_xaxes(type="category", side="top", title=None); fig.update_yaxes(title=None)
            fig.update_coloraxes(showscale=False)
            chart(fig, "Nota média por área", "Média dos presentes; 2025 sem resultados publicados" if sem_res else "Média dos presentes",
                  h=340, legend=False)

        a, b = st.columns(2)
        with a:
            d = YT.reset_index()[["ano", "comparecimento_d1", "comparecimento_2dias"]].dropna()
            fig = go.Figure()
            fig.add_scatter(x=d.ano, y=d.comparecimento_d1, name="1º dia", mode="lines+markers", line=dict(color=C1, width=3), marker_size=9)
            fig.add_scatter(x=d.ano, y=d.comparecimento_2dias, name="Nos 2 dias", mode="lines+markers", line=dict(color=C3, width=3), marker_size=9)
            fig.update_xaxes(dtick=1, title=None); fig.update_yaxes(tickformat=".0%", title=None, range=[0, 1])
            fig.update_layout(hovermode="x unified")
            chart(fig, "Comparecimento", "Quem se inscreveu e de fato fez a prova", h=320,
                  insight=f"Em {ANO_RES}, {pct(1 - RR.comparecimento_d1)} faltaram ao 1º dia.")
        with b:
            h = load("hist_notas")
            h = h[(h.ano == ANO_RES) & (h.nota == "GERAL") & h.rede.isin(["Pública", "Privada"])].copy()
            h["dens"] = h.n / h.groupby("rede").n.transform("sum")
            fig = px.area(h, x="bin", y="dens", color="rede", color_discrete_map={"Pública": C1, "Privada": C2})
            fig.update_traces(line_width=2, opacity=.55)
            fig.update_xaxes(title="Nota média geral"); fig.update_yaxes(title=None, tickformat=".0%")
            fig.update_layout(hovermode="x unified")
            pu, pr = summarize(res[(res.ano == ANO_RES) & res.dep.isin(["Federal", "Estadual", "Municipal"])], ["ano"]), \
                summarize(res[(res.ano == ANO_RES) & (res.dep == "Privada")], ["ano"])
            gap = (pr.m_GERAL.iloc[0] - pu.m_GERAL.iloc[0]) if len(pu) and len(pr) else np.nan
            chart(fig, f"Distribuição da nota por rede · {ANO_RES}", "Escola no Censo Escolar (pública × privada)", h=320,
                  insight=f"Gap médio privada − pública: <b>{gap:.0f} pontos</b>." if pd.notna(gap) else None)

        section("Onde estão os candidatos")
        a, b = st.columns([3, 2])
        with a:
            opts = {"Inscritos": ("inscritos", short), "SAM digital (internet, não-treineiros)": ("sam_internet_nt", short),
                    "% com internet": ("internet", pct), "% renda > 3 SM": ("renda_acima3sm", pct), "% egressos": ("pct_egressos", pct),
                    f"Nota média geral ({ANO_RES})": ("nota_GERAL", lambda x: f"{x:.0f}")}
            m_ = st.pills("Métrica do mapa", list(opts), default="Inscritos", key="map_geral", label_visibility="collapsed") or "Inscritos"
            col_, fmt_ = opts[m_]
            yy = ANO_RES if col_.startswith("nota") else ano_ref
            vals = PANEL[(PANEL.ano == yy) & PANEL.uf.isin(ufs)].set_index("uf")[col_]
            chart(tile_map(vals, fmt_), f"{m_} por UF", f"Cada bloco é uma UF (mapa estilizado) · ano {yy}", h=500, legend=False)
        with b:
            u = res[res.ano == ano_ref] if ano_ref in ANOS_RES else soc[soc.ano == ano_ref]
            u = u.groupby("uf", as_index=False).n_insc.sum().nlargest(10, "n_insc").sort_values("n_insc")
            fig = px.bar(u, x="n_insc", y="uf", orientation="h", color_discrete_sequence=[C1], text=u.n_insc.map(short))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False); fig.update_yaxes(title=None)
            tot_ = (soc[soc.ano == ano_ref].n_insc.sum())
            chart(fig, "Top 10 UFs", "Inscritos por UF de aplicação", h=500, legend=False,
                  insight=f"As 3 maiores UFs somam <b>{pct(u.n_insc.nlargest(3).sum() / tot_, 0)}</b> dos inscritos do recorte.")

        section("Perfil dos inscritos")
        dm = DEM[(DEM.ano == ano_ref) & DEM.uf.isin(ufs)].copy()
        a, b, c3 = st.columns(3)
        with a:
            dm["fx"] = dm.faixa.map(faixa_label)
            f = dm.groupby("fx", as_index=False).n_insc.sum()
            f["fx"] = pd.Categorical(f.fx, FAIXA_ORD, ordered=True)
            f = f.sort_values("fx")
            fig = px.bar(f, x="fx", y="n_insc", color_discrete_sequence=[C1], text=(f.n_insc / f.n_insc.sum()).map(lambda v: pct(v, 0)))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(title=None); fig.update_yaxes(visible=False, range=[0, f.n_insc.max() * 1.18])
            chart(fig, "Faixa etária", h=300, legend=False)
        with b:
            s = dm[dm.sexo.isin(["F", "M"])].groupby("sexo", as_index=False).n_insc.sum()
            s["sexo"] = s.sexo.map({"F": "Feminino", "M": "Masculino"})
            fig = px.pie(s, names="sexo", values="n_insc", hole=.6, color="sexo", color_discrete_map={"Feminino": C2, "Masculino": C1})
            fig.update_traces(textinfo="percent", textfont_size=14, hovertemplate="%{label}: %{value:,.0f}<extra></extra>")
            chart(fig, "Sexo", h=300)
        with c3:
            r_ = dm.groupby("cor", as_index=False).n_insc.sum()
            r_["cor"] = r_.cor.map(COR)
            r_ = r_[~r_.cor.isin(["Não declarado", "Sem info"])].sort_values("n_insc")
            fig = px.bar(r_, x="n_insc", y="cor", orientation="h", color_discrete_sequence=[C1], text=(r_.n_insc / r_.n_insc.sum()).map(lambda v: pct(v, 0)))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, r_.n_insc.max() * 1.2]); fig.update_yaxes(title=None)
            chart(fig, "Cor/raça declarada", h=300, legend=False)
        st.caption("Em 2024-2025 o INEP separa perfil (participantes) e notas (resultados), sem chave para cruzá-los: cruzamentos perfil × nota usam 2020-2023.")

# ============================================================ MERCADO
with tabs[1]:
    if _active(1):
        s_ref = soc[soc.ano == ano_ref]
        section(f"Dimensionamento · ENEM {ano_ref}")
        c1, c2, c3 = st.columns([2, 1, 1])
        renda_min = c1.select_slider("Renda familiar mínima do cliente B2C", RENDA_ORD[1:], value="2 a 3 SM", key="renda_min",
                                     help="SM = salários mínimos do ano da prova")
        so_internet = c2.toggle("Exigir internet em casa", True, key="so_net")
        excl_trein = c3.toggle("Excluir treineiros", True, key="excl_tr")
        ok_renda = RENDA_ORD[RENDA_ORD.index(renda_min):]
        tam = s_ref.n_insc.sum()
        s1 = s_ref[s_ref.treineiro == 0] if excl_trein else s_ref
        sam1 = s1[s1.internet == 1] if so_internet else s1
        sam2 = sam1[sam1.renda.isin(ok_renda)]
        sam_ef = sam2.n_insc.sum() * COMP2
        st.session_state.update(_tam=tam, _sam=sam2.n_insc.sum(), _sam_efetivo=sam_ef)
        funil = pd.DataFrame({"etapa": ["TAM · todos os inscritos", "Sem treineiros" if excl_trein else "Todos os perfis",
                                        "Com internet em casa" if so_internet else "Sem filtro de internet",
                                        f"SAM · renda ≥ {renda_min}", "SAM efetivo · comparecem aos 2 dias"],
                              "n": [tam, s1.n_insc.sum(), sam1.n_insc.sum(), sam2.n_insc.sum(), sam_ef]})
        a, b = st.columns([3, 2])
        with a:
            fig = go.Figure(go.Funnel(y=funil.etapa, x=funil.n, textinfo="value+percent initial",
                                      marker=dict(color=[C1, "#4f8fdc", "#7fabe6", C3, C4]),
                                      texttemplate="%{value:,.0f} · %{percentInitial:.0%}", connector=dict(line=dict(width=0))))
            chart(fig, "Funil de mercado", f"Comparecimento de {pct(COMP2)} (último ano com resultados) aplicado ao SAM", h=360, legend=False)
        with b:
            k = st.columns(2)
            k[0].metric("TAM", short(tam), border=True)
            k[1].metric("SAM", short(sam2.n_insc.sum()), border=True)
            k = st.columns(2)
            k[0].metric("SAM efetivo", short(sam_ef), border=True)
            k[1].metric("% do TAM", pct(sam_ef / tam), border=True)
            um_pct = sam_ef * 0.01 * P("sim_conv") * P("sim_ticket") * P("sim_meses")
            st.markdown(f"<div class='insight g'>Com ticket de R$ {P('sim_ticket'):.0f}/mês, {P('sim_meses')} meses e conversão de "
                        f"{pct(P('sim_conv'), 0)}, <b>cada 1% do SAM efetivo</b> cadastrado vale ≈ <b>{brl(um_pct)}</b>/ano "
                        "(premissas editáveis em Possibilidades).</div>", unsafe_allow_html=True)

        section("Segmentos")
        sg = segmentos(s_ref)
        sg["pct"] = sg.n_insc / sg.n_insc.sum()
        a, b = st.columns([3, 2])
        with a:
            fig = px.treemap(sg, path=["segmento"], values="n_insc", color="internet", color_continuous_scale=SEQ,
                             custom_data=["pct", "internet"])
            fig.update_traces(texttemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{customdata[0]:.0%}", textfont_size=14,
                              hovertemplate="%{label}<br>%{value:,.0f} inscritos<br>internet: %{customdata[1]:.0%}<extra></extra>",
                              marker_line_width=2)
            fig.update_coloraxes(showscale=False)
            chart(fig, "Mapa de segmentos", "Área = inscritos; cor = % com internet em casa", h=400, legend=False)
        with b:
            t = sg.assign(Segmento=sg.segmento, Inscritos=sg.n_insc, Share=sg.pct, Internet=sg.internet)[["Segmento", "Inscritos", "Share", "Internet"]]
            st.dataframe(t, hide_index=True, width="stretch", height=400, column_config={
                "Inscritos": st.column_config.NumberColumn(format="%d"),
                "Share": st.column_config.ProgressColumn("Share", min_value=0, max_value=float(t.Share.max()), format="percent"),
                "Internet": st.column_config.ProgressColumn("Internet", min_value=0, max_value=1, format="percent")})
            st.markdown("<div class='insight o'>Núcleo B2C: <b>egressos</b> e <b>concluintes de escola pública com renda &gt; 2 SM</b>. "
                        "Renda ≤ 2 SM → freemium, escolas e governo (B2B/B2G).</div>", unsafe_allow_html=True)

        section("Geografia da oportunidade")
        uu = s_ref.assign(w=s_ref.n_insc * s_ref.internet)
        g1 = uu.groupby("uf").apply(lambda d: pd.Series({
            "inscritos": d.n_insc.sum(), "internet": d.w.sum() / d.n_insc.sum(),
            "renda3": d[d.renda.isin(["3 a 5 SM", "5 a 10 SM", "Mais de 10 SM"])].n_insc.sum() / max(d[d.renda != "Sem info"].n_insc.sum(), 1),
            "sam": d[(d.treineiro == 0) & (d.internet == 1) & d.renda.isin(ok_renda)].n_insc.sum()}), include_groups=False).reset_index()
        nota_uf = summarize(res[res.ano == ANO_RES], ["uf"])[["uf", "m_GERAL", "red_p700"]]
        ufm = g1.merge(nota_uf, on="uf", how="left")
        ufm["regiao"] = ufm.uf.map(REGIAO)
        mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
        ufm["indice"] = 100 * (0.5 * mm(ufm.sam) + 0.3 * (1 - mm(ufm.m_GERAL.fillna(ufm.m_GERAL.mean()))) + 0.2 * mm(ufm.internet))
        a, b = st.columns([3, 2])
        with a:
            opt = st.pills("Mapa", ["Índice de oportunidade", "SAM", "Nota média", "Internet"], default="Índice de oportunidade", key="map_merc",
                           label_visibility="collapsed") or "Índice de oportunidade"
            mp = {"Índice de oportunidade": ("indice", lambda x: f"{x:.0f}"), "SAM": ("sam", short),
                  "Nota média": ("m_GERAL", lambda x: f"{x:.0f}"), "Internet": ("internet", pct)}[opt]
            chart(tile_map(ufm.set_index("uf")[mp[0]], mp[1]), opt, f"{ano_ref} · SAM com renda ≥ {renda_min}", h=500, legend=False)
        with b:
            top = ufm.nlargest(10, "indice").sort_values("indice")
            fig = px.bar(top, x="indice", y="uf", orientation="h", color_discrete_sequence=[C3], text=top.indice.map(lambda v: f"{v:.0f}"))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, 112]); fig.update_yaxes(title=None)
            chart(fig, "Top 10 · índice de oportunidade", "50% volume do SAM + 30% espaço de melhoria (nota baixa) + 20% conectividade",
                  h=500, legend=False)
        tb = ufm.sort_values("indice", ascending=False).rename(columns={
            "uf": "UF", "regiao": "Região", "inscritos": "Inscritos", "sam": "SAM", "internet": "Internet", "renda3": "Renda > 3 SM",
            "m_GERAL": "Nota", "red_p700": "Redação ≥ 700", "indice": "Índice"})[["UF", "Região", "Inscritos", "SAM", "Internet", "Renda > 3 SM", "Nota", "Redação ≥ 700", "Índice"]]
        with st.expander("Tabela completa por UF", expanded=False):
            st.dataframe(tb, hide_index=True, width="stretch", height=420, column_config={
                "Inscritos": st.column_config.NumberColumn(format="%d"), "SAM": st.column_config.NumberColumn(format="%d"),
                "Internet": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "Renda > 3 SM": st.column_config.ProgressColumn(min_value=0, max_value=float(tb["Renda > 3 SM"].max()), format="percent"),
                "Nota": st.column_config.NumberColumn(format="%.0f"),
                "Redação ≥ 700": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "Índice": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})

        a, b = st.columns(2)
        with a:
            m = MUN[(MUN.ano == ano_ref) & MUN.uf.isin(ufs)].nlargest(15, "n_insc").sort_values("n_insc")
            m["lbl"] = m.mun + " · " + m.uf
            fig = px.bar(m, x="n_insc", y="lbl", orientation="h", color_discrete_sequence=[C1], text=m.n_insc.map(short))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, m.n_insc.max() * 1.18]); fig.update_yaxes(title=None)
            top20 = MUN[(MUN.ano == ano_ref) & MUN.uf.isin(ufs)].nlargest(20, "n_insc").n_insc.sum() / MUN[(MUN.ano == ano_ref) & MUN.uf.isin(ufs)].n_insc.sum()
            chart(fig, "Top 15 municípios", "Local de aplicação da prova", h=470, legend=False,
                  insight=f"Os 20 maiores municípios concentram <b>{pct(top20, 0)}</b> dos inscritos — dá para entrar por poucas praças.")
        with b:
            eg = load("cubo_egressos")
            eg = eg[eg.ano == ano_ref].groupby("anos_bin", as_index=False).n_insc.sum()
            ordem = ["n/i", "1 ano", "2 anos", "3 anos", "4-5 anos", "6-10 anos", "11+ anos"]
            eg["anos_bin"] = pd.Categorical(eg.anos_bin, ordem, ordered=True)
            eg = eg.sort_values("anos_bin")
            fig = px.bar(eg, x="anos_bin", y="n_insc", color_discrete_sequence=[C2], text=eg.n_insc.map(short))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(title=None); fig.update_yaxes(visible=False, range=[0, eg.n_insc.max() * 1.2])
            chart(fig, "Egressos: há quanto tempo concluíram o ensino médio", "Mercado de 'repetentes' (estudam por conta própria)", h=225, legend=False)
            rr = s_ref[s_ref.renda != "Sem info"].groupby("renda", as_index=False).n_insc.sum()
            rr["renda"] = pd.Categorical(rr.renda, RENDA_ORD, ordered=True)
            rr = rr.sort_values("renda")
            fig = px.bar(rr, x="renda", y="n_insc", color_discrete_sequence=[C1], text=(rr.n_insc / rr.n_insc.sum()).map(lambda v: pct(v, 0)))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(title=None); fig.update_yaxes(visible=False, range=[0, rr.n_insc.max() * 1.2])
            chart(fig, "Renda familiar", "Faixas em salários mínimos", h=225, legend=False)

# ======================================================== POSSIBILIDADES
def simulate(sam0, n=5, mult=1.0, conv=None, ticket=None, g=None):
    conv = P("sim_conv") if conv is None else conv
    ticket = P("sim_ticket") if ticket is None else ticket
    g = P("sim_g") if g is None else g
    rows = []
    for i in range(1, n + 1):
        sam = sam0 * (1 + g) ** i
        cad = sam * P(f"p{i}") / 100 * mult
        pag = cad * conv
        rows.append(dict(ano=i, SAM=sam, cadastrados=cad, pagantes=pag, receita=pag * ticket * P("sim_meses")))
    return pd.DataFrame(rows)


with tabs[2]:
    if _active(2):
        sam_base = st.session_state.get("_sam_efetivo", R.sam_internet_nt * COMP2)
        section("Simulador de receita · hipóteses editáveis")
        st.caption(f"SAM efetivo de partida (aba Mercado): **{num_br(sam_base)}** candidatos — renda mínima, internet, treineiros e comparecimento definidos lá.")
        with st.container(border=True):
            c = st.columns(5)
            c[0].number_input("Ticket mensal (R$)", 5.0, 500.0, D["sim_ticket"], 1.0, key="sim_ticket")
            c[1].slider("Meses pagos por ano", 1, 12, D["sim_meses"], key="sim_meses")
            c[2].slider("Conversão em pagante", 0.01, 0.6, D["sim_conv"], 0.01, key="sim_conv", format="%.2f")
            c[3].slider("Crescimento anual do SAM", -0.10, 0.15, D["sim_g"], 0.01, key="sim_g", format="%.2f")
            c[4].slider("Margem de contribuição", 0.2, 0.95, D["sim_margem"], 0.05, key="sim_margem", format="%.2f")
            c = st.columns(5)
            for i in range(1, 6):
                c[i - 1].number_input(f"Penetração ano {i} (%)", 0.0, 30.0, D[f"p{i}"], 0.1, key=f"p{i}", help="% do SAM que se cadastra")
            c = st.columns(3)
            c[0].number_input("CAC por pagante (R$)", 1.0, 1000.0, D["sim_cac"], 5.0, key="sim_cac")
            c[1].slider("Anos médios de permanência", 1.0, 3.0, D["sim_anos"], 0.1, key="sim_anos")
            y21 = list(YT.index).index(2021)
            hist_cagr = cagr(YT.sam_internet_nt.iloc[y21], YT.sam_internet_nt.iloc[-1], len(YT) - 1 - y21)
            c[2].metric("CAGR histórico do SAM digital", pct(hist_cagr), help=f"2021→{ANOS[-1]} (2020 excluído: pandemia). Referência para o slider de crescimento.")

        cen = {"Conservador": simulate(sam_base, mult=0.5, conv=P("sim_conv") * 0.75), "Base": simulate(sam_base),
               "Otimista": simulate(sam_base, mult=1.5, conv=P("sim_conv") * 1.25)}
        long = pd.concat([d.assign(cenario=k) for k, d in cen.items()])
        ltv = P("sim_ticket") * P("sim_meses") * P("sim_anos") * P("sim_margem")
        pay = P("sim_cac") / (P("sim_ticket") * P("sim_margem"))
        k = st.columns(5)
        k[0].metric("Receita ano 3 (base)", brl(cen["Base"].receita.iloc[2]), border=True)
        k[1].metric("Receita ano 5 (base)", brl(cen["Base"].receita.iloc[4]), border=True)
        k[2].metric("LTV", f"R$ {ltv:,.0f}".replace(",", "."), border=True, help="ticket × meses × anos × margem")
        k[3].metric("LTV / CAC", dec(ltv / P("sim_cac"), 1) + "x", border=True, help="Referência de mercado: ≥ 3x")
        k[4].metric("Payback", dec(pay, 1) + " meses", border=True, help="CAC ÷ margem mensal")

        a, b = st.columns(2)
        with a:
            fig = px.bar(long, x="ano", y="receita", color="cenario", barmode="group",
                         color_discrete_map={"Conservador": C2, "Base": C1, "Otimista": C3}, text=long.receita.map(short))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(title="Ano de operação", dtick=1); fig.update_yaxes(visible=False)
            chart(fig, "Receita anual por cenário", "Conservador = ½ penetração e conversão −25% · Otimista = 1,5× e +25%", h=360)
        with b:
            rng = np.random.default_rng(42)
            N = 20000
            pen3 = P("p3") / 100 * rng.lognormal(0, 0.45, N)
            conv = np.clip(rng.beta(P("sim_conv") * 40, (1 - P("sim_conv")) * 40, N), 0, 1)
            tick = rng.triangular(P("sim_ticket") * 0.7, P("sim_ticket"), P("sim_ticket") * 1.3, N)
            rec = sam_base * (1 + P("sim_g")) ** 3 * pen3 * conv * tick * P("sim_meses")
            p10, p50, p90 = np.percentile(rec, [10, 50, 90])
            st.session_state["_mc"] = (p10, p50, p90)
            fig = px.histogram(x=rec, nbins=60, color_discrete_sequence=[C1])
            for v, n_ in ((p10, "P10"), (p50, "P50"), (p90, "P90")):
                fig.add_vline(x=v, line_dash="dot", annotation_text=f"{n_} {short(v)}", annotation_position="top")
            fig.update_xaxes(title="Receita do ano 3 (R$)"); fig.update_yaxes(visible=False)
            chart(fig, "Monte Carlo · receita do ano 3", "20 mil sorteios: penetração (lognormal), conversão (beta) e ticket (triangular)", h=360, legend=False)

        ts = np.linspace(P("sim_ticket") * 0.6, P("sim_ticket") * 1.6, 6)
        cs = np.linspace(max(P("sim_conv") * 0.4, 0.02), min(P("sim_conv") * 2.0, 0.6), 6)
        z = pd.DataFrame([[sam_base * (1 + P("sim_g")) ** 3 * P("p3") / 100 * c_ * t_ * P("sim_meses") for t_ in ts] for c_ in cs],
                         index=[pct(c_, 0) for c_ in cs], columns=[f"R$ {t_:.0f}" for t_ in ts])
        fig = px.imshow(z / 1e6, text_auto=".2f", color_continuous_scale=SEQ, aspect="auto")
        fig.update_coloraxes(showscale=False); fig.update_xaxes(title="Ticket mensal", side="top"); fig.update_yaxes(title="Conversão")
        chart(fig, "Sensibilidade · receita do ano 3 (R$ milhões)", "Ticket × conversão, mantendo as demais hipóteses", h=320, legend=False)

        section("Onde o aluno tem dor mensurável")
        anos_sel = st.pills("Anos considerados", ANOS_RES, selection_mode="multi", default=ANOS_RES[-3:], key="anos_dor") or ANOS_RES[-3:]
        a, b = st.columns(2)
        with a:
            cd = load("red_dist_comp")
            cd = cd[cd.ano.isin(anos_sel)]
            cm = cd.assign(w=cd.nota * cd.n).groupby("comp").apply(lambda d: d.w.sum() / d.n.sum(), include_groups=False).reset_index(name="media")
            cm["nome"] = cm.comp.map(COMP_NOME)
            cm = cm.sort_values("media")
            fig = px.bar(cm, x="media", y="nome", orientation="h", color_discrete_sequence=[C2], text=cm.media.round(0))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, 215]); fig.update_yaxes(title=None)
            z_ = cd.groupby(["comp", "nota"]).n.sum().reset_index()
            z_["share"] = z_.n / z_.groupby("comp").n.transform("sum")
            top5 = z_[(z_.comp == 5) & (z_.nota == 200)].share.sum()
            zero5 = z_[(z_.comp == 5) & (z_.nota == 0)].share.sum()
            chart(fig, "Redação · média por competência", "Escala 0-200, redações válidas", h=330, legend=False,
                  insight=f"Menor média: <b>{cm.nome.iloc[0]}</b>. Em C5, só {pct(top5)} tiram 200 e {pct(zero5)} zeram.")
        with b:
            dc = load("red_discordancia")
            dc = dc.assign(p3=dc.n_av3 / dc.n_pares, p200=dc.diff_gt200 / dc.n_pares)
            fig = go.Figure()
            fig.add_bar(x=dc.ano.astype(str), y=dc.p3, name="Foi ao 3º corretor", marker_color=C1)
            fig.add_bar(x=dc.ano.astype(str), y=dc.p200, name="Discordância > 200 pts", marker_color=C2)
            fig.update_layout(barmode="group"); fig.update_yaxes(tickformat=".0%", title=None)
            chart(fig, "Imprevisibilidade da correção", "Redações em que os 2 primeiros corretores divergiram", h=330,
                  insight="Mais de 1 em cada 5 redações exige 3º corretor: correção explicável por competência é diferencial.")

        it = load("itens_stats")
        it = it[it.ano.isin(anos_sel)]
        h = it.groupby(["area", "CO_HABILIDADE", "rede"]).agg(hits=("hits", "sum"), valid=("valid", "sum"), itens=("CO_ITEM", "nunique")).reset_index()
        h["taxa"] = h.hits / h.valid
        ht = h[(h.rede == "Todos") & (h["itens"] >= 4)].assign(hab=lambda d: d.area + "·H" + d.CO_HABILIDADE.astype(int).astype(str))
        a, b = st.columns(2)
        with a:
            w = ht.nsmallest(15, "taxa").sort_values("taxa", ascending=False)
            fig = px.bar(w, x="taxa", y="hab", orientation="h", color="area", text=w.taxa.map(lambda v: pct(v, 0)),
                         color_discrete_map={"CN": C1, "CH": C2, "LC": C3, "MT": C4})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, w.taxa.max() * 1.2]); fig.update_yaxes(title=None)
            chart(fig, "15 habilidades com menor acerto", "Taxa de acerto nacional por habilidade da matriz (acaso ≈ 20%)", h=470)
        with b:
            pv = h[h.rede.isin(["Pública", "Privada"])].pivot_table(index=["area", "CO_HABILIDADE"], columns="rede", values="taxa").dropna().reset_index()
            pv["gap"] = pv["Privada"] - pv["Pública"]
            ga = pv.groupby("area").gap.mean().reindex(["CN", "CH", "LC", "MT"]).reset_index()
            ga["nome"] = ga.area.map(NOTA_NOME)
            fig = px.bar(ga, x="nome", y="gap", color_discrete_sequence=[C2], text=ga.gap.map(lambda v: f"{v*100:.1f} p.p.".replace(".", ",")))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(title=None); fig.update_yaxes(visible=False, range=[0, ga.gap.max() * 1.25])
            chart(fig, "Gap de acerto: privada − pública", "Média por habilidade, em pontos percentuais de acerto", h=470, legend=False)

        section("O que explica a nota? · statsmodels (OLS e Logit, 2020-2023)")
        M = lib.fit_ols_nota()
        ols, lg = M["ols"], M["logit"]
        a, b = st.columns(2)
        with a:
            keep = {"log_renda": "ln(1 + renda em SM)", "esc_mae_ord": "Escolaridade da mãe (+1 nível)", "esc_pai_ord": "Escolaridade do pai (+1 nível)",
                    "internet": "Internet em casa", "computador": "Computador em casa", "privada": "Escola privada", "mulher": "Mulher"}
            ci = ols.conf_int()
            co = pd.DataFrame({"var": list(keep.values()), "coef": [ols.params[k_] for k_ in keep], "lo": [ci.loc[k_, 0] for k_ in keep],
                               "hi": [ci.loc[k_, 1] for k_ in keep]}).sort_values("coef")
            fig = go.Figure(go.Scatter(x=co.coef, y=co["var"], mode="markers", marker=dict(size=11, color=C1),
                                       error_x=dict(type="data", symmetric=False, array=co.hi - co.coef, arrayminus=co.coef - co.lo, color=C1, thickness=2),
                                       hovertemplate="%{y}: %{x:.1f} pts<extra></extra>"))
            fig.add_vline(x=0, line_dash="dot")
            fig.update_xaxes(title="Pontos na média geral (IC 95%, erro robusto HC1)")
            chart(fig, "Efeito sobre a nota média geral", f"n = {num_br(M['n'])} (concluintes com tipo de escola informado) · R² = {dec(M['r2'])}",
                  h=380, legend=False, insight="Associações controladas, não causalidade: renda, escola e conectividade são correlacionadas.")
        with b:
            rb = pd.Series(M["r2_blocks"]).sort_values().rename("dR2").reset_index()
            fig = px.bar(rb, x="dR2", y="index", orientation="h", color_discrete_sequence=[C3], text=rb.dR2.map(lambda v: pct(v, 1)))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, rb.dR2.max() * 1.25]); fig.update_yaxes(title=None)
            orr, orp, ori = (np.exp(lg.params[k_]) for k_ in ("log_renda", "privada", "internet"))
            chart(fig, "Poder explicativo único de cada bloco", "Queda do R² ao retirar o bloco do modelo", h=380, legend=False,
                  insight=f"Logit (redação ≥ 700): razão de chances {dec(orp)}x para escola privada e {dec(ori)}x para internet em casa.")

# ============================================================ CONSOLIDADO
with tabs[4]:
    if _active(4):
        section("Painel anual")
        tab = YT[["inscritos", "comparecimento_d1", "comparecimento_2dias", "nota_GERAL", "nota_RED", "red_p700", "red_pzero", "internet",
                  "pct_publica", "pct_egressos", "treineiros", "renda_ate2sm", "renda_acima3sm", "sam_internet_nt"]].copy()
        fmt = tab.copy()
        for c_ in fmt.columns:
            if c_ in ("inscritos", "sam_internet_nt"):
                fmt[c_] = fmt[c_].map(lambda x: num_br(x) if pd.notna(x) else "n/d")
            elif c_ in ("nota_GERAL", "nota_RED"):
                fmt[c_] = fmt[c_].map(lambda x: num_br(x, 0) if pd.notna(x) else "n/d")
            else:
                fmt[c_] = fmt[c_].map(lambda x: pct(x) if pd.notna(x) else "n/d")
        fmt.columns = ["Inscritos", "Comparec. 1º dia", "Comparec. 2 dias", "Nota geral", "Nota redação", "Redação ≥700", "Redação zero",
                       "Internet", "% escola pública", "% egressos", "% treineiros", "Renda ≤2 SM", "Renda >3 SM", "SAM digital (não-trein.)"]
        fmt.index = fmt.index.astype(str)
        st.dataframe(fmt.T, width="stretch", height=530)

        section("Achados principais")
        a0, a1 = YT.index[0], YT.index[-1]
        ols_ = lib.fit_ols_nota()
        pub = summarize(res[(res.ano == ANO_RES) & res.dep.isin(["Federal", "Estadual", "Municipal"])], ["ano"])
        prv = summarize(res[(res.ano == ANO_RES) & (res.dep == "Privada")], ["ano"])
        d_, t_, cd_ = lib.welch_d(prv.m_GERAL.iloc[0], prv.sd_GERAL.iloc[0], prv.cnt_GERAL.iloc[0], pub.m_GERAL.iloc[0], pub.sd_GERAL.iloc[0], pub.cnt_GERAL.iloc[0])
        ins = [
            ("", f"<b>Escala:</b> {short(R.inscritos)} inscritos em {ano_ref}; CAGR 2021→{a1}: {pct(cagr(YT.inscritos.loc[2021], YT.inscritos.iloc[-1], a1 - 2021))} (2020 excluído por ser atípico)."),
            ("o", f"<b>Abstenção:</b> {pct(1 - RR.comparecimento_d1)} faltaram ao 1º dia em {ANO_RES} (pico em 2020: {pct(1 - YT.loc[2020, 'comparecimento_d1']) if 2020 in YT.index else '–'})."),
            ("g", f"<b>Desigualdade:</b> escola privada vs pública = {d_:.0f} pontos na média geral ({ANO_RES}); Cohen d = {dec(cd_)}, Welch t = {t_:.0f}."),
            ("", f"<b>Redação:</b> média {RR.nota_RED:.0f}; {pct(RR.red_pzero)} zeradas; só {pct(RR.red_p900)} ≥ 900 ({ANO_RES})."),
            ("o", f"<b>Conectividade:</b> {pct(R.internet)} com internet em casa em {ano_ref} (era {pct(YT.loc[a0, 'internet'])} em {a0}; 2024+ pergunta sobre wi-fi)."),
            ("g", f"<b>Renda:</b> {pct(R.renda_ate2sm)} das famílias ≤ 2 SM e {pct(R.renda_acima3sm)} > 3 SM → freemium + B2G/escolas, e B2C para o núcleo > 2 SM."),
            ("", f"<b>Egressos:</b> {pct(R.pct_egressos)} dos inscritos já concluíram o EM — público que estuda por conta própria."),
            ("o", f"<b>Modelo (OLS):</b> R² = {dec(ols_['r2'])}; controlando o resto, escola privada ≈ +{ols_['ols'].params['privada']:.0f} pts, "
                  f"computador ≈ +{ols_['ols'].params['computador']:.0f}, internet ≈ +{ols_['ols'].params['internet']:.0f} (associações)."),
        ]
        cc = st.columns(2)
        for j, (cls, txt) in enumerate(ins):
            cc[j % 2].markdown(f"<div class='insight {cls}'>{txt}</div>", unsafe_allow_html=True)

        with st.expander("Limitações e cuidados", expanded=False):
            st.markdown("\n".join([
                '- **2020** foi atípico (pandemia; abstenção ~52%). Tendências são mais confiáveis de 2021 em diante.',
                '- **2024-2025**: perfil socioeconômico e resultados vêm de bases separadas **sem chave** de ligação; cruzamentos perfil × nota usam 2020-2023. O pacote de **2025** traz só o perfil (sem notas, presença ou competências da redação): métricas de desempenho usam o último ano publicado.',
                '- A rede da escola (pública/privada) vem do Censo Escolar e existe só para parte dos inscritos (concluintes); egressos ficam "sem info".',
                '- Renda é declarada em faixas de salário mínimo do ano; foram harmonizadas entre anos.',
                '- Internet/computador "não respondido" conta como *não* (leve subestimação). Em 2024-2025 a pergunta é sobre **wi-fi** (Q020).',
                '- Valores em R$, penetração e conversão são **hipóteses**, não observações.',
            ]))
        buf = io.StringIO(); tab.to_csv(buf)
        st.download_button("Baixar tabela anual (CSV)", buf.getvalue(), "enem_consolidado.csv", "text/csv")

# ========================================================== SÉRIE HISTÓRICA
METR = {
    "Mercado": {"Inscritos": ("inscritos", "n"), "SAM digital (internet, não-treineiros)": ("sam_internet_nt", "n"),
                "Compareceram nos 2 dias": ("presentes_2dias", "n"), "Comparecimento 1º dia": ("comparecimento_d1", "p")},
    "Desempenho": {"Nota média geral": ("nota_GERAL", "x"), "Nota Ciências da Natureza": ("nota_CN", "x"), "Nota Ciências Humanas": ("nota_CH", "x"),
                   "Nota Linguagens": ("nota_LC", "x"), "Nota Matemática": ("nota_MT", "x"), "Nota Redação": ("nota_RED", "x"),
                   "% Redação ≥ 700": ("red_p700", "p"), "% Redação ≥ 900": ("red_p900", "p"), "% Redação zerada": ("red_pzero", "p")},
    "Perfil": {"% com internet": ("internet", "p"), "% escola pública": ("pct_publica", "p"), "% egressos": ("pct_egressos", "p"),
               "% treineiros": ("treineiros", "p"), "% renda ≤ 2 SM": ("renda_ate2sm", "p"), "% renda > 3 SM": ("renda_acima3sm", "p")},
}
FLAT = {k: v for g in METR.values() for k, v in g.items()}
with tabs[5]:
    if _active(5):
        c = st.columns([1.2, 2, 1.6])
        grupo = c[0].segmented_control("Tema", list(METR), default="Mercado", key="serie_grupo", required=True)
        met = c[1].selectbox("Métrica", list(METR[grupo]), key=f"serie_met_{grupo}")
        comp_modo = c[2].segmented_control("Comparar por", ["Recorte", "Região", "UFs"], default="Recorte", key="serie_modo", required=True)
        col, kind = METR[grupo][met]
        if comp_modo == "UFs":
            uf_cmp = st.multiselect("UFs (até 4)", UFS, default=[u for u in ["SP", "CE", "BA"] if u in UFS], max_selections=4, key="serie_ufs")
        if comp_modo == "Recorte":
            series = {"Recorte": YT[col]}
        elif comp_modo == "Região":
            series = {r_: year_table(tuple(u for u in UFS if REGIAO[u] == r_))[col] for r_ in REGIOES}
        else:
            series = {u: year_table((u,))[col] for u in (uf_cmp or UFS[:3])}
        long = pd.concat([v.rename("v").reset_index().assign(serie=k_) for k_, v in series.items()]).dropna()
        fmt_y = ".0%" if kind == "p" else ",.0f"
        paleta = [C1, C2, C3, C4, "#e87ba4"]
        fig = px.line(long, x="ano", y="v", color="serie", markers=True, color_discrete_sequence=paleta)
        fig.update_traces(line_width=3, marker_size=9)
        fig.update_xaxes(dtick=1, title=None); fig.update_yaxes(tickformat=fmt_y, title=None)
        fig.add_vrect(x0=2019.6, x1=2020.4, fillcolor="rgba(128,128,128,.15)", line_width=0, annotation_text="pandemia", annotation_position="top left")
        fig.update_layout(hovermode="x unified")
        ymin = YT[col].dropna()
        chart(fig, met, f"{len(ufs)} UFs no recorte · faixa cinza = 2020 (pandemia)", h=400,
              insight=f"De {ymin.index[0]} a {ymin.index[-1]}: <b>{short(ymin.iloc[0]) if kind == 'n' else (pct(ymin.iloc[0]) if kind == 'p' else f'{ymin.iloc[0]:.0f}')}</b> → "
                      f"<b>{short(ymin.iloc[-1]) if kind == 'n' else (pct(ymin.iloc[-1]) if kind == 'p' else f'{ymin.iloc[-1]:.0f}')}</b>.")

        a, b = st.columns(2)
        with a:
            base_ = long.pivot(index="ano", columns="serie", values="v")
            b0 = 2021 if 2021 in base_.index else base_.index[0]
            idx = (base_ / base_.loc[b0] * 100).reset_index().melt("ano", var_name="serie", value_name="i").dropna()
            fig = px.line(idx, x="ano", y="i", color="serie", markers=True, color_discrete_sequence=paleta)
            fig.update_traces(line_width=3, marker_size=8)
            fig.update_xaxes(dtick=1, title=None); fig.update_yaxes(title=None)
            fig.add_hline(y=100, line_dash="dot"); fig.update_layout(hovermode="x unified")
            chart(fig, f"Índice ({b0} = 100)", "Compara trajetórias em base comum", h=330)
        with b:
            yoy = (base_.pct_change() * 100 if kind != "p" else base_.diff() * 100).dropna(how="all").reset_index().melt("ano", var_name="serie", value_name="v").dropna()
            fig = px.bar(yoy, x=yoy.ano.astype(str), y="v", color="serie", barmode="group", color_discrete_sequence=paleta)
            fig.update_xaxes(title=None); fig.update_yaxes(title=None, ticksuffix=" p.p." if kind == "p" else "%")
            chart(fig, "Variação anual", "p.p. para percentuais; % para volumes e notas", h=330)

        section("Tendência linear e cenário (statsmodels OLS)")
        alvo = YT[col].dropna()
        excl20 = st.toggle("Excluir 2020 (pandemia) do ajuste", True, key="ex20")
        xs, ys = alvo.index.values, alvo.values
        if excl20:
            keep_ = xs != 2020
            xs, ys = xs[keep_], ys[keep_]
        last = int(alvo.index.max())
        r_, pr = trend_fit(xs, ys, [last + 1, last + 2])
        slope = r_.params[1]
        fut = pd.DataFrame({"ano": [last + 1, last + 2], "central": pr["mean"].values, "lo": pr["obs_ci_lower"].values, "hi": pr["obs_ci_upper"].values})
        f2 = go.Figure()
        f2.add_scatter(x=alvo.index, y=alvo.values, mode="lines+markers", name="Observado", line=dict(color=C1, width=3), marker_size=9)
        f2.add_scatter(x=[last] + list(fut.ano), y=[alvo.iloc[-1]] + list(fut.central), mode="lines+markers", name="Tendência",
                       line=dict(color=C2, dash="dash", width=3), marker_size=9)
        f2.add_scatter(x=list(fut.ano) + list(fut.ano[::-1]), y=list(fut.hi) + list(fut.lo[::-1]), fill="toself", fillcolor="rgba(235,104,52,.18)",
                       line=dict(width=0), name="Intervalo 80%", hoverinfo="skip")
        f2.update_xaxes(dtick=1, title=None); f2.update_yaxes(tickformat=fmt_y, title=None)
        a, b = st.columns([3, 1.2])
        with a:
            chart(f2, "Observado × tendência", f"{len(xs)} pontos: é um cenário, não uma previsão — o intervalo é largo de propósito", h=360)
        with b:
            st.metric("Inclinação por ano", (f"{slope*100:+.2f}".replace(".", ",") + " p.p.") if kind == "p" else num_br(slope, 0), border=True)
            st.metric("R² do ajuste", dec(r_.rsquared), border=True)
            st.metric("p-valor da inclinação", dec(r_.pvalues[1], 3), border=True)

        section("Comparativo entre UFs")
        pp = PANEL.copy()
        a, b = st.columns([3, 2])
        with a:
            hm = pp.pivot(index="uf", columns="ano", values=col).dropna(axis=1, how="all")
            hm = hm.loc[hm.iloc[:, -1].sort_values(ascending=False).index]
            fig = px.imshow(hm, aspect="auto", color_continuous_scale=SEQ, text_auto=".0%" if kind == "p" else ",.0f")
            fig.update_xaxes(type="category", side="top", title=None); fig.update_yaxes(title=None, dtick=1)
            fig.update_coloraxes(showscale=False)
            chart(fig, f"{met} por UF e ano", "Ordenado pelo último ano disponível", h=780, legend=False)
        with b:
            a0_ = alvo.index[1] if (2020 in alvo.index and len(alvo) > 2) else alvo.index[0]
            ch = pp.pivot(index="uf", columns="ano", values=col)
            if a0_ in ch.columns and last in ch.columns:
                dlt = ((ch[last] - ch[a0_]) * 100 if kind == "p" else (ch[last] / ch[a0_] - 1) * 100).dropna().sort_values()
                sel = pd.concat([dlt.head(5), dlt.tail(5)]).reset_index()
                sel.columns = ["uf", "d"]
                fig = px.bar(sel, x="d", y="uf", orientation="h", text=sel.d.map(lambda v: f"{v:+.1f}".replace(".", ",")),
                             color=sel.d > 0, color_discrete_map={True: C3, False: C2})
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(visible=False); fig.update_yaxes(title=None)
                chart(fig, f"Maiores variações {a0_}→{last}", "p.p. (percentuais) ou % (demais) · 5 maiores quedas e 5 maiores altas", h=360, legend=False)
            cols = {"inscritos": "Inscritos", "nota_GERAL": "Nota geral", "internet": "Internet", "renda_acima3sm": "Renda >3 SM",
                    "pct_publica": "% pública", "red_p700": "Red ≥700", "comparecimento_d1": "Comparec."}
            cor = pp[pp.ano.between(2021, 2024)][list(cols)].rename(columns=cols).corr()
            fig = px.imshow(cor.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale=[C2, "#f4f4f4", C1])
            fig.update_coloraxes(showscale=False); fig.update_xaxes(side="top", title=None); fig.update_yaxes(title=None)
            chart(fig, "Correlação entre UFs", "Painel UF × ano (2021-2024): associação, não causalidade", h=400, legend=False)
        with st.expander("Tabela da série selecionada"):
            st.dataframe(pd.DataFrame(series).rename_axis("Ano"), width="stretch")


# ================================================== PÚBLICA × PRIVADA
with tabs[6]:
    if _active(6):
        section("Notas: escola pública × privada")
        st.caption("Rede da escola segundo o Censo Escolar (concluintes do ensino médio). Egressos e candidatos sem escola vinculada ficam de fora. "
                   "Disponível para os anos com resultados publicados.")
        c = st.columns([1.3, 1.3, 2])
        yy = c[0].segmented_control("Ano", ANOS_RES, default=ANOS_RES[-1], key="pp_ano", required=True)
        detalhe = c[1].toggle("Detalhar federal/estadual/municipal", False, key="pp_det")
        rr_ = res[res.dep != "Sem info"].copy()
        rr_["grp"] = rr_.dep if detalhe else rr_.dep.map(lambda d: "Privada" if d == "Privada" else "Pública")
        sg_ = summarize(rr_.drop(columns="dep").rename(columns={"grp": "dep"}), ["ano", "dep"])
        cor_g = {"Pública": C1, "Privada": C2, "Federal": C3, "Estadual": C1, "Municipal": C4}
        order = ["Privada", "Federal", "Municipal", "Estadual"] if detalhe else ["Privada", "Pública"]

        two = rr_.assign(dep=rr_.dep.map(lambda d: "Privada" if d == "Privada" else "Pública"))
        s2 = summarize(two, ["ano", "dep"])
        cy = s2[s2.ano == yy].set_index("dep")
        if {"Pública", "Privada"} <= set(cy.index):
            d_, t_, cd_ = lib.welch_d(cy.loc["Privada", "m_GERAL"], cy.loc["Privada", "sd_GERAL"], cy.loc["Privada", "cnt_GERAL"],
                                      cy.loc["Pública", "m_GERAL"], cy.loc["Pública", "sd_GERAL"], cy.loc["Pública", "cnt_GERAL"])
            k = st.columns(5)
            k[0].metric("Média geral · pública", num_br(cy.loc["Pública", "m_GERAL"], 0), border=True)
            k[1].metric("Média geral · privada", num_br(cy.loc["Privada", "m_GERAL"], 0), border=True)
            k[2].metric("Diferença (privada − pública)", f"{d_:.0f} pts", border=True, help=f"Welch t = {t_:.0f}")
            k[3].metric("Tamanho do efeito (Cohen d)", dec(cd_), border=True, help="≥ 0,8 é considerado grande")
            k[4].metric("Média ≥ 700 (pública → privada)", f"{pct(cy.loc['Pública', 'ger_p700'], 0)} → {pct(cy.loc['Privada', 'ger_p700'], 0)}", border=True)

        a, b = st.columns(2)
        with a:
            d = sg_[sg_.ano == yy].melt(id_vars="dep", value_vars=[f"m_{k_}" for k_ in ("CN", "CH", "LC", "MT", "RED", "GERAL")])
            d["area"] = d.variable.str[2:].map(NOTA_NOME)
            fig = px.bar(d, x="area", y="value", color="dep", barmode="group", color_discrete_map=cor_g,
                         category_orders={"dep": order}, text=d.value.round(0))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(title=None); fig.update_yaxes(visible=False, range=[0, 1000 if True else None])
            chart(fig, f"Nota média por área · {yy}", "Média dos presentes nas 4 provas e na redação", h=380)
        with b:
            gp = s2.pivot(index="ano", columns="dep", values=[f"m_{k_}" for k_ in ("CN", "CH", "LC", "MT", "RED", "GERAL")])
            rows = []
            for k_ in ("CN", "CH", "LC", "MT", "RED", "GERAL"):
                for a_ in gp.index:
                    rows.append(dict(ano=a_, area=NOTA_NOME[k_], gap=gp.loc[a_, (f"m_{k_}", "Privada")] - gp.loc[a_, (f"m_{k_}", "Pública")]))
            g = pd.DataFrame(rows)
            fig = px.line(g, x="ano", y="gap", color="area", markers=True,
                          color_discrete_map={"Média geral": "#888888", "Redação": C2, "Matemática": C1, "Ciências Humanas": C3, "Linguagens": C4,
                                              "Ciências da Natureza": "#e87ba4"})
            fig.update_traces(line_width=2.5)
            fig.update_xaxes(dtick=1, title=None); fig.update_yaxes(title="pontos")
            fig.update_layout(hovermode="x unified")
            chart(fig, "Gap privada − pública ao longo dos anos", "Maior distância em Redação e Matemática", h=380,
                  insight=f"Em {yy}, a maior distância é em <b>{g[g.ano == yy].sort_values('gap').iloc[-1].area}</b> "
                          f"({g[g.ano == yy].gap.max():.0f} pts).")

        a, b = st.columns(2)
        with a:
            hh = load("hist_notas")
            nota_h = st.pills("Distribuição de", ["GERAL", "RED", "MT", "CN", "CH", "LC"], default="GERAL", key="pp_hist",
                              format_func=lambda x: NOTA_NOME[x], label_visibility="collapsed") or "GERAL"
            hh = hh[(hh.ano == yy) & (hh.nota == nota_h) & hh.rede.isin(["Pública", "Privada"])].copy()
            hh["dens"] = hh.n / hh.groupby("rede").n.transform("sum")
            fig = px.area(hh, x="bin", y="dens", color="rede", color_discrete_map={"Pública": C1, "Privada": C2})
            fig.update_traces(line_width=2, opacity=.55)
            fig.update_xaxes(title="Nota"); fig.update_yaxes(title=None, tickformat=".0%")
            fig.update_layout(hovermode="x unified")
            chart(fig, f"Distribuição · {NOTA_NOME[nota_h]} · {yy}", "Participação dos candidatos em cada faixa de 20 pontos (sem a rede 'sem info')", h=340)
        with b:
            u = two[two.ano == yy]
            su = summarize(u, ["uf", "dep"])
            pv = su.pivot(index="uf", columns="dep", values="m_GERAL")
            if {"Pública", "Privada"} <= set(pv.columns):
                gap_uf = (pv.Privada - pv.Pública).dropna()
                chart(tile_map(gap_uf, lambda x: f"{x:.0f}", h=380), f"Gap da média geral por UF · {yy}", "Pontos de diferença (privada − pública)", h=380, legend=False)

        section("Tabela detalhada")
        tt = sg_.assign(Ano=sg_.ano, Rede=sg_.dep, Presentes=sg_.cnt_GERAL, Geral=sg_.m_GERAL, CN=sg_.m_CN, CH=sg_.m_CH, LC=sg_.m_LC, MT=sg_.m_MT,
                        Redação=sg_.m_RED, Red700=sg_.red_p700, Zero=sg_.red_pzero)[
            ["Ano", "Rede", "Presentes", "Geral", "CN", "CH", "LC", "MT", "Redação", "Red700", "Zero"]].sort_values(["Ano", "Rede"], ascending=[False, True])
        st.dataframe(tt, hide_index=True, width="stretch", height=380, column_config={
            "Presentes": st.column_config.NumberColumn(format="%d"),
            **{c_: st.column_config.NumberColumn(format="%.0f") for c_ in ("Geral", "CN", "CH", "LC", "MT", "Redação")},
            "Red700": st.column_config.ProgressColumn("Redação ≥ 700", min_value=0, max_value=1, format="percent"),
            "Zero": st.column_config.ProgressColumn("Redação zero", min_value=0, max_value=1, format="percent")})
        buf2 = io.StringIO(); tt.to_csv(buf2, index=False)
        st.download_button("Baixar relação pública × privada (CSV)", buf2.getvalue(), "enem_publica_x_privada.csv", "text/csv")
        st.markdown("<div class='insight o'><b>Leitura com cuidado:</b> o gap mistura qualidade da escola com perfil socioeconômico dos alunos. "
                    "Controlando renda e escolaridade dos pais, a escola privada ainda se associa a cerca de +40 pontos (aba Possibilidades). "
                    "Em 2024 a base pública inclui muito mais candidatos vinculados ao Censo, o que muda a composição.</div>", unsafe_allow_html=True)



        section("Dispersão: alunos e escolas")
        st.caption("Cada ponto é um aluno (amostra individual 2020-2023, presentes nas 4 provas e na redação) ou uma escola (médias de 2024, com mínimo de presentes). "
                   "As retas são ajustes lineares por rede (statsmodels; nas escolas ponderados pelo nº de presentes).")
        AREAS_SC = {"MT": "Matemática", "CN": "Ciências da Natureza", "CH": "Ciências Humanas", "LC": "Linguagens", "RED": "Redação", "GERAL": "Média geral"}
        smp0 = load("amostra_regressao")
        anos_smp = sorted(int(a_) for a_ in smp0.ano.unique())
        c = st.columns([1.3, 1.5, 1.5, 1.2, 1.1])
        sc_ano = c[0].segmented_control("Ano (alunos)", anos_smp, default=anos_smp[-1], key="pp_sc_ano", required=True)
        xk = c[1].selectbox("Eixo X", list(AREAS_SC), index=0, format_func=AREAS_SC.get, key="pp_sc_x")
        yk = c[2].selectbox("Eixo Y", list(AREAS_SC), index=4, format_func=AREAS_SC.get, key="pp_sc_y")
        sc_min = c[3].slider("Mín. presentes/escola", 10, 80, 20, 5, key="pp_sc_min")
        sc_zero = c[4].toggle("Incluir notas zero", False, key="pp_sc_zero", help="Zero = prova anulada/eliminada ou redação zerada. Por padrão ficam fora do gráfico e do ajuste.")


        def _fit(x, y, w=None):
            X = sm.add_constant(np.asarray(x, float))
            m_ = (sm.WLS(np.asarray(y, float), X, weights=w) if w is not None else sm.OLS(np.asarray(y, float), X)).fit()
            return m_.params[0], m_.params[1], m_.rsquared


        def _scatter(df, xcol, ycol, hover, size=None, wcol=None, cap=6000):
            fig = go.Figure()
            stats = {}
            rng_ = np.random.default_rng(0)
            for rd, col_ in (("Pública", C1), ("Privada", C2)):
                d = df[df.rede == rd]
                if len(d) < 5:
                    continue
                a_, b_, r2 = _fit(d[xcol], d[ycol], d[wcol] if wcol else None)
                stats[rd] = (b_, np.sqrt(r2), len(d))
                v = d.sample(min(len(d), cap), random_state=1).copy()
                for ax_, k_ in ((xcol, xk), (ycol, yk)):  # redação vem em degraus de 20: jitter só na exibição
                    if k_ == "RED" and (v[ax_] % 20 == 0).mean() > .9:
                        v[ax_] = v[ax_] + rng_.uniform(-9, 9, len(v))
                fig.add_trace(go.Scattergl(x=v[xcol], y=v[ycol], mode="markers", name=f"{rd} ({len(d):,})".replace(",", "."),
                                           marker=dict(color=col_, opacity=.35, size=(np.clip(np.sqrt(v[size]) * 0.8, 3, 13) if size else 4)),
                                           customdata=v[hover].to_numpy(),
                                           hovertemplate="%{customdata[0]}<br>X: %{x:.0f} · Y: %{y:.0f}<extra>" + rd + "</extra>"))
                xs = np.array([d[xcol].quantile(.01), d[xcol].quantile(.99)])
                # retas em WebGL (mesma camada dos pontos) e com contorno branco para aparecerem sobre a nuvem
                fig.add_trace(go.Scattergl(x=xs, y=a_ + b_ * xs, mode="lines", line=dict(color="#ffffff", width=7), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scattergl(x=xs, y=a_ + b_ * xs, mode="lines", line=dict(color=col_, width=4), showlegend=False, hoverinfo="skip"))
            lo = float(min(df[xcol].quantile(.005), df[ycol].quantile(.005))); hi = float(max(df[xcol].quantile(.995), df[ycol].quantile(.995)))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(128,128,128,.6)", dash="dot", width=1.5), name="X = Y", hoverinfo="skip"))
            fig.update_xaxes(title=AREAS_SC[xk]); fig.update_yaxes(title=AREAS_SC[yk])
            return fig, stats


        def _txt(stats):
            return " · ".join(f"<b>{k_}</b>: r = {dec(v_[1])}, +100 pts em X → {v_[0] * 100:+.0f} pts em Y" for k_, v_ in stats.items())


        if xk == yk:
            st.info("Escolha áreas diferentes nos eixos X e Y.")
        else:
            a, b = st.columns(2)
            with a:
                sm_ = smp0[(smp0.ano == sc_ano) & smp0.uf.isin(ufs)].copy()
                sm_["rede"] = np.select([sm_.dep == "Privada", sm_.dep.isin(["Federal", "Estadual", "Municipal"]), sm_.escola_auto == "Privada", sm_.escola_auto == "Pública"],
                                        ["Privada", "Pública", "Privada", "Pública"], default="")
                sm_ = sm_[sm_.rede != ""].dropna(subset=["n_" + xk, "n_" + yk])
                sm_["x"], sm_["y"] = sm_["n_" + xk], sm_["n_" + yk]
                sm_["info"] = sm_.uf + " · " + sm_.sexo
                if not sc_zero:
                    sm_ = sm_[(sm_.x > 0) & (sm_.y > 0)]
                if len(sm_) > 50:
                    fig, stt = _scatter(sm_, "x", "y", ["info"])
                    chart(fig, f"Alunos · {AREAS_SC[xk]} × {AREAS_SC[yk]} · {sc_ano}", f"{len(sm_):,} alunos da amostra (até 6 mil pontos por rede no gráfico)".replace(",", "."),
                          h=500, insight=_txt(stt))
                else:
                    st.warning("Poucos alunos no recorte atual.")
            with b:
                ee = load("escolas_2024")
                ee = ee[(ee.n >= sc_min) & ee.uf.isin(ufs) & (ee.dep != "n/d")].copy()
                ee["rede"] = np.where(ee.dep == "Privada", "Privada", "Pública")
                mp_ = {"RED": "REDACAO"}
                ee["x"] = ee["s_" + mp_.get(xk, xk)] / ee.n
                ee["y"] = ee["s_" + mp_.get(yk, yk)] / ee.n
                ee["info"] = ee.mun + " · " + ee.uf + " · " + ee.dep + " · " + ee.n.astype(int).astype(str) + " presentes"
                if len(ee) > 30:
                    fig, stt = _scatter(ee, "x", "y", ["info"], size="n", wcol="n", cap=7000)
                    chart(fig, f"Escolas · {AREAS_SC[xk]} × {AREAS_SC[yk]} · 2024", f"{len(ee):,} escolas; o tamanho do ponto cresce com o nº de presentes".replace(",", "."),
                          h=500, insight=_txt(stt))
                else:
                    st.warning("Poucas escolas no recorte atual.")

# ============================================================ MUNICÍPIOS
with tabs[7]:
    if _active(7):
        section("Municípios por região")
        st.caption("Município de aplicação da prova. Inscritos existem para 2020-2025; nota, presença e comparecimento só para anos com resultados "
                   "(o ano sem resultado mostra o último ano publicado). Os filtros de região/UF da barra lateral valem aqui.")
        c = st.columns([1.6, 1, 1, 2])
        anoM = c[0].segmented_control("Ano", ANOS, default=ANOS[-1], key="mun_ano", required=True)
        min_insc = c[1].number_input("Mín. inscritos", 0, 50000, 500, 100, key="mun_min")
        min_pres = c[2].number_input("Mín. presentes p/ nota", 0, 5000, 100, 50, key="mun_minp")
        busca = c[3].text_input("Buscar município", "", key="mun_busca", placeholder="ex.: Fortaleza")
        yres = anoM if anoM in ANOS_RES else max([y_ for y_ in ANOS_RES if y_ <= anoM] or ANOS_RES)

        mu = MUN[MUN.uf.isin(ufs)].copy()
        mu["regiao"] = mu.uf.map(REGIAO)
        cur = mu[mu.ano == anoM][["regiao", "uf", "cod_mun", "mun", "n_insc"]].rename(columns={"n_insc": "insc"})
        prv_ = mu[mu.ano == anoM - 1][["cod_mun", "n_insc"]].rename(columns={"n_insc": "insc_ant"})
        rs = mu[mu.ano == yres][["cod_mun", "n_pres", "sum_GERAL", "cnt_GERAL"]]
        tb = cur.merge(prv_, on="cod_mun", how="left").merge(rs, on="cod_mun", how="left")
        tb["var"] = tb.insc / tb.insc_ant - 1
        tb["comparecimento"] = tb.n_pres / tb.insc
        tb["nota"] = tb.sum_GERAL / tb.cnt_GERAL
        tb["share"] = tb.insc / tb.insc.sum()
        tb = tb[tb.insc >= min_insc]
        if busca:
            tb = tb[tb.mun.str.contains(busca, case=False, na=False)]

        k = st.columns(4)
        k[0].metric("Municípios", num_br(len(tb)), border=True)
        k[1].metric("Inscritos", short(tb.insc.sum()), border=True)
        top_m = tb.nlargest(1, "insc")
        k[2].metric("Maior município", f"{top_m.mun.iloc[0]} · {top_m.uf.iloc[0]}" if len(top_m) else "–", short(top_m.insc.iloc[0]) if len(top_m) else None, border=True, delta_color="off")
        k[3].metric("Top 50 concentram", pct(tb.nlargest(50, "insc").insc.sum() / tb.insc.sum(), 0) if len(tb) else "–", border=True)

        a, b = st.columns(2)
        with a:
            rg = mu.groupby(["regiao", "ano"], as_index=False).agg(insc=("n_insc", "sum"))
            fig = px.line(rg, x="ano", y="insc", color="regiao", markers=True,
                          color_discrete_map={"Sudeste": C1, "Nordeste": C2, "Sul": C3, "Norte": C4, "Centro-Oeste": "#e87ba4"},
                          category_orders={"regiao": ["Sudeste", "Nordeste", "Sul", "Norte", "Centro-Oeste"]})
            fig.update_traces(line_width=3, marker_size=8)
            fig.update_xaxes(dtick=1, title=None); fig.update_yaxes(title=None)
            fig.update_layout(hovermode="x unified")
            chart(fig, "Inscritos por região ao longo dos anos", "Inclui 2025", h=360)
        with b:
            rr_ = tb.groupby("regiao").apply(lambda d: pd.Series({
                "Municípios": len(d), "Inscritos": d.insc.sum(), "Presentes": d.n_pres.sum(),
                "Nota média": d.sum_GERAL.sum() / d.cnt_GERAL.sum() if d.cnt_GERAL.sum() else np.nan,
                "Comparecimento": d.n_pres.sum() / d.insc.sum() if d.n_pres.sum() else np.nan}), include_groups=False).reset_index().rename(columns={"regiao": "Região"})
            rr_["Share"] = rr_.Inscritos / rr_.Inscritos.sum()
            rr_ = rr_.sort_values("Inscritos", ascending=False)
            st.markdown(f"**Resumo por região · {anoM}** <span class='note'>(nota e comparecimento de {yres})</span>", unsafe_allow_html=True)
            st.dataframe(rr_, hide_index=True, width="stretch", height=250, column_config={
                "Municípios": st.column_config.NumberColumn(format="%d"), "Inscritos": st.column_config.NumberColumn(format="%d"),
                "Presentes": st.column_config.NumberColumn(format="%d"), "Nota média": st.column_config.NumberColumn(format="%.0f"),
                "Comparecimento": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                "Share": st.column_config.ProgressColumn(min_value=0, max_value=float(rr_.Share.max()), format="percent")})
            st.markdown("<div class='insight'>Região é um recorte grosso: a dispersão <b>entre municípios da mesma região</b> costuma ser maior que "
                        "a diferença entre regiões (veja o box plot ao lado).</div>", unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            bx = tb[(tb.n_pres >= min_pres) & tb.nota.notna()]
            fig = px.box(bx, x="regiao", y="nota", points="outliers", color_discrete_sequence=[C1],
                         category_orders={"regiao": ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]}, hover_data=["mun", "uf", "n_pres"])
            fig.update_xaxes(title=None); fig.update_yaxes(title="Nota média geral do município")
            chart(fig, f"Nota média dos municípios por região · {yres}", f"Municípios com ≥ {min_pres} presentes ({len(bx)} municípios)", h=380, legend=False)
        with b:
            fig = px.scatter(bx, x="n_pres", y="nota", log_x=True, color_discrete_sequence=[C1], hover_data=["mun", "uf", "regiao"], opacity=.6)
            fig.update_xaxes(title="Presentes (escala log)"); fig.update_yaxes(title="Nota média geral")
            chart(fig, "Tamanho do município × nota", "Cada ponto é um município", h=380, legend=False)

        a, b = st.columns(2)
        with a:
            top = bx.nlargest(15, "nota").sort_values("nota")
            top["lbl"] = top.mun + " · " + top.uf
            fig = px.bar(top, x="nota", y="lbl", orientation="h", color_discrete_sequence=[C3], text=top.nota.round(0))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, top.nota.max() * 1.1] if len(top) else None); fig.update_yaxes(title=None)
            chart(fig, "15 maiores notas médias", f"Mín. {min_pres} presentes", h=440, legend=False)
        with b:
            bot = bx.nsmallest(15, "nota").sort_values("nota", ascending=False)
            bot["lbl"] = bot.mun + " · " + bot.uf
            fig = px.bar(bot, x="nota", y="lbl", orientation="h", color_discrete_sequence=[C2], text=bot.nota.round(0))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_xaxes(visible=False, range=[0, bot.nota.max() * 1.1] if len(bot) else None); fig.update_yaxes(title=None)
            chart(fig, "15 menores notas médias", f"Mín. {min_pres} presentes — onde o gap de preparação é maior", h=440, legend=False)

        section(f"Tabela de municípios · {anoM}")
        show_t = tb.sort_values("insc", ascending=False).rename(columns={
            "regiao": "Região", "uf": "UF", "mun": "Município", "insc": f"Inscritos {anoM}", "var": f"Δ vs {anoM-1}", "share": "Share",
            "n_pres": "Presentes", "comparecimento": "Comparec.", "nota": "Nota geral"})[
            ["Região", "UF", "Município", f"Inscritos {anoM}", f"Δ vs {anoM-1}", "Share", "Presentes", "Comparec.", "Nota geral"]]
        st.dataframe(show_t, hide_index=True, width="stretch", height=480, column_config={
            f"Inscritos {anoM}": st.column_config.NumberColumn(format="%d"), "Presentes": st.column_config.NumberColumn(format="%d"),
            f"Δ vs {anoM-1}": st.column_config.NumberColumn(format="percent"),
            "Share": st.column_config.ProgressColumn(min_value=0, max_value=float(show_t.Share.max()) if len(show_t) else 1, format="percent"),
            "Comparec.": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Nota geral": st.column_config.NumberColumn(format="%.0f")})
        b_ = io.StringIO(); show_t.to_csv(b_, index=False)
        st.download_button("Baixar municípios (CSV)", b_.getvalue(), f"enem_municipios_{anoM}.csv", "text/csv")

# ========================================================= ALUNO × ESCOLA
with tabs[8]:
    if _active(8):
        section("Gap aluno × escola · 2024")
        st.caption("Base: RESULTADOS 2024, que traz o código da escola (só 2024 tem). O INEP mascara escolas com menos de 10 participantes, "
                   "por isso há um mínimo de presentes. A nota é a média geral (4 provas + redação) dos presentes nos 2 dias.")
        ESC = load("escolas_2024")
        c = st.columns([1.2, 1, 2])
        min_n = c[0].slider("Mín. presentes por escola", 10, 80, 15, 5, key="esc_min")
        c[1].markdown("<br>", unsafe_allow_html=True)
        e = ESC[(ESC.n >= min_n) & ESC.uf.isin(ufs) & (ESC.dep != "n/d")].copy()
        e["rede"] = np.where(e.dep == "Privada", "Privada", "Pública")
        e["m"] = e.s_GERAL / e.n
        e["m_red"] = e.s_REDACAO / e.n
        e["p700"] = e.red700 / e.n
        e["sd"] = np.sqrt(((e.ss_GERAL - e.s_GERAL ** 2 / e.n) / (e.n - 1)).clip(lower=0))
        e["regiao"] = e.uf.map(REGIAO)

        def decomp(d):
            N, S, SS, J = d.n.sum(), d.s_GERAL.sum(), d.ss_GERAL.sum(), len(d)
            sst = SS - S ** 2 / N
            ssw = (d.ss_GERAL - d.s_GERAL ** 2 / d.n).sum()
            return dict(escolas=J, alunos=N, media=S / N, sd_total=np.sqrt(sst / (N - 1)), sd_dentro=np.sqrt(ssw / (N - J)),
                        icc=(sst - ssw) / sst, sd_entre=d.m.std())

        if len(e) < 20:
            st.warning("Poucas escolas no recorte atual; amplie regiões/UF ou reduza o mínimo de presentes.")
        else:
            all_ = decomp(e)
            k = st.columns(5)
            k[0].metric("Escolas analisadas", num_br(all_["escolas"]), border=True)
            k[1].metric("Alunos", short(all_["alunos"]), border=True)
            k[2].metric("A escola explica", pct(all_["icc"], 0), border=True, help="Parcela da variação das notas dos alunos que é diferença ENTRE escolas (ICC)")
            k[3].metric("Variação dentro da escola", f"±{all_['sd_dentro']:.0f} pts", border=True, help="Desvio-padrão da nota entre alunos da mesma escola")
            k[4].metric("Variação entre escolas", f"±{all_['sd_entre']:.0f} pts", border=True, help="Desvio-padrão das médias das escolas")
            st.markdown(f"<div class='insight'><b>Como ler o gap:</b> {pct(all_['icc'], 0)} da diferença de notas entre alunos é explicada pela escola em que estudam; "
                        f"os outros {pct(1 - all_['icc'], 0)} são diferenças entre alunos <i>dentro</i> da mesma escola (±{all_['sd_dentro']:.0f} pts). "
                        "Dentro de uma mesma escola há alunos bem acima e bem abaixo da média dela — espaço para personalização.</div>", unsafe_allow_html=True)

            rows = [("Todas", decomp(e))] + [(g_, decomp(e[e.rede == g_])) for g_ in ("Pública", "Privada") if (e.rede == g_).sum() > 10] + \
                   [(g_, decomp(e[e.dep == g_])) for g_ in ("Federal", "Estadual", "Municipal") if (e.dep == g_).sum() > 10]
            gd = pd.DataFrame([dict(grupo=n_, **v) for n_, v in rows])
            a, b = st.columns(2)
            with a:
                fig = px.bar(gd, x="grupo", y="icc", color_discrete_sequence=[C1], text=gd.icc.map(lambda v: pct(v, 0)))
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(title=None); fig.update_yaxes(visible=False, range=[0, gd.icc.max() * 1.25])
                chart(fig, "Quanto da nota é 'efeito escola' (ICC)", "Maior = alunos da mesma escola são mais parecidos entre si", h=340, legend=False)
            with b:
                d2 = gd.melt(id_vars="grupo", value_vars=["sd_dentro", "sd_entre"], var_name="tipo", value_name="pts")
                d2["tipo"] = d2.tipo.map({"sd_dentro": "Dentro da escola (entre alunos)", "sd_entre": "Entre escolas (médias)"})
                fig = px.bar(d2, x="grupo", y="pts", color="tipo", barmode="group", text=d2.pts.round(0),
                             color_discrete_map={"Dentro da escola (entre alunos)": C2, "Entre escolas (médias)": C1})
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(title=None); fig.update_yaxes(visible=False)
                chart(fig, "Dispersão em pontos", "Desvio-padrão da nota geral", h=340)

            a, b = st.columns(2)
            with a:
                fig = px.histogram(e, x="m", color="rede", nbins=60, barmode="overlay", histnorm="percent", opacity=.6,
                                   color_discrete_map={"Pública": C1, "Privada": C2})
                fig.update_xaxes(title="Média geral da escola"); fig.update_yaxes(title="% das escolas")
                med = e.groupby("rede").m.median()
                chart(fig, "Distribuição das médias das escolas", "Cada escola conta uma vez", h=360,
                      insight=f"Mediana: pública <b>{med.get('Pública', np.nan):.0f}</b> × privada <b>{med.get('Privada', np.nan):.0f}</b>. "
                              f"Só <b>{pct((e[e.rede == 'Pública'].m >= med.get('Privada', np.inf)).mean(), 1)}</b> das públicas alcançam a mediana das privadas.")
            with b:
                e["decil"] = pd.qcut(e.m.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
                dd = e.groupby("decil", observed=True).agg(m=("m", "mean"), esc=("m", "size")).reset_index()
                fig = px.bar(dd, x="decil", y="m", color_discrete_sequence=[C1], text=dd.m.round(0))
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_xaxes(title="Decil de escolas (D1 = piores)"); fig.update_yaxes(visible=False, range=[0, dd.m.max() * 1.15])
                chart(fig, "Média por decil de escolas", "Do pior ao melhor décimo", h=360, legend=False,
                      insight=f"Gap entre o melhor e o pior decil: <b>{dd.m.iloc[-1] - dd.m.iloc[0]:.0f} pontos</b>.")

            section("Escolas fora da curva")
            med_priv = e[e.rede == "Privada"].m.median()
            med_pub = e[e.rede == "Pública"].m.median()
            pub_acima = e[(e.rede == "Pública") & (e.m >= med_priv)]
            priv_abaixo = e[(e.rede == "Privada") & (e.m <= med_pub)]
            k = st.columns(4)
            k[0].metric("Públicas ≥ mediana das privadas", num_br(len(pub_acima)), pct(len(pub_acima) / max((e.rede == "Pública").sum(), 1), 1) + " das públicas", border=True, delta_color="off")
            k[1].metric("Privadas ≤ mediana das públicas", num_br(len(priv_abaixo)), pct(len(priv_abaixo) / max((e.rede == "Privada").sum(), 1), 1) + " das privadas", border=True, delta_color="off")
            k[2].metric("Mediana pública", f"{med_pub:.0f}", border=True)
            k[3].metric("Mediana privada", f"{med_priv:.0f}", border=True)

            a, b = st.columns([3, 2])
            with a:
                samp = e.sample(min(len(e), 7000), random_state=1)
                fig = go.Figure()
                for g_, col_ in (("Pública", C1), ("Privada", C2)):
                    d_ = samp[samp.rede == g_]
                    fig.add_trace(go.Scattergl(x=d_.m, y=d_.sd, mode="markers", name=g_, marker=dict(color=col_, size=5, opacity=.5),
                                               customdata=np.c_[d_.uf, d_.mun, d_.n], hovertemplate="%{customdata[1]} · %{customdata[0]}<br>média %{x:.0f} · dispersão %{y:.0f} · %{customdata[2]:.0f} alunos<extra>" + g_ + "</extra>"))
                fig.update_xaxes(title="Média da escola"); fig.update_yaxes(title="Dispersão dentro da escola (desvio-padrão)")
                chart(fig, "Média × dispersão de cada escola", "Alta média + baixa dispersão = ensino homogêneo; alta dispersão = escola 'puxada' por poucos", h=420)
            with b:
                gap_reg = e.groupby(["regiao", "rede"]).apply(lambda d: d.s_GERAL.sum() / d.n.sum(), include_groups=False).unstack()
                if {"Pública", "Privada"} <= set(gap_reg.columns):
                    gg = (gap_reg.Privada - gap_reg.Pública).rename("gap").reset_index().sort_values("gap")
                    fig = px.bar(gg, x="gap", y="regiao", orientation="h", color_discrete_sequence=[C2], text=gg.gap.round(0))
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    fig.update_xaxes(visible=False, range=[0, gg.gap.max() * 1.2]); fig.update_yaxes(title=None)
                    chart(fig, "Gap privada − pública por região", "Pontos na média geral (escolas com mín. de presentes)", h=420, legend=False)

            tabE = pd.DataFrame()
            sel = st.pills("Ver lista de", ["Públicas acima da mediana privada", "Melhores escolas", "Piores escolas", "Privadas abaixo da mediana pública"],
                           default="Públicas acima da mediana privada", key="esc_lista") or "Melhores escolas"
            src = {"Públicas acima da mediana privada": pub_acima.sort_values("m", ascending=False), "Melhores escolas": e.sort_values("m", ascending=False),
                   "Piores escolas": e.sort_values("m"), "Privadas abaixo da mediana pública": priv_abaixo.sort_values("m")}[sel]
            tabE = src.head(200).rename(columns={"escola": "Código INEP", "uf": "UF", "mun": "Município", "dep": "Rede", "loc": "Local.", "n": "Presentes",
                                                 "m": "Média geral", "m_red": "Redação", "p700": "Red ≥ 700", "sd": "Dispersão"})[
                ["Código INEP", "UF", "Município", "Rede", "Local.", "Presentes", "Média geral", "Redação", "Red ≥ 700", "Dispersão"]]
            st.dataframe(tabE, hide_index=True, width="stretch", height=420, column_config={
                "Código INEP": st.column_config.NumberColumn(format="%d"), "Presentes": st.column_config.NumberColumn(format="%d"),
                "Média geral": st.column_config.NumberColumn(format="%.0f"), "Redação": st.column_config.NumberColumn(format="%.0f"),
                "Dispersão": st.column_config.NumberColumn(format="%.0f"),
                "Red ≥ 700": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
            bb = io.StringIO(); e[["escola", "uf", "mun", "dep", "loc", "n", "m", "m_red", "p700", "sd"]].to_csv(bb, index=False)
            st.download_button("Baixar todas as escolas analisadas (CSV)", bb.getvalue(), "enem_escolas_2024.csv", "text/csv")

            section("Gap público × privado dentro do mesmo município")
            mm_ = e.groupby(["uf", "cod_mun", "mun", "rede"]).agg(n=("n", "sum"), s=("s_GERAL", "sum")).reset_index()
            mm_["m"] = mm_.s / mm_.n
            pv = mm_.pivot_table(index=["uf", "cod_mun", "mun"], columns="rede", values=["m", "n"]).dropna()
            if len(pv):
                pv = pv[(pv[("n", "Pública")] >= 30) & (pv[("n", "Privada")] >= 30)]
                gm = pd.DataFrame({"UF": pv.index.get_level_values(0), "Município": pv.index.get_level_values(2),
                                   "Pública": pv[("m", "Pública")].values, "Privada": pv[("m", "Privada")].values,
                                   "Alunos pública": pv[("n", "Pública")].values, "Alunos privada": pv[("n", "Privada")].values})
                gm["Gap"] = gm.Privada - gm.Pública
                a, b = st.columns([2, 3])
                with a:
                    st.markdown(f"**{len(gm)} municípios** com ≥ 30 presentes em cada rede. Gap mediano: **{gm.Gap.median():.0f} pts**.")
                    fig = px.histogram(gm, x="Gap", nbins=40, color_discrete_sequence=[C2])
                    fig.update_xaxes(title="Privada − pública (pts)"); fig.update_yaxes(visible=False)
                    chart(fig, "Distribuição do gap por município", None, h=280, legend=False)
                with b:
                    st.dataframe(gm.sort_values("Gap", ascending=False).head(100), hide_index=True, width="stretch", height=340, column_config={
                        c_: st.column_config.NumberColumn(format="%.0f") for c_ in ("Pública", "Privada", "Gap", "Alunos pública", "Alunos privada")})


# =========================================================== MAPA DO BRASIL
UF_COD = {"RO": "11", "AC": "12", "AM": "13", "RR": "14", "PA": "15", "AP": "16", "TO": "17", "MA": "21", "PI": "22", "CE": "23", "RN": "24",
          "PB": "25", "PE": "26", "AL": "27", "SE": "28", "BA": "29", "MG": "31", "ES": "32", "RJ": "33", "SP": "35", "PR": "41", "SC": "42",
          "RS": "43", "MS": "50", "MT": "51", "GO": "52", "DF": "53"}
COD_UF = {v: k for k, v in UF_COD.items()}


@st.cache_resource(show_spinner="Carregando malha do IBGE…")
def load_geo():
    import json
    from pathlib import Path
    g = Path(__file__).resolve().parent / "geo"
    return json.load(open(g / "br_municipios.json")), json.load(open(g / "br_uf.json"))


@st.cache_data(show_spinner=False)
def uf_view(ufs_t):
    """Centro e zoom que enquadram as UFs selecionadas."""
    _, geo_uf = load_geo()
    xs, ys = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            xs.append(c[0]); ys.append(c[1])
        else:
            for k_ in c:
                walk(k_)
    for f in geo_uf["features"]:
        if COD_UF.get(f["properties"]["codarea"]) in ufs_t:
            walk(f["geometry"]["coordinates"])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    lat = (y0 + y1) / 2
    span_x, span_y = max(x1 - x0, 0.5), max((y1 - y0) / np.cos(np.radians(lat)), 0.5)
    zoom = min(np.log2(360 * 950 / 512 / span_x), np.log2(360 * 720 / 512 / span_y)) - 0.15
    return dict(lat=lat, lon=(x0 + x1) / 2), float(np.clip(zoom, 2.5, 8))


@st.cache_data(show_spinner=False)
def map_data(ano, nivel, rede, min_pres, ufs_t):
    d = load("cubo_mun_escola")
    d = d[d.ano == ano]
    if rede != "Todas":
        d = d[d.rede == rede]
    nac = d[["s_GERAL", "cnt", "n_pres", "n_insc"]].sum()
    d = d[d.uf.isin(ufs_t)]
    keys = ["uf"] if nivel == "UF" else ["cod_mun", "uf", "mun"]
    g = d.groupby(keys, as_index=False)[["n_insc", "n_pres", "cnt", "s_GERAL", "red700"]].sum()
    g["nota"] = g.s_GERAL / g.cnt
    g["part"] = g.n_pres / g.n_insc
    if nivel != "UF":
        g.loc[g.n_pres < min_pres, ["nota", "part"]] = np.nan
    g["loc"] = g.uf.map(UF_COD) if nivel == "UF" else g.cod_mun.astype(int).astype(str)
    nome = g.uf if nivel == "UF" else g.mun + " · " + g.uf
    g["hover"] = ("<b>" + nome + "</b><br>Nota média: " + g.nota.round(0).astype("Int64").astype(str) + "<br>Participação: " +
                  (g.part * 100).round(1).astype(str) + "%<br>Inscritos: " + g.n_insc.astype(int).map("{:,}".format).str.replace(",", ".") +
                  "<br>Presentes: " + g.n_pres.astype(int).map("{:,}".format).str.replace(",", "."))
    return g, float(nac.s_GERAL / nac.cnt), float(nac.n_pres / nac.n_insc)


def build_map(g, visao, nivel, nota_ref, part_ref, ufs_t, dark):
    geo_m, geo_uf = load_geo()
    bg, nodata = ("#0e1117", "#262d3a") if dark else ("#ffffff", "#e6e9ef")
    border = "rgba(255,255,255,.75)" if dark else "rgba(30,40,60,.55)"
    mid = "#aab0bb"
    gm = geo_uf if nivel == "UF" else geo_m
    all_uf = list(UF_COD.values())
    fig = go.Figure()
    fig.add_trace(go.Choroplethmap(geojson=geo_uf, locations=all_uf, z=[0] * len(all_uf), featureidkey="properties.codarea",
                                   colorscale=[[0, nodata], [1, nodata]], showscale=False, marker_line_width=0, hoverinfo="skip"))
    quad = None
    common = dict(geojson=gm, locations=g["loc"], featureidkey="properties.codarea", text=g.hover, hovertemplate="%{text}<extra></extra>",
                  marker_line_width=0.15 if nivel != "UF" else 0.5, marker_line_color="rgba(255,255,255,.25)", marker_opacity=0.96)
    if visao == "Nota × Participação":
        q = ((g.nota >= nota_ref).astype(int) * 2 + (g.part >= part_ref).astype(int)).where(g.nota.notna())
        cols = [C2, C4, C1, C3]
        scale = []
        for i, c_ in enumerate(cols):
            scale += [[i / 4, c_], [(i + 1) / 4, c_]]
        fig.add_trace(go.Choroplethmap(z=q, colorscale=scale, zmin=-0.5, zmax=3.5, showscale=False, **common))
        quad = g.assign(q=q)
    elif visao == "Inscritos":
        z = np.log10(g.n_insc.clip(lower=1))
        fig.add_trace(go.Choroplethmap(z=z, colorscale=SEQ, zmin=0.7, zmax=max(float(z.quantile(.99)), 2),
                                       colorbar=dict(title="Inscritos", tickvals=[1, 2, 3, 4, 5], ticktext=["10", "100", "1 mil", "10 mil", "100 mil"], len=.55, x=.02, xanchor="left"),
                                       **common))
    else:
        col, ref = ("nota", nota_ref) if visao == "Nota média" else ("part", part_ref)
        z = g[col]
        span = max(abs(z.quantile(.03) - ref), abs(z.quantile(.97) - ref), 1e-6)
        fig.add_trace(go.Choroplethmap(z=z, colorscale=[[0, C2], [0.5, mid], [1, C1]], zmin=ref - span, zmax=ref + span,
                                       colorbar=dict(title="Participação" if col == "part" else "Nota", tickformat=".0%" if col == "part" else ".0f", len=.55, x=.02, xanchor="left"),
                                       **common))
    sel = [UF_COD[u] for u in ufs_t if u in UF_COD]
    fig.add_trace(go.Choroplethmap(geojson=geo_uf, locations=sel, z=[0] * len(sel), featureidkey="properties.codarea",
                                   colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False, hoverinfo="skip",
                                   marker_line_width=0.9, marker_line_color=border))
    center, zoom = uf_view(ufs_t)
    world = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {
        "type": "Polygon", "coordinates": [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]]]}}]}
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      map=dict(style="white-bg", center=center, zoom=zoom,
                               layers=[dict(sourcetype="geojson", source=world, type="fill", color=bg, below="traces")]))
    return fig, quad


@st.fragment
def mapa_ui():
    section("Mapa do Brasil · nota e participação")
    st.caption("Base: município **da escola** (cobre ~5,5 mil municípios), candidatos com escola vinculada no Censo — em geral concluintes do ensino médio. "
               "Participação = compareceram nos 2 dias ÷ inscritos. 2025 não entra: o pacote não traz notas nem presença. "
               "Use o scroll para dar zoom e arraste para mover.")
    c = st.columns([1.2, 1.1, 2.2, 1.2, 1])
    ano_m = c[0].segmented_control("Ano", ANOS_RES, default=ANOS_RES[-1], key="mp_ano", required=True)
    nivel = c[1].segmented_control("Nível", ["Município", "UF"], default="Município", key="mp_nivel", required=True)
    visao = c[2].segmented_control("Visão", ["Nota × Participação", "Nota média", "Participação", "Inscritos"], default="Nota × Participação", key="mp_visao", required=True)
    rede_m = c[3].segmented_control("Rede", ["Todas", "Pública", "Privada"], default="Todas", key="mp_rede", required=True)
    minp = c[4].number_input("Mín. presentes", 1, 500, 20, 5, key="mp_min", help="Municípios com menos presentes ficam sem cor (nota instável)", disabled=nivel == "UF")
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    g, nota_ref, part_ref = map_data(ano_m, nivel, rede_m, int(minp), tuple(ufs))
    fig, quad = build_map(g, visao, nivel, nota_ref, part_ref, tuple(ufs), dark)
    a, b = st.columns([3, 1.1])
    with a:
        with st.container(border=True):
            st.plotly_chart(fig, width="stretch", key="mapa_brasil", config={"scrollZoom": True, "displayModeBar": False})
    with b:
        st.metric("Nota média nacional", f"{nota_ref:.0f}", border=True, help=f"{rede_m} · {ano_m}")
        st.metric("Participação nacional", pct(part_ref), border=True)
        if quad is not None:
            nomes = {3: ("Alta nota · alta participação", C3), 2: ("Alta nota · baixa participação", C1),
                     1: ("Baixa nota · alta participação", C4), 0: ("Baixa nota · baixa participação", C2)}
            qq = quad.dropna(subset=["q"]).groupby("q").agg(mun=("q", "size"), cand=("n_insc", "sum")).reindex([3, 2, 1, 0]).fillna(0)
            tot = max(qq.cand.sum(), 1)
            html = "<div style='margin-top:6px'>"
            for k_, (lbl, col_) in nomes.items():
                html += (f"<div style='display:flex;gap:10px;align-items:center;margin:9px 0'>"
                         f"<span style='width:16px;height:16px;border-radius:4px;background:{col_};flex:none'></span>"
                         f"<div style='font-size:.88rem'><b>{lbl}</b><br><span class='note'>{int(qq.loc[k_, 'mun']):,} {'UFs' if nivel == 'UF' else 'municípios'} · "
                         f"{pct(qq.loc[k_, 'cand'] / tot, 0)} dos candidatos</span></div></div>".replace(",", "."))
            st.markdown(html + "</div>", unsafe_allow_html=True)
            st.markdown("<div class='insight'>Cortes: <b>nota média</b> e <b>participação</b> nacionais. "
                        "Baixa nota + baixa participação = onde o gap de acesso e de preparação se somam.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='insight'>Azul = acima da média nacional; laranja = abaixo. "
                        "Passe o mouse para ver nota, participação e inscritos.</div>", unsafe_allow_html=True)

    if quad is not None and nivel == "Município":
        section("Municípios por quadrante")
        escolha = st.pills("Quadrante", ["Baixa nota · baixa participação", "Alta nota · baixa participação", "Baixa nota · alta participação", "Alta nota · alta participação"],
                           default="Baixa nota · baixa participação", key="mp_quad") or "Baixa nota · baixa participação"
        qmap = {"Baixa nota · baixa participação": 0, "Baixa nota · alta participação": 1, "Alta nota · baixa participação": 2, "Alta nota · alta participação": 3}
        sub = quad[quad.q == qmap[escolha]].copy()
        sub["Região"] = sub.uf.map(REGIAO)
        sub = sub.sort_values("n_insc", ascending=False)
        a, b = st.columns([2, 3])
        with a:
            rg = sub.groupby("Região").agg(Municípios=("q", "size"), Candidatos=("n_insc", "sum")).reset_index().sort_values("Candidatos")
            fig2 = px.bar(rg, x="Candidatos", y="Região", orientation="h", color_discrete_sequence=[C1], text=rg.Candidatos.map(short))
            fig2.update_traces(textposition="outside", cliponaxis=False)
            fig2.update_xaxes(visible=False, range=[0, rg.Candidatos.max() * 1.2] if len(rg) else None); fig2.update_yaxes(title=None)
            chart(fig2, f"Candidatos por região · {escolha}", f"{len(sub):,} municípios".replace(",", "."), h=300, legend=False)
        with b:
            t = sub.rename(columns={"mun": "Município", "uf": "UF", "n_insc": "Inscritos", "n_pres": "Presentes", "nota": "Nota", "part": "Participação"})[
                ["Município", "UF", "Região", "Inscritos", "Presentes", "Nota", "Participação"]]
            st.dataframe(t, hide_index=True, width="stretch", height=300, column_config={
                "Inscritos": st.column_config.NumberColumn(format="%d"), "Presentes": st.column_config.NumberColumn(format="%d"),
                "Nota": st.column_config.NumberColumn(format="%.0f"),
                "Participação": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
            bm = io.StringIO(); t.to_csv(bm, index=False)
            st.download_button("Baixar lista (CSV)", bm.getvalue(), f"enem_mapa_{ano_m}.csv", "text/csv")


with tabs[9]:
    if _active(9) or st.session_state.get("_force_map"):
        mapa_ui()
