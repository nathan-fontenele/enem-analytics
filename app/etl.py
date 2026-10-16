"""
ETL ENEM 2020-2024  ->  cubos agregados (parquet) para o dashboard.

Lê os CSVs gigantes (~8 GB) em streaming com pyarrow, harmoniza os esquemas
entre os anos e grava tabelas pequenas em app/data/.

Uso:
    .venv/bin/python app/etl.py            # roda tudo (um processo por tarefa)
    .venv/bin/python app/etl.py --limit 2  # teste rápido: 2 blocos por arquivo

Particularidades dos dados (tratadas aqui):
  * 2020-2023: um único MICRODADOS_ENEM_AAAA.csv (socioeconômico + notas + escola).
  * 2024: o INEP separou em PARTICIPANTES (socioeconômico, sem notas/presença) e
    RESULTADOS (notas, presença, escola) -- sem chave para cruzar as duas bases.
  * Renda: Q006 (2020-23) == Q007 (2024); as faixas são múltiplos do salário
    mínimo do ano, então são comparáveis em "SM".
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

YEARS = [2020, 2021, 2022, 2023, 2024]
BLOCK = 64 << 20

# ----------------------------------------------------------------- caminhos
def micro_path(y):
    return ROOT / f"microdados_enem_{y}" / "DADOS" / f"MICRODADOS_ENEM_{y}.csv"

def itens_path(y):
    return ROOT / f"microdados_enem_{y}" / "DADOS" / f"ITENS_PROVA_{y}.csv"

def comp_path(y):
    for p in (ROOT / f"COMPETENCIAS_REDACAO_{y}" / "DADOS",
              ROOT / f"competencias_redacao_enem_{y}" / f"COMPETENCIAS_REDACAO_{y}" / "DADOS"):
        f = p / f"COMPETENCIAS_REDACAO_{y}.csv"
        if f.exists():
            return f
    raise FileNotFoundError(y)

# ------------------------------------------------------------ mapeamentos
RENDA_GRUPO = {
    "A": "Sem renda", "B": "Até 1 SM", "C": "1 a 2 SM", "D": "1 a 2 SM",
    "E": "2 a 3 SM", "F": "2 a 3 SM", "G": "3 a 5 SM", "H": "3 a 5 SM",
    "I": "5 a 10 SM", "J": "5 a 10 SM", "K": "5 a 10 SM", "L": "5 a 10 SM", "M": "5 a 10 SM",
    "N": "Mais de 10 SM", "O": "Mais de 10 SM", "P": "Mais de 10 SM", "Q": "Mais de 10 SM",
}
# ponto médio da faixa em salários mínimos (usado só em análises quantitativas)
RENDA_SM_MID = dict(zip("ABCDEFGHIJKLMNOPQ",
                        [0, .5, 1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11, 13.5, 17.5, 25]))
ESC_GRUPO = {"A": "Até fund. incompleto", "B": "Até fund. incompleto", "C": "Até fund. incompleto",
             "D": "Fund. completo / médio incompleto", "E": "Médio completo",
             "F": "Superior", "G": "Superior", "H": "Não sabe"}
ESC_ORD = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
DEP = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
REDE_DEP = {"Federal": "Pública", "Estadual": "Pública", "Municipal": "Pública", "Privada": "Privada"}
AREAS = ["CN", "CH", "LC", "MT"]
AREA_BASE = {"LC": 1, "CH": 46, "CN": 91, "MT": 136}


def _encoding(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(4 << 20)
    try:
        head.decode("utf-8")
        return "utf8"
    except UnicodeDecodeError:
        try:  # corte no meio de um caractere multibyte?
            head[:-4].decode("utf-8")
            return "utf8"
        except UnicodeDecodeError:
            return "latin1"


def stream(path: Path, cols: list[str], limit: int | None = None):
    ro = pcsv.ReadOptions(block_size=BLOCK, encoding=_encoding(path))
    po = pcsv.ParseOptions(delimiter=";")
    co = pcsv.ConvertOptions(include_columns=cols, column_types={c: pa.string() for c in cols},
                             null_values=[""], strings_can_be_null=True)
    rd = pcsv.open_csv(path, read_options=ro, parse_options=po, convert_options=co)
    for i, b in enumerate(rd):
        if limit and i >= limit:
            break
        yield b.to_pandas()


def num(s):
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------- harmonização
def _common_geo(o, df):
    o["uf"] = df["SG_UF_PROVA"].fillna("n/d")
    o["cod_mun"] = num(df["CO_MUNICIPIO_PROVA"])
    o["mun"] = df["NO_MUNICIPIO_PROVA"].fillna("n/d")


def _socio(o, df, y):
    o["sexo"] = df["TP_SEXO"].fillna("n/d")
    o["faixa"] = num(df["TP_FAIXA_ETARIA"]).fillna(0).astype(int)
    o["cor"] = num(df["TP_COR_RACA"]).fillna(0).astype(int)
    o["st"] = num(df["TP_ST_CONCLUSAO"]).fillna(0).astype(int)
    o["treineiro"] = num(df["IN_TREINEIRO"]).fillna(0).astype(int)
    o["anos_concl"] = num(df["TP_ANO_CONCLUIU"]).fillna(0).astype(int)
    if y <= 2023:
        renda, mae, pai, net, pc = df["Q006"], df["Q002"], df["Q001"], df["Q025"], df["Q024"]
        te = num(df["TP_ESCOLA"])
        o["escola_auto"] = te.map({2: "Pública", 3: "Privada"}).fillna("Sem info")
        o["computador"] = (pc.notna() & (pc != "A")).astype(int)
    else:
        renda, mae, pai, net, pc = df["Q007"], df["Q002"], df["Q001"], df["Q020"], df["Q021"]
        q23 = df["Q023"]
        o["escola_auto"] = q23.map({"A": "Pública", "D": "Privada", "E": "Privada",
                                    "B": "Mista", "C": "Mista"}).fillna("Sem info")
        o["computador"] = (pc.notna() & (pc != "A")).astype(int)
    o["renda_letra"] = renda
    o["renda"] = renda.map(RENDA_GRUPO).fillna("Sem info")
    o["renda_sm"] = renda.map(RENDA_SM_MID)
    o["esc_mae"] = mae.map(ESC_GRUPO).fillna("Sem info")
    o["esc_pai"] = pai.map(ESC_GRUPO).fillna("Sem info")
    o["esc_mae_ord"] = mae.map(ESC_ORD)
    o["esc_pai_ord"] = pai.map(ESC_ORD)
    o["internet"] = (net == "B").astype(int)


def _resultados(o, df):
    for a in AREAS:
        o["pres_" + a] = num(df["TP_PRESENCA_" + a]).fillna(0).astype(int)
    o["pres_d1"] = ((o.pres_CH == 1) & (o.pres_LC == 1)).astype(int)
    o["pres_d2"] = ((o.pres_CN == 1) & (o.pres_MT == 1)).astype(int)
    o["pres_all"] = o.pres_d1 * o.pres_d2
    for a in AREAS:
        o["n_" + a] = num(df["NU_NOTA_" + a])
    o["n_RED"] = num(df["NU_NOTA_REDACAO"])
    o["red_status"] = num(df["TP_STATUS_REDACAO"])
    for k in range(1, 6):
        o[f"c{k}"] = num(df[f"NU_NOTA_COMP{k}"])
    full = o[["n_CN", "n_CH", "n_LC", "n_MT", "n_RED"]]
    o["n_GERAL"] = full.mean(axis=1).where(full.notna().all(axis=1))
    dep = num(df["TP_DEPENDENCIA_ADM_ESC"]).map(DEP)
    o["dep"] = dep.fillna("Sem info")
    o["rede"] = dep.map(REDE_DEP).fillna("Sem info")


def harmonize_micro(df, y):
    o = pd.DataFrame(index=df.index)
    o["ano"] = y
    _common_geo(o, df)
    _socio(o, df, y)
    _resultados(o, df)
    return o


def harmonize_part_2024(df, y=2024):
    o = pd.DataFrame(index=df.index)
    o["ano"] = y
    _common_geo(o, df)
    _socio(o, df, y)
    return o


def harmonize_res_2024(df, y=2024):
    o = pd.DataFrame(index=df.index)
    o["ano"] = y
    o["uf"] = df["SG_UF_PROVA"].fillna("n/d")
    o["cod_mun"] = num(df["CO_MUNICIPIO_PROVA"])
    o["mun"] = df["NO_MUNICIPIO_PROVA"].fillna("n/d")
    _resultados(o, df)
    # sem socioeconômico nessa base
    o["renda"] = "n/d"
    o["st"] = -1
    return o


# --------------------------------------------------------- agregadores
NOTAS = ["CN", "CH", "LC", "MT", "RED", "GERAL"]


def agg_resultados(o: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    t = o[dims].copy()
    t["n_insc"] = 1
    for c in ("pres_d1", "pres_d2", "pres_all"):
        t[c] = o[c]
    for k in NOTAS:
        v = o["n_" + k]
        t[f"cnt_{k}"] = v.notna().astype(int)
        t[f"sum_{k}"] = v.fillna(0)
        t[f"ssq_{k}"] = (v ** 2).fillna(0)
    r = o["n_RED"]
    t["red_ge700"] = (r >= 700).astype(int)
    t["red_ge900"] = (r >= 900).astype(int)
    t["red_1000"] = (r == 1000).astype(int)
    t["red_zero"] = (r == 0).astype(int)
    g = o["n_GERAL"]
    t["ger_ge600"] = (g >= 600).astype(int)
    t["ger_ge700"] = (g >= 700).astype(int)
    for k in range(1, 6):
        c = o[f"c{k}"]
        t[f"cntc{k}"] = c.notna().astype(int)
        t[f"sumc{k}"] = c.fillna(0)
    return t.groupby(dims, observed=True, dropna=False).sum().reset_index()


def agg_hist(o: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for k in NOTAS:
        v = o[["ano", "rede", "n_" + k]].dropna()
        if k != "RED":
            v = v[v["n_" + k] > 0]
        b = (v["n_" + k] // 20).clip(0, 50).astype(int)
        h = v.assign(bin=b * 20).groupby(["ano", "rede", "bin"]).size().rename("n").reset_index()
        h["nota"] = k
        parts.append(h)
    return pd.concat(parts)


def agg_socio(p: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = p.assign(n_insc=1)
    has_pres = "pres_all" in p
    if has_pres:
        base["n_pres"] = p["pres_all"]
    else:
        base["n_pres"] = np.nan
    out = {}
    def g(name, dims):
        d = base[dims + ["n_insc", "n_pres"]].groupby(dims, observed=True, dropna=False).sum(min_count=1).reset_index()
        out[name] = d
    g("socio", ["ano", "uf", "st", "escola_auto", "renda", "internet", "treineiro"])
    g("demo", ["ano", "uf", "sexo", "faixa", "cor", "st"])
    g("familia", ["ano", "esc_mae", "esc_pai", "renda", "escola_auto", "computador", "internet"])
    eg = base[base.st == 1].copy()
    eg["anos_bin"] = pd.cut(eg.anos_concl, [-1, 0, 1, 2, 3, 5, 10, 99],
                            labels=["n/i", "1 ano", "2 anos", "3 anos", "4-5 anos", "6-10 anos", "11+ anos"]).astype(str)
    out["egressos"] = eg[["ano", "anos_bin", "escola_auto", "renda", "n_insc", "n_pres"]].groupby(
        ["ano", "anos_bin", "escola_auto", "renda"], observed=True, dropna=False).sum(min_count=1).reset_index()
    return out


def agg_mun(o):
    t = o[["ano", "uf", "cod_mun", "mun"]].copy()
    t["n_insc"] = 1
    if "pres_all" in o:
        t["n_pres"] = o["pres_all"]
        t["sum_GERAL"] = o["n_GERAL"].fillna(0)
        t["cnt_GERAL"] = o["n_GERAL"].notna().astype(int)
    return t.groupby(["ano", "uf", "cod_mun", "mun"], observed=True, dropna=False).sum().reset_index()


# --------------------------------------------------------- itens
def item_hits(df: pd.DataFrame, y: int, acc: dict):
    rede = df["TP_DEPENDENCIA_ADM_ESC"].pipe(num).map(DEP).map(REDE_DEP)
    lingua_all = num(df["TP_LINGUA"])
    for a in AREAS:
        pres = num(df["TP_PRESENCA_" + a]) == 1
        r = df["TX_RESPOSTAS_" + a]
        g = df["TX_GABARITO_" + a]
        m = pres & r.notna() & g.notna()
        if a == "LC":
            m &= lingua_all.notna() & (g.str.len() == 50)
        else:
            m &= g.str.len() == 45
        m &= r.str.len() >= 45
        if not m.any():
            continue
        r = r[m]
        g = g[m]
        ling = lingua_all[m].astype(int) if a == "LC" else pd.Series(-1, index=r.index)
        if a == "LC":
            # gabarito: 5 itens de inglês + 5 de espanhol + 40 comuns. Em 2020-21 as respostas
            # também vêm com 50 posições (mesmo layout); a partir de 2022 já vêm com 45.
            pick = lambda x: pd.Series(np.where(ling == 0, x.str.slice(0, 5), x.str.slice(5, 10)), index=x.index) + x.str.slice(10, 50)
            g = pick(g)
            r = pick(r).where(r.str.len() == 50, r.str.slice(0, 45))
        else:
            r = r.str.slice(0, 45)
        prova = num(df["CO_PROVA_" + a])[m].astype(int)
        rd = rede[m]
        R = np.frombuffer("".join(r.tolist()).encode("latin1"), dtype=np.uint8).reshape(-1, 45)
        G = np.frombuffer("".join(g.tolist()).encode("latin1"), dtype=np.uint8).reshape(-1, 45)
        valid = (G != ord("X")) & (R != ord("9"))
        hit = (R == G) & valid
        pv, lv, rv = prova.to_numpy(), ling.to_numpy(), rd.fillna("").to_numpy()
        for p, l in pd.DataFrame({"p": pv, "l": lv}).drop_duplicates().itertuples(index=False):
            base = (pv == p) & (lv == l)
            for lbl, mm in (("Todos", base), ("Pública", base & (rv == "Pública")), ("Privada", base & (rv == "Privada"))):
                if not mm.any():
                    continue
                k = (y, a, int(p), lbl, int(l))
                h, v = hit[mm].sum(0).astype(np.int64), valid[mm].sum(0).astype(np.int64)
                if k in acc:
                    acc[k][0] += h
                    acc[k][1] += v
                else:
                    acc[k] = [h, v]


def finalize_items(acc, y):
    rows = []
    for (yy, a, p, rd, l), (h, v) in acc.items():
        for i in range(45):
            rows.append((yy, a, p, rd, l, i, int(h[i]), int(v[i])))
    d = pd.DataFrame(rows, columns=["ano", "area", "prova", "rede", "lingua", "idx", "hits", "valid"])
    if d.empty:
        return d
    # itens de língua estrangeira só nas 5 primeiras posições; o resto é comum
    d.loc[~((d.area == "LC") & (d.idx < 5)), "lingua"] = -1
    d = d.groupby(["ano", "area", "prova", "rede", "lingua", "idx"], as_index=False)[["hits", "valid"]].sum()
    it = pd.read_csv(itens_path(y), sep=";", encoding=_encoding(itens_path(y)).replace("utf8", "utf-8"))
    it = it[it.IN_ITEM_ABAN == 0]
    it["idx"] = it.CO_POSICAO - it.SG_AREA.map(AREA_BASE)  # posição global -> índice 0..44 dentro da área
    it["lingua"] = it.TP_LINGUA.fillna(-1).astype(int)
    it = it.rename(columns={"SG_AREA": "area", "CO_PROVA": "prova"})
    m = d.merge(it[["area", "prova", "idx", "lingua", "CO_ITEM", "CO_HABILIDADE", "NU_PARAM_A", "NU_PARAM_B", "NU_PARAM_C"]],
                on=["area", "prova", "idx", "lingua"], how="inner")
    m = m.groupby(["ano", "area", "rede", "CO_ITEM", "CO_HABILIDADE", "NU_PARAM_A", "NU_PARAM_B", "NU_PARAM_C"],
                  as_index=False)[["hits", "valid"]].sum()
    return m


# ----------------------------------------------------------- tarefas
def task_micro(y, limit=None):
    t0 = time.time()
    cols = pd.read_csv(micro_path(y), sep=";", nrows=0, encoding="latin1").columns.tolist()
    A = {k: [] for k in ("res", "res_geo", "hist", "mun", "socio", "demo", "familia", "egressos", "sample")}
    acc = {}
    rng = np.random.default_rng(y)
    for i, df in enumerate(stream(micro_path(y), cols, limit)):
        o = harmonize_micro(df, y)
        A["res"].append(agg_resultados(o, ["ano", "uf", "dep", "renda", "st"]))
        A["hist"].append(agg_hist(o))
        A["mun"].append(agg_mun(o))
        for k, v in agg_socio(o).items():
            A[k].append(v)
        item_hits(df, y, acc)
        s = o[(o.pres_all == 1) & o.n_GERAL.notna() & (o.treineiro == 0)]
        s = s.sample(frac=0.02, random_state=int(rng.integers(1e9)))
        A["sample"].append(s[["ano", "uf", "sexo", "faixa", "cor", "st", "escola_auto", "dep", "renda_sm", "esc_mae_ord",
                              "esc_pai_ord", "internet", "computador", "n_CN", "n_CH", "n_LC", "n_MT", "n_RED", "n_GERAL"]])
        print(f"[{y}] bloco {i} ok  ({time.time()-t0:.0f}s)", flush=True)
    return y, A, finalize_items(acc, y)


def find_csv(y, name):
    hits = sorted(ROOT.glob(f"**/DADOS/{name}_{y}.csv"))
    return hits[0] if hits else None


def task_new(y, limit=None):
    """Anos no formato novo (2024+): PARTICIPANTES (perfil) e, se existir, RESULTADOS (notas)."""
    t0 = time.time()
    A = {k: [] for k in ("res", "hist", "mun", "socio", "demo", "familia", "egressos", "sample")}
    cp = find_csv(y, "PARTICIPANTES")
    cols = pd.read_csv(cp, sep=";", nrows=0, encoding="latin1").columns.tolist()
    for i, df in enumerate(stream(cp, cols, limit)):
        o = harmonize_part_2024(df, y)
        for k, v in agg_socio(o).items():
            A[k].append(v)
        if find_csv(y, "RESULTADOS") is None:
            A["mun"].append(agg_mun(o))
        print(f"[{y} part] bloco {i} ok ({time.time()-t0:.0f}s)", flush=True)
    items = pd.DataFrame()
    rp = find_csv(y, "RESULTADOS")
    if rp is not None:
        cols = pd.read_csv(rp, sep=";", nrows=0, encoding="latin1").columns.tolist()
        acc = {}
        for i, df in enumerate(stream(rp, cols, limit)):
            o = harmonize_res_2024(df, y)
            A["res"].append(agg_resultados(o, ["ano", "uf", "dep", "renda", "st"]))
            A["hist"].append(agg_hist(o))
            A["mun"].append(agg_mun(o))
            item_hits(df, y, acc)
            print(f"[{y} res] bloco {i} ok ({time.time()-t0:.0f}s)", flush=True)
        items = finalize_items(acc, y)
    else:
        print(f"[{y}] sem RESULTADOS_{y}.csv: só perfil socioeconômico/mercado", flush=True)
    return y, A, items


def task_2024(limit=None):
    return task_new(2024, limit)


def task_comp(y, limit=None):
    cols = ["NU_INSCRICAO", "TP_STATUS_REDACAO", "NU_NOTA_COMP1", "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4",
            "NU_NOTA_COMP5", "NU_NOTA_REDACAO", "NU_NOTA_AV1", "NU_NOTA_AV2", "NU_NOTA_AV3", "NU_NOTA_AV4"] + \
           [f"NU_NOTA_COMP{k}_AV{a}" for k in range(1, 6) for a in (1, 2)]
    status, dist, disc = [], [], []
    for df in stream(comp_path(y), cols, limit):
        d = df.drop(columns=["NU_INSCRICAO"]).apply(pd.to_numeric, errors="coerce")
        d["st"] = d.TP_STATUS_REDACAO.fillna(-1).astype(int)
        status.append(d.groupby("st").size().rename("n").reset_index())
        ok = d[d.st == 1]
        for k in range(1, 6):
            dd = ok.groupby(f"NU_NOTA_COMP{k}").size().rename("n").reset_index()
            dd.columns = ["nota", "n"]
            dd["comp"] = k
            dist.append(dd)
        v = d[d.NU_NOTA_AV1.notna() & d.NU_NOTA_AV2.notna() & (d.st.isin([1, 2, 3, 6, 7, 8, 9]))].copy()
        diff = (v.NU_NOTA_AV1 - v.NU_NOTA_AV2).abs()
        cdiff = pd.concat([(v[f"NU_NOTA_COMP{k}_AV1"] - v[f"NU_NOTA_COMP{k}_AV2"]).abs() for k in range(1, 6)], axis=1).max(axis=1)
        s = pd.DataFrame({
            "n_pares": [len(v)], "n_av3": [int(v.NU_NOTA_AV3.notna().sum())], "n_av4": [int(v.NU_NOTA_AV4.notna().sum())],
            "diff_gt100": [int((diff > 100).sum())], "diff_gt200": [int((diff > 200).sum())],
            "cdiff_gt80": [int((cdiff > 80).sum())], "sum_absdiff": [float(diff.sum())],
            "n_final_ne_media": [int(((v.NU_NOTA_REDACAO - (v.NU_NOTA_AV1 + v.NU_NOTA_AV2) / 2).abs() > 0).sum())],
        })
        disc.append(s)
    st = pd.concat(status).groupby("st", as_index=False).n.sum()
    st["ano"] = y
    di = pd.concat(dist).groupby(["comp", "nota"], as_index=False).n.sum()
    di["ano"] = y
    dc = pd.concat(disc).sum().to_frame().T
    dc["ano"] = y
    print(f"[comp {y}] ok", flush=True)
    return y, st, di, dc


# ------------------------------------------------------------- main
def _cat(parts, dims):
    d = pd.concat(parts, ignore_index=True)
    num_cols = [c for c in d.columns if c not in dims]
    return d.groupby(dims, observed=True, dropna=False)[num_cols].sum(min_count=1).reset_index()


CUBES = {"res": ("cubo_resultados", ["ano", "uf", "dep", "renda", "st"]),
         "hist": ("hist_notas", ["ano", "rede", "bin", "nota"]),
         "mun": ("cubo_municipio", ["ano", "uf", "cod_mun", "mun"]),
         "socio": ("cubo_socio", ["ano", "uf", "st", "escola_auto", "renda", "internet", "treineiro"]),
         "demo": ("cubo_demo", ["ano", "uf", "sexo", "faixa", "cor", "st"]),
         "familia": ("cubo_familia", ["ano", "esc_mae", "esc_pai", "renda", "escola_auto", "computador", "internet"]),
         "egressos": ("cubo_egressos", ["ano", "anos_bin", "escola_auto", "renda"])}


def main_incremental(y, limit=None):
    """Processa só o ano y e o mescla nos cubos já existentes (substitui as linhas desse ano)."""
    _, A, it = task_new(y, limit)
    for k, (name, dims) in CUBES.items():
        if not A.get(k):
            continue
        new = _cat(A[k], dims)
        f = OUT / f"{name}.parquet"
        old = pd.read_parquet(f) if f.exists() else new.iloc[:0]
        pd.concat([old[old.ano != y], new], ignore_index=True).sort_values("ano").to_parquet(f)
        print(f"  {name}: +{len(new)} linhas ({y})", flush=True)
    if len(it):
        f = OUT / "itens_stats.parquet"
        old = pd.read_parquet(f)
        pd.concat([old[old.ano != y], it], ignore_index=True).to_parquet(f)
    print("FINALIZADO incremental", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", type=int, default=None, help="processa só este ano (2024+) e mescla nos cubos existentes")
    args = ap.parse_args()
    if args.only:
        return main_incremental(args.only, args.limit)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        fm = [ex.submit(task_micro, y, args.limit) for y in YEARS[:4]]
        f24 = ex.submit(task_2024, args.limit)
        fc = [ex.submit(task_comp, y, args.limit) for y in YEARS]
        res = [f.result() for f in fm] + [f24.result()]
        comp = [f.result() for f in fc]
    P = {k: [] for k in ("res", "hist", "mun", "socio", "demo", "familia", "egressos", "sample")}
    items = []
    for y, A, it in res:
        for k in P:
            P[k].extend(A.get(k, []))
        items.append(it)
    _cat(P["res"], ["ano", "uf", "dep", "renda", "st"]).to_parquet(OUT / "cubo_resultados.parquet")
    _cat(P["hist"], ["ano", "rede", "bin", "nota"]).to_parquet(OUT / "hist_notas.parquet")
    _cat(P["mun"], ["ano", "uf", "cod_mun", "mun"]).to_parquet(OUT / "cubo_municipio.parquet")
    _cat(P["socio"], ["ano", "uf", "st", "escola_auto", "renda", "internet", "treineiro"]).to_parquet(OUT / "cubo_socio.parquet")
    _cat(P["demo"], ["ano", "uf", "sexo", "faixa", "cor", "st"]).to_parquet(OUT / "cubo_demo.parquet")
    _cat(P["familia"], ["ano", "esc_mae", "esc_pai", "renda", "escola_auto", "computador", "internet"]).to_parquet(OUT / "cubo_familia.parquet")
    _cat(P["egressos"], ["ano", "anos_bin", "escola_auto", "renda"]).to_parquet(OUT / "cubo_egressos.parquet")
    pd.concat(P["sample"], ignore_index=True).to_parquet(OUT / "amostra_regressao.parquet")
    pd.concat(items, ignore_index=True).to_parquet(OUT / "itens_stats.parquet")
    pd.concat([c[1] for c in comp]).to_parquet(OUT / "red_status.parquet")
    pd.concat([c[2] for c in comp]).to_parquet(OUT / "red_dist_comp.parquet")
    pd.concat([c[3] for c in comp]).to_parquet(OUT / "red_discordancia.parquet")
    print(f"FINALIZADO em {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
