"""Agrega as notas de 2024 por ESCOLA (CO_ESCOLA) -> app/data/escolas_2024.parquet

Só 2024 tem o código da escola (RESULTADOS_2024). O INEP mascara o código de escolas com <10 participantes,
por isso o dashboard usa apenas escolas com amostra mínima de presentes.
"""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from etl import stream, num, find_csv, DEP, OUT

COLS = ["CO_ESCOLA", "CO_MUNICIPIO_ESC", "NO_MUNICIPIO_ESC", "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC",
        "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
        "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
KEYS = ["escola", "uf", "cod_mun", "mun", "dep", "loc"]


def main(limit=None):
    t0 = time.time()
    path = find_csv(2024, "RESULTADOS")
    parts = []
    for i, df in enumerate(stream(path, COLS, limit)):
        df = df[df.CO_ESCOLA.notna()]
        n = {k: num(df["NU_NOTA_" + k]) for k in ("CN", "CH", "LC", "MT", "REDACAO")}
        pres = (num(df.TP_PRESENCA_CN) == 1) & (num(df.TP_PRESENCA_CH) == 1) & (num(df.TP_PRESENCA_LC) == 1) & (num(df.TP_PRESENCA_MT) == 1)
        full = pd.concat(n, axis=1)
        ok = pres & full.notna().all(axis=1)
        t = pd.DataFrame({"escola": num(df.CO_ESCOLA).astype("int64"), "uf": df.SG_UF_ESC.fillna("n/d"), "cod_mun": num(df.CO_MUNICIPIO_ESC),
                          "mun": df.NO_MUNICIPIO_ESC.fillna("n/d"), "dep": num(df.TP_DEPENDENCIA_ADM_ESC).map(DEP).fillna("n/d"),
                          "loc": num(df.TP_LOCALIZACAO_ESC).map({1: "Urbana", 2: "Rural"}).fillna("n/d")})[ok]
        f = full[ok]
        g = f.mean(axis=1)
        t["n"] = 1
        t["s_GERAL"], t["ss_GERAL"] = g, g ** 2
        for k in ("CN", "CH", "LC", "MT", "REDACAO"):
            t["s_" + k] = f[k]
        t["red700"] = (f.REDACAO >= 700).astype(int)
        parts.append(t.groupby(KEYS, observed=True, dropna=False).sum().reset_index())
        print(f"[escolas] bloco {i} ok ({time.time()-t0:.0f}s)", flush=True)
    d = pd.concat(parts).groupby(KEYS, observed=True, dropna=False).sum().reset_index()
    d.to_parquet(OUT / "escolas_2024.parquet")
    print("FINALIZADO", len(d), "escolas", flush=True)




# ----------------------------------------------------------------------------
# Município da ESCOLA (cobre quase todos os 5.570 municípios) para o mapa do Brasil
# ----------------------------------------------------------------------------
def mun_escola(y, limit=None):
    from etl import micro_path
    path = micro_path(y) if y <= 2023 else find_csv(y, "RESULTADOS")
    cols = ["CO_MUNICIPIO_ESC", "NO_MUNICIPIO_ESC", "SG_UF_ESC", "TP_DEPENDENCIA_ADM_ESC",
            "TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT",
            "NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
    parts = []
    for df in stream(path, cols, limit):
        df = df[df.CO_MUNICIPIO_ESC.notna()]
        pres = (num(df.TP_PRESENCA_CN) == 1) & (num(df.TP_PRESENCA_CH) == 1) & (num(df.TP_PRESENCA_LC) == 1) & (num(df.TP_PRESENCA_MT) == 1)
        f = pd.concat({k: num(df["NU_NOTA_" + k]) for k in ("CN", "CH", "LC", "MT", "REDACAO")}, axis=1)
        ok = pres & f.notna().all(axis=1)
        g = f.mean(axis=1)
        t = pd.DataFrame({"ano": y, "uf": df.SG_UF_ESC.fillna("n/d"), "cod_mun": num(df.CO_MUNICIPIO_ESC).astype("int64"),
                          "mun": df.NO_MUNICIPIO_ESC.fillna("n/d"),
                          "rede": np.where(num(df.TP_DEPENDENCIA_ADM_ESC) == 4, "Privada", "Pública")})
        t["n_insc"] = 1
        t["n_pres"] = pres.astype(int)
        t["cnt"] = ok.astype(int)
        t["s_GERAL"] = g.where(ok, 0)
        t["s_RED"] = f.REDACAO.where(ok, 0)
        t["red700"] = ((f.REDACAO >= 700) & ok).astype(int)
        parts.append(t.groupby(["ano", "uf", "cod_mun", "mun", "rede"], observed=True).sum().reset_index())
    print(f"[mun_escola {y}] ok", flush=True)
    return pd.concat(parts).groupby(["ano", "uf", "cod_mun", "mun", "rede"], observed=True).sum().reset_index()


def run_mun_escola(limit=None):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        fs = [ex.submit(mun_escola, y, limit) for y in (2020, 2021, 2022, 2023, 2024)]
        d = pd.concat([f.result() for f in fs], ignore_index=True)
    d.to_parquet(OUT / "cubo_mun_escola.parquet")
    print("FINALIZADO mun_escola", len(d), "linhas;", d[d.ano == 2024].cod_mun.nunique(), "municípios em 2024", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mun":
        run_mun_escola(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
