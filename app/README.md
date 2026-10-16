# Análise ENEM - 2020 - 2025

Dashboard em Streamlit com análise de mercado, desempenho e oportunidades a partir dos microdados do ENEM (INEP).

## Como rodar

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/streamlit run app/dashboard.py

Os dados processados já ficam em `app/data/` e as malhas do IBGE em `app/geo/`; não é preciso ter os CSVs brutos para rodar o painel.

## Refazer os dados (opcional, exige os CSVs brutos do INEP)

    .venv/bin/python app/etl.py                  # cubos principais 2020-2024 (~5 min)
    .venv/bin/python app/etl.py --only 2025      # incorpora/atualiza só 2025
    .venv/bin/python app/etl_escolas.py          # notas por escola (2024)
    .venv/bin/python app/etl_escolas.py mun      # notas por município da escola (mapa)

## Abas
Geral · Mapa do Brasil · Mercado · Possibilidades · Consolidado · Série Histórica · Pública × Privada · Municípios · Aluno × Escola

## Estrutura
- `app/etl.py`, `app/etl_escolas.py`: leitura em streaming dos CSVs e geração de cubos agregados (pandas, numpy, pyarrow)
- `app/lib.py`: métricas e modelos estatísticos (statsmodels: OLS robusto, Logit, tendência com intervalo)
- `app/dashboard.py`: interface
- `app/data/`: cubos agregados em Parquet · `app/geo/`: malhas municipal e estadual (IBGE)

## Observações sobre os dados
- 2025 traz só o perfil dos inscritos; notas, presença e redação dependem do `RESULTADOS_2025`.
- Em 2024-2025 perfil e resultados estão em bases separadas, sem chave de ligação.
- Valores em R$ da aba Possibilidades são hipóteses editáveis, não dados.
