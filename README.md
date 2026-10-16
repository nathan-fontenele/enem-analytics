# Análise ENEM - 2020 - 2025

Painel interativo em **Streamlit** que transforma os microdados do ENEM (INEP) em análise de mercado, desempenho e oportunidades para quem constrói produtos de educação voltados ao exame.

O projeto lê mais de 10 GB de CSVs brutos, condensa tudo em tabelas agregadas pequenas (Parquet) e serve um painel rápido, com filtros por ano, região e UF.

## O que o painel mostra

| Aba | Conteúdo |
|---|---|
| **Geral** | KPIs com tendência, evolução de inscritos e comparecimento, notas por área, distribuição por rede e perfil dos candidatos |
| **Mapa do Brasil** | Mapa por município (WebGL) de nota média, participação e inscritos, mais um cruzamento nota × participação em 4 quadrantes |
| **Mercado** | Funil TAM → SAM, segmentos de público, índice de oportunidade por UF, maiores municípios e egressos |
| **Possibilidades** | Simulador de receita com cenários e Monte Carlo, dores mensuráveis (redação, habilidades) e modelos estatísticos |
| **Consolidado** | Painel anual, achados principais calculados dos dados e limitações |
| **Série Histórica** | Métricas 2020-2025, índice base 100, tendência linear com intervalo e comparação entre UFs |
| **Pública × Privada** | Relação de notas por área e ano, com opção de detalhar federal, estadual e municipal |
| **Municípios** | Municípios por região, rankings e tabela completa com download em CSV |
| **Aluno × Escola** | Quanto da nota é explicado pela escola (ICC), dispersão dentro das escolas e escolas fora da curva |

## Tecnologias

- **Streamlit** para a interface
- **Pandas** e **NumPy** para manipulação e agregação
- **StatsModels** para OLS robusto (HC1), Logit e tendência com intervalo de previsão
- **Plotly** para os gráficos e o mapa
- **PyArrow** para leitura dos CSVs em streaming e armazenamento em Parquet

## Como rodar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app/dashboard.py
```

Os dados já processados estão em `app/data/` e as malhas do IBGE em `app/geo/`, então **não é preciso ter os CSVs brutos** para abrir o painel.

## Refazer os dados (opcional)

Exige os microdados do INEP baixados localmente (ficam fora do repositório, ~12 GB).

```bash
.venv/bin/python app/etl.py                 # cubos principais 2020-2024 (~5 min)
.venv/bin/python app/etl.py --only 2025     # incorpora ou atualiza só um ano
.venv/bin/python app/etl_escolas.py         # notas por escola (2024)
.venv/bin/python app/etl_escolas.py mun     # notas por município da escola (mapa)
```

Esperado na raiz do projeto, uma pasta por ano: `microdados_enem_AAAA/DADOS/` e `COMPETENCIAS_REDACAO_AAAA/DADOS/`.

## Estrutura

```
app/
  dashboard.py        interface Streamlit
  lib.py              métricas e modelos estatísticos
  etl.py              leitura e harmonização dos microdados, geração dos cubos
  etl_escolas.py      agregações por escola e por município da escola
  data/               cubos agregados (Parquet)
  geo/                malhas municipal e estadual do IBGE
.streamlit/config.toml
requirements.txt
```

## Cuidados na leitura dos dados

- **2025** traz apenas o perfil dos inscritos. Notas, presença e redação dependem do arquivo `RESULTADOS_2025`; enquanto ele não existir, as métricas de desempenho usam o último ano publicado e o painel avisa.
- **2024 e 2025**: o INEP separa participantes (perfil) e resultados (notas) **sem chave de ligação**. Cruzamentos de perfil × nota usam 2020-2023.
- **2020** foi atípico (pandemia, abstenção de ~52%). Tendências são mais confiáveis a partir de 2021.
- A **rede da escola** (pública/privada) vem do Censo Escolar e só existe para parte dos inscritos, em geral concluintes. O código da escola só está disponível em 2024.
- Renda é declarada em faixas de salário mínimo do ano e foi harmonizada entre os anos. Em 2024-2025 a pergunta de internet passou a ser sobre wi-fi, então a variação desse indicador pode refletir o questionário.
- **Valores em R$** (ticket, conversão, penetração, CAC) da aba Possibilidades são **hipóteses editáveis**, não dados observados.
- Os modelos mostram **associações**, não causalidade.

## Fonte dos dados

- Microdados do ENEM, [INEP](https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem/microdados)
- Malhas territoriais, [IBGE](https://servicodados.ibge.gov.br/api/docs/malhas)

## Publicação

O Streamlit precisa de um servidor sempre ligado, por isso **não roda na Vercel**. Uma opção gratuita é o [Streamlit Community Cloud](https://share.streamlit.io): conecte este repositório e use `app/dashboard.py` como arquivo principal.

## Licença

MIT, veja [LICENSE](LICENSE).
