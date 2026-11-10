"""Visualização interativa dos agrupamentos de imagens de flores."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from clustering_imagens.visualizacao import (
    carregar_dados_execucao,
    carregar_execucoes,
    resolver_caminho_imagem,
    rotulo_execucao,
    selecionar_galeria,
)

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_EXPERIMENTOS = Path(
    os.environ.get(
        "CLUSTERING_ARTEFATOS_DIR",
        str(RAIZ_PROJETO / "artefatos" / "experimentos"),
    )
)
DIRETORIO_IMAGENS = Path(
    os.environ.get(
        "CLUSTERING_IMAGENS_DIR",
        str(RAIZ_PROJETO / "102flowers" / "jpg"),
    )
)


@st.cache_data(max_entries=4)
def carregar_execucoes_cache(
    diretorio: str, modificacao: int
) -> pd.DataFrame:
    """Mantém a tabela de execuções em cache até o arquivo mudar."""

    del modificacao
    return carregar_execucoes(Path(diretorio))


@st.cache_data(max_entries=12)
def carregar_modelo_cache(
    diretorio: str, diretorio_relativo: str, modificacao: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mantém os dados de cada modelo em cache até seus artefatos mudarem."""

    del modificacao
    return carregar_dados_execucao(Path(diretorio), diretorio_relativo)


st.set_page_config(
    page_title="Agrupamento visual de flores",
    page_icon="🌸",
    layout="wide",
)

st.title("Agrupamento visual de flores")
st.caption(
    "Explore as afinidades fuzzy, compare execuções e inspecione imagens "
    "representativas ou de fronteira."
)

caminho_resultados = DIRETORIO_EXPERIMENTOS / "resultados.csv"
if not caminho_resultados.is_file():
    st.error(
        "Execute a grade experimental antes de abrir a visualização.",
        title="Artefatos não encontrados",
    )
    st.code(
        ".venv\\Scripts\\executar-experimentos "
        "--arquivo-caracteristicas artefatos\\piloto\\caracteristicas.npz"
    )
    st.stop()

execucoes = carregar_execucoes_cache(
    str(DIRETORIO_EXPERIMENTOS), caminho_resultados.stat().st_mtime_ns
)
execucoes = execucoes.copy()
execucoes["rotulo"] = execucoes.apply(rotulo_execucao, axis=1)

with st.sidebar:
    st.header("Modelo")
    rotulo_selecionado = st.selectbox(
        "Execução",
        execucoes["rotulo"].tolist(),
        key="execucao_selecionada",
        bind="query-params",
    )
    st.caption(
        "A representação PCA é fixa entre sementes; apenas a inicialização do "
        "agrupamento muda."
    )

linha_execucao = execucoes.loc[execucoes["rotulo"] == rotulo_selecionado].iloc[0]
diretorio_relativo = str(linha_execucao["diretorio_execucao"])
caminho_pertinencias = (
    DIRETORIO_EXPERIMENTOS / diretorio_relativo / "pertinencias.csv"
)
tabela, metricas = carregar_modelo_cache(
    str(DIRETORIO_EXPERIMENTOS),
    diretorio_relativo,
    caminho_pertinencias.stat().st_mtime_ns,
)

caminho_resumo = DIRETORIO_EXPERIMENTOS / "resumo_representacoes.csv"
resumo = pd.read_csv(caminho_resumo)
resumo_modelo = resumo.loc[
    resumo["componentes_pca"] == int(linha_execucao["componentes_pca"])
].iloc[0]

with st.grid(6, min_column_width=145, gap="small"):
    st.metric(
        "Imagens",
        f"{int(metricas['quantidade_imagens']):,}".replace(",", "."),
        border=True,
    )
    st.metric(
        "Variância PCA",
        f"{float(metricas['variancia_explicada_pca']):.1%}",
        border=True,
    )
    st.metric(
        "Pertinência principal",
        f"{float(metricas['pertinencia_principal_media']):.3f}",
        border=True,
    )
    st.metric(
        "Silhueta FCM",
        f"{float(metricas['silhueta_atribuicao_dominante']):.3f}",
        border=True,
    )
    st.metric(
        "Concordância entre sementes",
        f"{float(resumo_modelo['concordancia_dominante_media']):.1%}",
        border=True,
    )
    st.metric(
        "Convergência",
        "Sim" if bool(metricas["convergiu"]) else "Não",
        border=True,
    )

if not bool(metricas["convergiu"]):
    st.warning(
        "Esta execução atingiu o limite de iterações. Use-a para diagnóstico, "
        "não como modelo final.",
        title="Execução sem convergência",
    )
elif float(metricas["menor_distancia_centroides_quadrada"]) < 1e-4:
    st.warning(
        "Há centroides praticamente coincidentes; métricas de estabilidade "
        "podem parecer altas mesmo sem grupos bem separados.",
        title="Possível degeneração dos grupos",
    )

dados_grafico = tabela.assign(
    grupo=tabela["cluster_principal"].map(lambda valor: f"Grupo {int(valor)}")
)
contagens_grupos = dados_grafico["grupo"].value_counts()
distribuicao = pd.DataFrame(
    {
        "grupo": contagens_grupos.index.astype(str),
        "imagens": contagens_grupos.to_numpy(),
    }
)

base_dispersao: Any = alt.Chart(dados_grafico)
grafico_dispersao = (
    base_dispersao
    .mark_circle(size=75)
    .encode(
        x=alt.X("componente_1:Q", title="Componente visual 1"),
        y=alt.Y("componente_2:Q", title="Componente visual 2"),
        color=alt.Color("grupo:N", title="Grupo"),
        opacity=alt.Opacity(
            "pertinencia_principal:Q",
            title="Pertinência",
            scale=alt.Scale(domain=[0, 1], range=[0.3, 1]),
        ),
        tooltip=[
            alt.Tooltip("caminho_relativo:N", title="Imagem"),
            alt.Tooltip("grupo:N", title="Grupo"),
            alt.Tooltip(
                "pertinencia_principal:Q", title="Pertinência", format=".3f"
            ),
            alt.Tooltip("diferenca_top2:Q", title="Diferença top 2", format=".3f"),
        ],
    )
    .properties(height=420)
    .interactive()
)
base_distribuicao: Any = alt.Chart(distribuicao)
grafico_distribuicao = (
    base_distribuicao
    .mark_bar()
    .encode(
        x=alt.X("grupo:N", title="Grupo", sort="-y"),
        y=alt.Y("imagens:Q", title="Quantidade de imagens"),
        color=alt.Color("grupo:N", title="Grupo", legend=None),
        tooltip=["grupo:N", alt.Tooltip("imagens:Q", title="Imagens")],
    )
    .properties(height=420)
)

coluna_dispersao, coluna_distribuicao = st.columns((2, 1))
with coluna_dispersao.container(border=True):
    st.subheader("Mapa visual da amostra")
    st.altair_chart(
        grafico_dispersao,
        key="mapa_visual",
        alt="Projeção bidimensional das flores colorida pelo grupo FCM dominante",
    )
with coluna_distribuicao.container(border=True):
    st.subheader("Tamanho dos grupos")
    st.altair_chart(
        grafico_distribuicao,
        key="tamanho_grupos",
        alt="Quantidade de imagens em cada grupo FCM dominante",
    )

st.header("Galeria de imagens")
with st.container(horizontal=True, vertical_alignment="bottom"):
    modo_galeria = st.segmented_control(
        "Seleção",
        ("Típicas", "Ambíguas"),
        default="Típicas",
        required=True,
        key="modo_galeria",
    )
    opcoes_cluster: list[str | int] = ["Todos", *range(1, int(metricas["quantidade_clusters"]) + 1)]
    cluster_selecionado = st.selectbox(
        "Grupo",
        opcoes_cluster,
        key="cluster_galeria",
    )
    quantidade_galeria = st.slider(
        "Quantidade",
        min_value=4,
        max_value=24,
        value=12,
        step=4,
        key="quantidade_galeria",
    )

cluster = None if cluster_selecionado == "Todos" else int(cluster_selecionado)
imagens_galeria = selecionar_galeria(
    tabela,
    modo=str(modo_galeria),
    cluster=cluster,
    limite=quantidade_galeria,
)
if modo_galeria == "Típicas":
    st.caption(
        "Ordenadas pela maior pertinência ao grupo selecionado; sem filtro, "
        "mostra as imagens mais típicas de seus grupos dominantes."
    )
else:
    st.caption(
        "Ordenadas pela menor diferença entre as duas maiores pertinências, "
        "destacando fronteiras entre grupos."
    )

galeria = st.grid("auto", min_column_width=190, gap="small")
for _, imagem in imagens_galeria.iterrows():
    caminho = resolver_caminho_imagem(
        DIRETORIO_IMAGENS, str(imagem["caminho_relativo"])
    )
    with galeria.cell():
        if caminho.is_file():
            st.image(
                caminho,
                width="stretch",
                alt=(
                    f"Flor associada ao grupo {int(imagem['cluster_principal'])} "
                    "na análise visual"
                ),
            )
        else:
            st.warning(f"Imagem não encontrada: {imagem['caminho_relativo']}")
        st.caption(
            f"`{imagem['caminho_relativo']}`  \n"
            f"Grupo {int(imagem['cluster_principal'])} · "
            f"μ={float(imagem['pertinencia_principal']):.3f} · "
            f"Δ={float(imagem['diferenca_top2']):.3f}"
        )

st.header("Comparação experimental")
tabela_comparacao = execucoes[
    [
        "componentes_pca",
        "semente",
        "convergiu",
        "indice_xie_beni",
        "silhueta_atribuicao_dominante",
        "pertinencia_principal_media",
        "clusters_dominantes_ocupados",
        "menor_distancia_centroides_quadrada",
        "kmeans_silhueta",
    ]
].rename(
    columns={
        "componentes_pca": "PCA",
        "semente": "Semente",
        "convergiu": "Convergiu",
        "indice_xie_beni": "Xie–Beni",
        "silhueta_atribuicao_dominante": "Silhueta FCM",
        "pertinencia_principal_media": "Pertinência principal",
        "clusters_dominantes_ocupados": "Grupos ocupados",
        "menor_distancia_centroides_quadrada": "Menor distância²",
        "kmeans_silhueta": "Silhueta K-Means",
    }
)
st.dataframe(
    tabela_comparacao,
    hide_index=True,
    alt="Métricas das execuções por representação PCA e semente",
    column_config={
        "Xie–Beni": st.column_config.NumberColumn(format="%.3f"),
        "Silhueta FCM": st.column_config.NumberColumn(format="%.3f"),
        "Pertinência principal": st.column_config.NumberColumn(format="%.3f"),
        "Menor distância²": st.column_config.NumberColumn(format="%.4g"),
        "Silhueta K-Means": st.column_config.NumberColumn(format="%.3f"),
    },
)

st.download_button(
    "Baixar pertinências desta execução",
    data=caminho_pertinencias.read_bytes(),
    file_name=f"pertinencias_pca{int(linha_execucao['componentes_pca'])}_semente{int(linha_execucao['semente'])}.csv",
    mime="text/csv",
    icon=":material/download:",
)
