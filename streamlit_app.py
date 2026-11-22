"""Visualização interativa dos agrupamentos de imagens de flores."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from clustering_imagens.estabilidade_imagens import identificar_imagens_instaveis
from clustering_imagens.visualizacao import (
    carregar_dados_execucao,
    carregar_execucoes,
    resolver_caminho_imagem,
    rotulo_execucao,
    selecionar_expoente_candidato,
    selecionar_galeria,
    selecionar_galeria_instavel,
    selecionar_quantidade_candidata,
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


@st.cache_data(max_entries=4)
def carregar_estabilidade_cache(
    diretorio_modelos: str,
    sementes: tuple[int, ...],
    modificacoes: tuple[int, ...],
) -> pd.DataFrame:
    """Calcula a estabilidade por imagem enquanto os modelos não mudarem."""

    del modificacoes
    return identificar_imagens_instaveis(Path(diretorio_modelos), sementes)


@st.cache_data(max_entries=4)
def carregar_tabela_cache(caminho: str, modificacao: int) -> pd.DataFrame:
    """Carrega uma tabela pequena novamente apenas quando o arquivo mudar."""

    del modificacao
    return pd.read_csv(caminho)


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

componentes_selecionados = int(linha_execucao["componentes_pca"])
sementes_modelo = tuple(
    sorted(
        execucoes.loc[
            execucoes["componentes_pca"] == componentes_selecionados,
            "semente",
        ].astype(int)
    )
)
diretorio_modelos = (
    DIRETORIO_EXPERIMENTOS / f"pca_{componentes_selecionados:03}"
)
caminhos_modelos = tuple(
    diretorio_modelos / f"semente_{semente:04}" / "modelo_fcm.npz"
    for semente in sementes_modelo
)
estabilidade_disponivel = len(sementes_modelo) >= 2 and all(
    caminho.is_file() for caminho in caminhos_modelos
)
estabilidade = (
    carregar_estabilidade_cache(
        str(diretorio_modelos),
        sementes_modelo,
        tuple(caminho.stat().st_mtime_ns for caminho in caminhos_modelos),
    )
    if estabilidade_disponivel
    else pd.DataFrame()
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
        ("Típicas", "Ambíguas", "Instáveis")
        if estabilidade_disponivel
        else ("Típicas", "Ambíguas"),
        default="Típicas",
        required=True,
        key="modo_galeria",
    )
    opcoes_cluster: list[str | int] = [
        "Todos",
        *range(1, int(metricas["quantidade_clusters"]) + 1),
    ]
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
if modo_galeria == "Instáveis":
    imagens_galeria = selecionar_galeria_instavel(
        estabilidade,
        grupo_referencia=cluster,
        limite=quantidade_galeria,
    )
else:
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
elif modo_galeria == "Ambíguas":
    st.caption(
        "Ordenadas pela menor diferença entre as duas maiores pertinências, "
        "destacando fronteiras entre grupos."
    )
else:
    quantidade_instaveis = int((estabilidade["mudancas_grupo"] > 0).sum())
    mudaram_em_todas = int(
        (estabilidade["mudancas_grupo"] == len(sementes_modelo) - 1).sum()
    )
    st.caption(
        f"Comparação alinhada às atribuições da semente {sementes_modelo[0]}. "
        "O filtro de grupo também usa essa referência."
    )
    with st.container(horizontal=True):
        st.metric(
            "Imagens instáveis",
            f"{quantidade_instaveis} de {len(estabilidade)}",
            border=True,
        )
        st.metric(
            "Taxa de instabilidade",
            f"{quantidade_instaveis / len(estabilidade):.1%}",
            border=True,
        )
        st.metric(
            "Mudaram em todas as comparações",
            str(mudaram_em_todas),
            border=True,
        )

galeria = st.grid("auto", min_column_width=190, gap="small")
for _, imagem in imagens_galeria.iterrows():
    caminho = resolver_caminho_imagem(
        DIRETORIO_IMAGENS, str(imagem["caminho_relativo"])
    )
    with galeria.cell():
        if caminho.is_file():
            grupo_alt = (
                int(imagem["grupo_referencia"])
                if modo_galeria == "Instáveis"
                else int(imagem["cluster_principal"])
            )
            st.image(
                caminho,
                width="stretch",
                alt=(
                    f"Flor associada ao grupo {grupo_alt} "
                    "na análise visual"
                ),
            )
        else:
            st.warning(f"Imagem não encontrada: {imagem['caminho_relativo']}")
        if modo_galeria == "Instáveis":
            atribuicoes = [
                f"s{sementes_modelo[0]}: G{int(imagem['grupo_referencia'])}"
            ]
            atribuicoes.extend(
                f"s{semente}: G{int(imagem[f'grupo_semente_{semente:04}'])}"
                for semente in sementes_modelo[1:]
            )
            st.caption(
                f"`{imagem['caminho_relativo']}`  \n"
                f"{' · '.join(atribuicoes)}  \n"
                f"Mudanças: {int(imagem['mudancas_grupo'])} · "
                "diferença média das pertinências="
                f"{float(imagem['diferenca_media_pertinencias']):.3f}"
            )
        else:
            st.caption(
                f"`{imagem['caminho_relativo']}`  \n"
                f"Grupo {int(imagem['cluster_principal'])} · "
                f"μ={float(imagem['pertinencia_principal']):.3f} · "
                f"Δ={float(imagem['diferenca_top2']):.3f}"
            )

if modo_galeria == "Instáveis":
    st.download_button(
        "Baixar estabilidade por imagem",
        data=estabilidade.to_csv(index=False).encode("utf-8"),
        file_name=f"estabilidade_imagens_pca{componentes_selecionados}.csv",
        mime="text/csv",
        icon=":material/download:",
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

caminho_avaliacao_clusters = (
    DIRETORIO_EXPERIMENTOS.parent
    / "avaliacao_clusters"
    / "resumo_clusters.csv"
)
if caminho_avaliacao_clusters.is_file():
    resumo_clusters = carregar_tabela_cache(
        str(caminho_avaliacao_clusters),
        caminho_avaliacao_clusters.stat().st_mtime_ns,
    )
    melhor_quantidade = selecionar_quantidade_candidata(resumo_clusters)

    st.header("Escolha da quantidade de grupos")
    st.caption(
        "Comparação com PCA-64, m=1,2 e três sementes. Valores altos de "
        "estabilidade não compensam falta de convergência ou centroides "
        "praticamente coincidentes."
    )
    if melhor_quantidade is not None:
        st.success(
            f"{melhor_quantidade} grupos apresentou o melhor equilíbrio entre "
            "convergência, separação e Xie–Beni neste piloto.",
            title="Configuração candidata",
        )

    dados_clusters = resumo_clusters.assign(
        grupos=resumo_clusters["quantidade_clusters"].astype(str),
        convergencia=resumo_clusters.apply(
            lambda linha: (
                f"{int(linha['convergencias'])} de {int(linha['execucoes'])}"
            ),
            axis=1,
        ),
    )
    base_clusters: Any = alt.Chart(dados_clusters)
    grafico_xie_beni = (
        base_clusters.mark_line(point=True, size=3)
        .encode(
            x=alt.X("grupos:O", title="Quantidade de grupos", sort=None),
            y=alt.Y(
                "xie_beni_mediano:Q",
                title="Xie–Beni mediano (escala log)",
                scale=alt.Scale(type="log"),
            ),
            tooltip=[
                alt.Tooltip("grupos:O", title="Grupos"),
                alt.Tooltip(
                    "xie_beni_mediano:Q", title="Xie–Beni mediano", format=".4g"
                ),
                alt.Tooltip("convergencia:N", title="Convergências"),
            ],
        )
        .properties(height=300)
    )
    grafico_convergencia = (
        base_clusters.mark_bar()
        .encode(
            x=alt.X("grupos:O", title="Quantidade de grupos", sort=None),
            y=alt.Y(
                "convergencias:Q",
                title="Execuções convergentes",
                scale=alt.Scale(domain=[0, int(resumo_clusters["execucoes"].max())]),
            ),
            color=alt.Color(
                "convergencia:N", title="Convergências", legend=None
            ),
            tooltip=[
                alt.Tooltip("grupos:O", title="Grupos"),
                alt.Tooltip("convergencia:N", title="Convergências"),
                alt.Tooltip(
                    "silhueta_fcm_media:Q", title="Silhueta FCM", format=".3f"
                ),
            ],
        )
        .properties(height=300)
    )

    coluna_xie, coluna_convergencia = st.columns(2)
    with coluna_xie.container(border=True):
        st.subheader("Separação fuzzy")
        st.altair_chart(
            grafico_xie_beni,
            key="comparacao_xie_beni_clusters",
            alt="Oito grupos tem o menor Xie–Beni; doze e quinze apresentam valores extremos",
        )
    with coluna_convergencia.container(border=True):
        st.subheader("Convergência entre sementes")
        st.altair_chart(
            grafico_convergencia,
            key="comparacao_convergencia_clusters",
            alt="Oito grupos convergiu nas três sementes e foi a configuração mais consistente",
        )

    tabela_clusters = resumo_clusters[
        [
            "quantidade_clusters",
            "convergencias",
            "execucoes",
            "xie_beni_mediano",
            "silhueta_fcm_media",
            "concordancia_dominante_media",
            "indice_rand_ajustado_medio",
            "menor_distancia_centroides_quadrada",
        ]
    ].rename(
        columns={
            "quantidade_clusters": "Grupos",
            "convergencias": "Convergências",
            "execucoes": "Execuções",
            "xie_beni_mediano": "Xie–Beni mediano",
            "silhueta_fcm_media": "Silhueta FCM",
            "concordancia_dominante_media": "Concordância",
            "indice_rand_ajustado_medio": "Rand ajustado",
            "menor_distancia_centroides_quadrada": "Menor distância²",
        }
    )
    st.dataframe(
        tabela_clusters,
        hide_index=True,
        alt="Resumo da qualidade para oito, dez, doze e quinze grupos",
        column_config={
            "Xie–Beni mediano": st.column_config.NumberColumn(format="%.4g"),
            "Silhueta FCM": st.column_config.NumberColumn(format="%.3f"),
            "Concordância": st.column_config.NumberColumn(format="percent"),
            "Rand ajustado": st.column_config.NumberColumn(format="%.3f"),
            "Menor distância²": st.column_config.NumberColumn(format="%.4g"),
        },
    )

caminho_avaliacao_fuzzificacao = (
    DIRETORIO_EXPERIMENTOS.parent
    / "avaliacao_fuzzificacao"
    / "resumo_fuzzificacao.csv"
)
if caminho_avaliacao_fuzzificacao.is_file():
    resumo_fuzzificacao = carregar_tabela_cache(
        str(caminho_avaliacao_fuzzificacao),
        caminho_avaliacao_fuzzificacao.stat().st_mtime_ns,
    )
    melhor_expoente = selecionar_expoente_candidato(resumo_fuzzificacao)

    st.header("Escolha do expoente de fuzzificação")
    st.caption(
        "Comparação com PCA-64, oito grupos e três sementes. Pertinências mais "
        "altas em m baixo significam uma solução mais rígida; não representam "
        "automaticamente um agrupamento fuzzy melhor."
    )
    if melhor_expoente is not None:
        st.success(
            f"m={melhor_expoente:.2f} preservou os oito grupos, convergiu nas "
            "três sementes e apresentou a maior concordância entre as soluções "
            "numericamente válidas.",
            title="Expoente candidato",
        )

    dados_fuzzificacao = resumo_fuzzificacao.assign(
        expoente=resumo_fuzzificacao["expoente_fuzzificacao"].map(
            lambda valor: f"{float(valor):.2f}"
        ),
        convergencia=resumo_fuzzificacao.apply(
            lambda linha: (
                f"{int(linha['convergencias'])} de {int(linha['execucoes'])}"
            ),
            axis=1,
        ),
        status=resumo_fuzzificacao.apply(
            lambda linha: (
                "Válida"
                if (
                    linha["convergencias"] == linha["execucoes"]
                    and linha["grupos_ocupados_minimo"]
                    == linha["quantidade_clusters"]
                    and linha["menor_distancia_centroides_quadrada"] >= 1e-4
                )
                else "Degenerada ou sem convergência"
            ),
            axis=1,
        ),
    )
    dados_xie_finitos = dados_fuzzificacao[
        dados_fuzzificacao["xie_beni_mediano"] < float("inf")
    ]
    base_xie_fuzzificacao: Any = alt.Chart(dados_xie_finitos)
    grafico_xie_fuzzificacao = (
        base_xie_fuzzificacao.mark_line(point=True, size=3)
        .encode(
            x=alt.X("expoente:O", title="Expoente m", sort=None),
            y=alt.Y(
                "xie_beni_mediano:Q",
                title="Xie–Beni mediano (escala log)",
                scale=alt.Scale(type="log"),
            ),
            tooltip=[
                alt.Tooltip("expoente:O", title="m"),
                alt.Tooltip(
                    "xie_beni_mediano:Q", title="Xie–Beni mediano", format=".4g"
                ),
                alt.Tooltip("convergencia:N", title="Convergências"),
            ],
        )
        .properties(height=300)
    )
    base_estabilidade_fuzzificacao: Any = alt.Chart(dados_fuzzificacao)
    grafico_estabilidade_fuzzificacao = (
        base_estabilidade_fuzzificacao.mark_bar()
        .encode(
            x=alt.X("expoente:O", title="Expoente m", sort=None),
            y=alt.Y(
                "concordancia_dominante_media:Q",
                title="Concordância entre sementes",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("status:N", title="Diagnóstico"),
            tooltip=[
                alt.Tooltip("expoente:O", title="m"),
                alt.Tooltip(
                    "concordancia_dominante_media:Q",
                    title="Concordância",
                    format=".1%",
                ),
                alt.Tooltip("convergencia:N", title="Convergências"),
                alt.Tooltip("status:N", title="Diagnóstico"),
            ],
        )
        .properties(height=300)
    )

    coluna_xie_m, coluna_estabilidade_m = st.columns(2)
    with coluna_xie_m.container(border=True):
        st.subheader("Separação por expoente")
        st.altair_chart(
            grafico_xie_fuzzificacao,
            key="comparacao_xie_beni_fuzzificacao",
            alt="Xie–Beni cresce rapidamente quando m chega a 1,25 e os centroides colapsam",
        )
    with coluna_estabilidade_m.container(border=True):
        st.subheader("Estabilidade e diagnóstico")
        st.altair_chart(
            grafico_estabilidade_fuzzificacao,
            key="comparacao_estabilidade_fuzzificacao",
            alt="m igual a 1,20 é o mais estável entre os expoentes que convergiram sem colapso",
        )

    st.warning(
        "A concordância muito alta de m=1,25 e m=1,30 é artificial: nenhuma "
        "dessas execuções convergiu e os centroides ficaram praticamente "
        "coincidentes.",
        title="Estabilidade degenerada",
    )
    tabela_fuzzificacao = resumo_fuzzificacao[
        [
            "expoente_fuzzificacao",
            "convergencias",
            "execucoes",
            "xie_beni_mediano",
            "silhueta_fcm_media",
            "pertinencia_principal_media",
            "concordancia_dominante_media",
            "indice_rand_ajustado_medio",
            "menor_distancia_centroides_quadrada",
        ]
    ].rename(
        columns={
            "expoente_fuzzificacao": "Expoente m",
            "convergencias": "Convergências",
            "execucoes": "Execuções",
            "xie_beni_mediano": "Xie–Beni mediano",
            "silhueta_fcm_media": "Silhueta FCM",
            "pertinencia_principal_media": "Pertinência principal",
            "concordancia_dominante_media": "Concordância",
            "indice_rand_ajustado_medio": "Rand ajustado",
            "menor_distancia_centroides_quadrada": "Menor distância²",
        }
    )
    st.dataframe(
        tabela_fuzzificacao,
        hide_index=True,
        alt="Resumo dos expoentes de fuzzificação entre 1,10 e 1,30",
        column_config={
            "Expoente m": st.column_config.NumberColumn(format="%.2f"),
            "Xie–Beni mediano": st.column_config.NumberColumn(format="%.4g"),
            "Silhueta FCM": st.column_config.NumberColumn(format="%.3f"),
            "Pertinência principal": st.column_config.NumberColumn(format="%.3f"),
            "Concordância": st.column_config.NumberColumn(format="percent"),
            "Rand ajustado": st.column_config.NumberColumn(format="%.3f"),
            "Menor distância²": st.column_config.NumberColumn(format="%.4g"),
        },
    )

st.download_button(
    "Baixar pertinências desta execução",
    data=caminho_pertinencias.read_bytes(),
    file_name=f"pertinencias_pca{int(linha_execucao['componentes_pca'])}_semente{int(linha_execucao['semente'])}.csv",
    mime="text/csv",
    icon=":material/download:",
)
