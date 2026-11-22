"""Leitura e filtragem dos artefatos usados pela visualização Streamlit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def carregar_execucoes(diretorio_experimentos: Path) -> pd.DataFrame:
    """Carrega a tabela de execuções e valida as colunas essenciais."""

    caminho = diretorio_experimentos / "resultados.csv"
    if not caminho.is_file():
        raise FileNotFoundError(f"Resultados experimentais não encontrados: {caminho}")
    tabela = pd.read_csv(caminho)
    colunas = {
        "componentes_pca",
        "semente",
        "quantidade_clusters",
        "expoente_fuzzificacao",
        "diretorio_execucao",
    }
    ausentes = colunas - set(tabela.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes em resultados.csv: {sorted(ausentes)}")
    return tabela


def descobrir_fontes_dados(
    diretorio_configurado: Path,
) -> dict[str, Path]:
    """Localiza os artefatos do piloto e do dataset completo, quando existem."""

    configurado = diretorio_configurado.resolve()
    if configurado.name != "experimentos":
        return (
            {"configurada": configurado}
            if (configurado / "resultados.csv").is_file()
            else {}
        )

    if configurado.parent.name == "completo":
        raiz_artefatos = configurado.parent.parent
    else:
        raiz_artefatos = configurado.parent

    candidatos = {
        "piloto": raiz_artefatos / "experimentos",
        "completo": raiz_artefatos / "completo" / "experimentos",
    }
    fontes = {
        identificador: caminho
        for identificador, caminho in candidatos.items()
        if (caminho / "resultados.csv").is_file()
    }
    if not fontes and (configurado / "resultados.csv").is_file():
        fontes["configurada"] = configurado
    return fontes


def carregar_dados_execucao(
    diretorio_experimentos: Path, diretorio_relativo: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combina projeção, pertinências completas e métricas de uma execução."""

    diretorio_base = diretorio_experimentos.resolve()
    diretorio_execucao = (diretorio_base / diretorio_relativo).resolve()
    if not diretorio_execucao.is_relative_to(diretorio_base):
        raise ValueError("O diretório da execução está fora da pasta de experimentos.")

    projecao = pd.read_csv(diretorio_execucao / "projecao_2d.csv")
    pertinencias = pd.read_csv(diretorio_execucao / "pertinencias.csv")
    colunas_pertinencia = [
        coluna
        for coluna in pertinencias.columns
        if coluna.startswith("pertinencia_cluster_")
    ]
    tabela = projecao.merge(
        pertinencias[["caminho_relativo", *colunas_pertinencia]],
        on="caminho_relativo",
        how="inner",
        validate="one_to_one",
    )
    if len(tabela) != len(projecao):
        raise ValueError("Nem todas as imagens da projeção têm pertinências completas.")
    with (diretorio_execucao / "metricas.json").open(encoding="utf-8") as arquivo:
        metricas: dict[str, Any] = json.load(arquivo)
    return tabela, metricas


def selecionar_galeria(
    tabela: pd.DataFrame,
    modo: str,
    cluster: int | None,
    limite: int,
) -> pd.DataFrame:
    """Seleciona exemplos típicos ou de fronteira para a galeria."""

    if limite <= 0:
        raise ValueError("O limite da galeria deve ser maior que zero.")
    if modo not in {"Típicas", "Ambíguas"}:
        raise ValueError("O modo deve ser 'Típicas' ou 'Ambíguas'.")

    candidatas = tabela.copy()
    if modo == "Típicas":
        coluna = "pertinencia_principal"
        if cluster is not None:
            coluna = f"pertinencia_cluster_{cluster:03}"
            if coluna not in candidatas:
                raise ValueError(f"Pertinências do cluster {cluster} não encontradas.")
        candidatas = candidatas.sort_values(coluna, ascending=False)
    else:
        if cluster is not None:
            candidatas = candidatas[
                (candidatas["cluster_principal"] == cluster)
                | (candidatas["cluster_secundario"] == cluster)
            ]
        candidatas = candidatas.sort_values("diferenca_top2", ascending=True)
    return candidatas.head(limite).reset_index(drop=True)


def selecionar_galeria_instavel(
    tabela: pd.DataFrame,
    grupo_referencia: int | None,
    limite: int,
) -> pd.DataFrame:
    """Seleciona imagens que mudaram de grupo em outras sementes."""

    if limite <= 0:
        raise ValueError("O limite da galeria deve ser maior que zero.")
    colunas_obrigatorias = {
        "grupo_referencia",
        "mudancas_grupo",
        "diferenca_media_pertinencias",
    }
    ausentes = colunas_obrigatorias - set(tabela.columns)
    if ausentes:
        raise ValueError(f"Colunas de estabilidade ausentes: {sorted(ausentes)}")

    candidatas = tabela[tabela["mudancas_grupo"] > 0].copy()
    if grupo_referencia is not None:
        candidatas = candidatas[
            candidatas["grupo_referencia"] == grupo_referencia
        ]
    return (
        candidatas.sort_values(
            ["mudancas_grupo", "diferenca_media_pertinencias"],
            ascending=False,
        )
        .head(limite)
        .reset_index(drop=True)
    )


def selecionar_quantidade_candidata(resumo: pd.DataFrame) -> int | None:
    """Seleciona a menor Xie–Beni entre configurações numericamente válidas."""

    colunas = {
        "quantidade_clusters",
        "convergencias",
        "execucoes",
        "grupos_ocupados_minimo",
        "menor_distancia_centroides_quadrada",
        "xie_beni_mediano",
    }
    ausentes = colunas - set(resumo.columns)
    if ausentes:
        raise ValueError(f"Colunas do resumo ausentes: {sorted(ausentes)}")
    elegiveis = resumo[
        (resumo["convergencias"] == resumo["execucoes"])
        & (
            resumo["grupos_ocupados_minimo"]
            == resumo["quantidade_clusters"]
        )
        & (resumo["menor_distancia_centroides_quadrada"] >= 1e-4)
    ]
    if elegiveis.empty:
        return None
    return int(
        elegiveis.sort_values("xie_beni_mediano").iloc[0]["quantidade_clusters"]
    )


def selecionar_expoente_candidato(resumo: pd.DataFrame) -> float | None:
    """Seleciona o expoente válido mais estável, usando Xie–Beni no desempate."""

    colunas = {
        "expoente_fuzzificacao",
        "quantidade_clusters",
        "convergencias",
        "execucoes",
        "grupos_ocupados_minimo",
        "menor_distancia_centroides_quadrada",
        "concordancia_dominante_media",
        "xie_beni_mediano",
    }
    ausentes = colunas - set(resumo.columns)
    if ausentes:
        raise ValueError(f"Colunas do resumo ausentes: {sorted(ausentes)}")
    elegiveis = resumo[
        (resumo["convergencias"] == resumo["execucoes"])
        & (
            resumo["grupos_ocupados_minimo"]
            == resumo["quantidade_clusters"]
        )
        & (resumo["menor_distancia_centroides_quadrada"] >= 1e-4)
    ]
    if elegiveis.empty:
        return None
    ordenadas = elegiveis.sort_values(
        ["concordancia_dominante_media", "xie_beni_mediano"],
        ascending=[False, True],
    )
    return float(ordenadas.iloc[0]["expoente_fuzzificacao"])


def resolver_caminho_imagem(
    diretorio_imagens: Path, caminho_relativo: str
) -> Path:
    """Resolve um caminho do CSV sem permitir fuga do diretório do dataset."""

    raiz = diretorio_imagens.resolve()
    caminho = (raiz / caminho_relativo).resolve()
    if not caminho.is_relative_to(raiz):
        raise ValueError(f"Caminho de imagem inválido: {caminho_relativo}")
    return caminho


def rotulo_execucao(linha: pd.Series) -> str:
    """Cria um rótulo curto e estável para o seletor de modelo."""

    return (
        f"PCA {int(linha['componentes_pca'])} · "
        f"semente {int(linha['semente'])} · "
        f"m={float(linha['expoente_fuzzificacao']):.1f}"
    )
