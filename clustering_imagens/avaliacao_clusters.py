"""Comparação do Fuzzy C-Means para diferentes quantidades de grupos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from clustering_imagens.experimentos import (
    carregar_modelo,
    comparar_com_referencia,
)
from clustering_imagens.piloto import executar_piloto


def resumir_avaliacao(
    resultados: pd.DataFrame,
    estabilidade: pd.DataFrame,
    quantidades_clusters: Sequence[int],
) -> pd.DataFrame:
    """Resume qualidade, convergência e estabilidade de cada quantidade."""

    linhas: list[dict[str, object]] = []
    for quantidade in quantidades_clusters:
        execucoes = resultados[
            resultados["quantidade_clusters"] == quantidade
        ]
        comparacoes = estabilidade[
            (estabilidade["quantidade_clusters"] == quantidade)
            & (
                estabilidade["semente_comparada"]
                != estabilidade["semente_referencia"]
            )
        ]
        if execucoes.empty:
            raise ValueError(f"Não há execuções para {quantidade} grupos.")
        linhas.append(
            {
                "quantidade_clusters": quantidade,
                "execucoes": len(execucoes),
                "convergencias": int(execucoes["convergiu"].sum()),
                "grupos_ocupados_minimo": int(
                    execucoes["clusters_dominantes_ocupados"].min()
                ),
                "xie_beni_medio": execucoes["indice_xie_beni"].mean(),
                "xie_beni_mediano": execucoes["indice_xie_beni"].median(),
                "silhueta_fcm_media": execucoes[
                    "silhueta_atribuicao_dominante"
                ].mean(),
                "silhueta_kmeans_media": execucoes["kmeans_silhueta"].mean(),
                "pertinencia_principal_media": execucoes[
                    "pertinencia_principal_media"
                ].mean(),
                "menor_distancia_centroides_quadrada": execucoes[
                    "menor_distancia_centroides_quadrada"
                ].min(),
                "concordancia_dominante_media": (
                    comparacoes["concordancia_dominante"].mean()
                    if not comparacoes.empty
                    else 1.0
                ),
                "indice_rand_ajustado_medio": (
                    comparacoes["indice_rand_ajustado"].mean()
                    if not comparacoes.empty
                    else 1.0
                ),
                "diferenca_media_pertinencias": (
                    comparacoes["diferenca_media_pertinencias"].mean()
                    if not comparacoes.empty
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(linhas)


def executar_avaliacao_clusters(
    arquivo_caracteristicas: Path,
    diretorio_saida: Path,
    quantidades_clusters: Sequence[int] = (8, 10, 12, 15),
    sementes: Sequence[int] = (42, 123, 2026),
    componentes_pca: int = 64,
    expoente_fuzzificacao: float = 1.2,
    semente_pca: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa e salva a comparação entre quantidades de clusters."""

    if not quantidades_clusters:
        raise ValueError("Informe ao menos uma quantidade de clusters.")
    if any(quantidade < 2 for quantidade in quantidades_clusters):
        raise ValueError("Cada quantidade de clusters deve ser pelo menos 2.")
    if len(set(quantidades_clusters)) != len(quantidades_clusters):
        raise ValueError("As quantidades de clusters não podem se repetir.")
    if len(sementes) < 2:
        raise ValueError("Informe ao menos duas sementes para medir estabilidade.")
    if len(set(sementes)) != len(sementes):
        raise ValueError("As sementes não podem se repetir.")

    diretorio_saida.mkdir(parents=True, exist_ok=True)
    linhas_resultados: list[dict[str, object]] = []
    linhas_estabilidade: list[dict[str, object]] = []

    for quantidade in quantidades_clusters:
        modelos: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for semente in sementes:
            diretorio_execucao = (
                diretorio_saida
                / f"k_{quantidade:03}"
                / f"semente_{semente:04}"
            )
            metricas = executar_piloto(
                arquivo_caracteristicas=arquivo_caracteristicas,
                diretorio_saida=diretorio_execucao,
                componentes_pca=componentes_pca,
                quantidade_clusters=quantidade,
                expoente_fuzzificacao=expoente_fuzzificacao,
                semente=semente,
                semente_pca=semente_pca,
                arquivo_cache_representacao=(
                    diretorio_saida / f"representacao_pca_{componentes_pca:03}.npz"
                ),
            )
            linha = asdict(metricas)
            linha["diretorio_execucao"] = diretorio_execucao.relative_to(
                diretorio_saida
            ).as_posix()
            linhas_resultados.append(linha)
            modelos[semente] = carregar_modelo(diretorio_execucao / "modelo_fcm.npz")

        semente_referencia = sementes[0]
        centroides_referencia, pertinencias_referencia = modelos[semente_referencia]
        for semente in sementes:
            centroides, pertinencias = modelos[semente]
            comparacao = comparar_com_referencia(
                centroides_referencia,
                pertinencias_referencia,
                centroides,
                pertinencias,
                componentes_pca,
                semente_referencia,
                semente,
            )
            linha_estabilidade = asdict(comparacao)
            linha_estabilidade["quantidade_clusters"] = quantidade
            linhas_estabilidade.append(linha_estabilidade)

    resultados = pd.DataFrame(linhas_resultados)
    estabilidade = pd.DataFrame(linhas_estabilidade)
    resumo = resumir_avaliacao(resultados, estabilidade, quantidades_clusters)

    resultados.to_csv(diretorio_saida / "resultados_clusters.csv", index=False)
    estabilidade.to_csv(diretorio_saida / "estabilidade_clusters.csv", index=False)
    resumo.to_csv(diretorio_saida / "resumo_clusters.csv", index=False)
    with (diretorio_saida / "configuracao.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            {
                "arquivo_caracteristicas": str(arquivo_caracteristicas.resolve()),
                "quantidades_clusters": list(quantidades_clusters),
                "sementes": list(sementes),
                "componentes_pca": componentes_pca,
                "expoente_fuzzificacao": expoente_fuzzificacao,
                "semente_pca": semente_pca,
            },
            arquivo,
            ensure_ascii=False,
            indent=2,
        )
        arquivo.write("\n")
    return resultados, estabilidade, resumo


def criar_analisador_argumentos() -> argparse.ArgumentParser:
    """Configura a linha de comando da avaliação."""

    analisador = argparse.ArgumentParser(
        description="Compara diferentes quantidades de grupos no PCA-64."
    )
    analisador.add_argument("--arquivo-caracteristicas", type=Path, required=True)
    analisador.add_argument(
        "--diretorio-saida",
        type=Path,
        default=Path("artefatos/avaliacao_clusters"),
    )
    analisador.add_argument(
        "--quantidades-clusters", type=int, nargs="+", default=(8, 10, 12, 15)
    )
    analisador.add_argument("--sementes", type=int, nargs="+", default=(42, 123, 2026))
    analisador.add_argument("--componentes-pca", type=int, default=64)
    analisador.add_argument("--expoente-fuzzificacao", type=float, default=1.2)
    analisador.add_argument("--semente-pca", type=int, default=42)
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa a avaliação pela linha de comando."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    _, _, resumo = executar_avaliacao_clusters(
        arquivo_caracteristicas=opcoes.arquivo_caracteristicas,
        diretorio_saida=opcoes.diretorio_saida,
        quantidades_clusters=opcoes.quantidades_clusters,
        sementes=opcoes.sementes,
        componentes_pca=opcoes.componentes_pca,
        expoente_fuzzificacao=opcoes.expoente_fuzzificacao,
        semente_pca=opcoes.semente_pca,
    )
    print(resumo.to_json(orient="records", force_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
