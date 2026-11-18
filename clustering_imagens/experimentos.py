"""Execução comparável do piloto com múltiplas representações e sementes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from clustering_imagens.piloto import executar_piloto


@dataclass(frozen=True, slots=True)
class EstabilidadeExecucao:
    """Comparação de uma execução com a semente de referência."""

    componentes_pca: int
    semente_referencia: int
    semente_comparada: int
    concordancia_dominante: float
    indice_rand_ajustado: float
    diferenca_media_pertinencias: float
    distancia_media_centroides: float


def carregar_modelo(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega cópias dos centroides e pertinências de um resultado salvo."""

    with np.load(caminho) as dados:
        return dados["centroides"].copy(), dados["pertinencias"].copy()


def comparar_com_referencia(
    centroides_referencia: np.ndarray,
    pertinencias_referencia: np.ndarray,
    centroides_comparados: np.ndarray,
    pertinencias_comparadas: np.ndarray,
    componentes_pca: int,
    semente_referencia: int,
    semente_comparada: int,
) -> EstabilidadeExecucao:
    """Alinha clusters por centroides e mede a estabilidade das pertinências."""

    if centroides_referencia.shape != centroides_comparados.shape:
        raise ValueError("Os conjuntos de centroides têm formatos diferentes.")
    if pertinencias_referencia.shape != pertinencias_comparadas.shape:
        raise ValueError("As matrizes de pertinência têm formatos diferentes.")

    custos = np.sum(
        (centroides_referencia[:, None, :] - centroides_comparados[None, :, :]) ** 2,
        axis=2,
    )
    linhas, colunas = linear_sum_assignment(custos)
    ordem = colunas[np.argsort(linhas)]
    pertinencias_alinhadas = pertinencias_comparadas[:, ordem]
    centroides_alinhados = centroides_comparados[ordem]

    rotulos_referencia = pertinencias_referencia.argmax(axis=1)
    rotulos_alinhados = pertinencias_alinhadas.argmax(axis=1)
    distancias_centroides = np.linalg.norm(
        centroides_referencia - centroides_alinhados, axis=1
    )
    return EstabilidadeExecucao(
        componentes_pca=componentes_pca,
        semente_referencia=semente_referencia,
        semente_comparada=semente_comparada,
        concordancia_dominante=float(
            np.mean(rotulos_referencia == rotulos_alinhados)
        ),
        indice_rand_ajustado=float(
            adjusted_rand_score(rotulos_referencia, rotulos_alinhados)
        ),
        diferenca_media_pertinencias=float(
            np.mean(np.abs(pertinencias_referencia - pertinencias_alinhadas))
        ),
        distancia_media_centroides=float(distancias_centroides.mean()),
    )


def executar_grade_piloto(
    arquivo_caracteristicas: Path,
    diretorio_saida: Path,
    componentes_pca: Sequence[int] = (64, 128),
    sementes: Sequence[int] = (42, 123, 2026),
    quantidade_clusters: int = 10,
    expoente_fuzzificacao: float = 1.2,
    semente_pca: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa a grade, alinha resultados e salva tabelas comparáveis."""

    if not componentes_pca:
        raise ValueError("Informe ao menos uma quantidade de componentes PCA.")
    if not sementes:
        raise ValueError("Informe ao menos uma semente do agrupamento.")
    if len(set(sementes)) != len(sementes):
        raise ValueError("As sementes não podem se repetir.")

    diretorio_saida.mkdir(parents=True, exist_ok=True)
    linhas_resultados: list[dict[str, object]] = []
    linhas_estabilidade: list[dict[str, object]] = []

    for quantidade_componentes in componentes_pca:
        modelos: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for semente in sementes:
            diretorio_execucao = (
                diretorio_saida
                / f"pca_{quantidade_componentes:03}"
                / f"semente_{semente:04}"
            )
            metricas = executar_piloto(
                arquivo_caracteristicas=arquivo_caracteristicas,
                diretorio_saida=diretorio_execucao,
                componentes_pca=quantidade_componentes,
                quantidade_clusters=quantidade_clusters,
                expoente_fuzzificacao=expoente_fuzzificacao,
                semente=semente,
                semente_pca=semente_pca,
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
            estabilidade = comparar_com_referencia(
                centroides_referencia,
                pertinencias_referencia,
                centroides,
                pertinencias,
                quantidade_componentes,
                semente_referencia,
                semente,
            )
            linhas_estabilidade.append(asdict(estabilidade))

    resultados = pd.DataFrame(linhas_resultados)
    estabilidade = pd.DataFrame(linhas_estabilidade)
    linhas_resumo: list[dict[str, object]] = []
    for quantidade_componentes in componentes_pca:
        execucoes = resultados[resultados["componentes_pca"] == quantidade_componentes]
        comparacoes = estabilidade[
            (estabilidade["componentes_pca"] == quantidade_componentes)
            & (
                estabilidade["semente_comparada"]
                != estabilidade["semente_referencia"]
            )
        ]
        linhas_resumo.append(
            {
                "componentes_pca": quantidade_componentes,
                "variancia_explicada_pca": execucoes[
                    "variancia_explicada_pca"
                ].mean(),
                "execucoes": len(execucoes),
                "convergencias": int(execucoes["convergiu"].sum()),
                "xie_beni_medio": execucoes["indice_xie_beni"].mean(),
                "silhueta_fcm_media": execucoes[
                    "silhueta_atribuicao_dominante"
                ].mean(),
                "silhueta_kmeans_media": execucoes["kmeans_silhueta"].mean(),
                "pertinencia_principal_media": execucoes[
                    "pertinencia_principal_media"
                ].mean(),
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
    resumo = pd.DataFrame(linhas_resumo)

    resultados.to_csv(diretorio_saida / "resultados.csv", index=False)
    estabilidade.to_csv(diretorio_saida / "estabilidade.csv", index=False)
    resumo.to_csv(diretorio_saida / "resumo_representacoes.csv", index=False)
    with (diretorio_saida / "configuracao.json").open("w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "arquivo_caracteristicas": str(arquivo_caracteristicas.resolve()),
                "componentes_pca": list(componentes_pca),
                "sementes": list(sementes),
                "quantidade_clusters": quantidade_clusters,
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
    """Configura a linha de comando da grade experimental."""

    analisador = argparse.ArgumentParser(
        description="Compara PCA-64/PCA-128 e múltiplas sementes no piloto."
    )
    analisador.add_argument("--arquivo-caracteristicas", type=Path, required=True)
    analisador.add_argument(
        "--diretorio-saida",
        type=Path,
        default=Path("artefatos/experimentos"),
    )
    analisador.add_argument(
        "--componentes-pca", type=int, nargs="+", default=(64, 128)
    )
    analisador.add_argument("--sementes", type=int, nargs="+", default=(42, 123, 2026))
    analisador.add_argument("--quantidade-clusters", type=int, default=10)
    analisador.add_argument("--expoente-fuzzificacao", type=float, default=1.2)
    analisador.add_argument("--semente-pca", type=int, default=42)
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa a grade pela linha de comando."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    _, _, resumo = executar_grade_piloto(
        arquivo_caracteristicas=opcoes.arquivo_caracteristicas,
        diretorio_saida=opcoes.diretorio_saida,
        componentes_pca=opcoes.componentes_pca,
        sementes=opcoes.sementes,
        quantidade_clusters=opcoes.quantidade_clusters,
        expoente_fuzzificacao=opcoes.expoente_fuzzificacao,
        semente_pca=opcoes.semente_pca,
    )
    print(resumo.to_json(orient="records", force_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
