"""Comparação de expoentes de fuzzificação para a configuração candidata."""

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


def rotulo_expoente(expoente: float) -> str:
    """Converte o expoente em um nome estável para diretórios."""

    return f"m_{expoente:.3f}".replace(".", "p")


def resumir_fuzzificacao(
    resultados: pd.DataFrame,
    estabilidade: pd.DataFrame,
    expoentes: Sequence[float],
) -> pd.DataFrame:
    """Resume qualidade, convergência e estabilidade de cada expoente."""

    linhas: list[dict[str, object]] = []
    for expoente in expoentes:
        execucoes = resultados[
            np.isclose(resultados["expoente_fuzzificacao"], expoente)
        ]
        comparacoes = estabilidade[
            np.isclose(estabilidade["expoente_fuzzificacao"], expoente)
            & (
                estabilidade["semente_comparada"]
                != estabilidade["semente_referencia"]
            )
        ]
        if execucoes.empty:
            raise ValueError(f"Não há execuções para m={expoente:.3f}.")
        quantidades = execucoes["quantidade_clusters"].unique()
        if len(quantidades) != 1:
            raise ValueError("Cada expoente deve usar uma única quantidade de grupos.")
        linhas.append(
            {
                "expoente_fuzzificacao": expoente,
                "quantidade_clusters": int(quantidades[0]),
                "execucoes": len(execucoes),
                "convergencias": int(execucoes["convergiu"].sum()),
                "grupos_ocupados_minimo": int(
                    execucoes["clusters_dominantes_ocupados"].min()
                ),
                "iteracoes_media": execucoes["iteracoes_realizadas"].mean(),
                "xie_beni_medio": execucoes["indice_xie_beni"].mean(),
                "xie_beni_mediano": execucoes["indice_xie_beni"].median(),
                "silhueta_fcm_media": execucoes[
                    "silhueta_atribuicao_dominante"
                ].mean(),
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


def executar_avaliacao_fuzzificacao(
    arquivo_caracteristicas: Path,
    diretorio_saida: Path,
    expoentes: Sequence[float] = (1.1, 1.15, 1.2, 1.25, 1.3),
    sementes: Sequence[int] = (42, 123, 2026),
    componentes_pca: int = 64,
    quantidade_clusters: int = 8,
    semente_pca: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Executa e salva a comparação entre expoentes de fuzzificação."""

    if not expoentes:
        raise ValueError("Informe ao menos um expoente de fuzzificação.")
    if any(expoente <= 1 for expoente in expoentes):
        raise ValueError("Todos os expoentes devem ser maiores que 1.")
    if len(set(expoentes)) != len(expoentes):
        raise ValueError("Os expoentes não podem se repetir.")
    if len(sementes) < 2:
        raise ValueError("Informe ao menos duas sementes para medir estabilidade.")
    if len(set(sementes)) != len(sementes):
        raise ValueError("As sementes não podem se repetir.")

    diretorio_saida.mkdir(parents=True, exist_ok=True)
    linhas_resultados: list[dict[str, object]] = []
    linhas_estabilidade: list[dict[str, object]] = []

    for expoente in expoentes:
        modelos: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for semente in sementes:
            diretorio_execucao = (
                diretorio_saida
                / rotulo_expoente(expoente)
                / f"semente_{semente:04}"
            )
            metricas = executar_piloto(
                arquivo_caracteristicas=arquivo_caracteristicas,
                diretorio_saida=diretorio_execucao,
                componentes_pca=componentes_pca,
                quantidade_clusters=quantidade_clusters,
                expoente_fuzzificacao=expoente,
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
            linha_estabilidade["expoente_fuzzificacao"] = expoente
            linhas_estabilidade.append(linha_estabilidade)

    resultados = pd.DataFrame(linhas_resultados)
    estabilidade = pd.DataFrame(linhas_estabilidade)
    resumo = resumir_fuzzificacao(resultados, estabilidade, expoentes)

    resultados.to_csv(diretorio_saida / "resultados_fuzzificacao.csv", index=False)
    estabilidade.to_csv(
        diretorio_saida / "estabilidade_fuzzificacao.csv", index=False
    )
    resumo.to_csv(diretorio_saida / "resumo_fuzzificacao.csv", index=False)
    with (diretorio_saida / "configuracao.json").open(
        "w", encoding="utf-8"
    ) as arquivo:
        json.dump(
            {
                "arquivo_caracteristicas": str(arquivo_caracteristicas.resolve()),
                "expoentes": list(expoentes),
                "sementes": list(sementes),
                "componentes_pca": componentes_pca,
                "quantidade_clusters": quantidade_clusters,
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
        description="Compara expoentes de fuzzificação com PCA-64 e 8 grupos."
    )
    analisador.add_argument("--arquivo-caracteristicas", type=Path, required=True)
    analisador.add_argument(
        "--diretorio-saida",
        type=Path,
        default=Path("artefatos/avaliacao_fuzzificacao"),
    )
    analisador.add_argument(
        "--expoentes", type=float, nargs="+", default=(1.1, 1.15, 1.2, 1.25, 1.3)
    )
    analisador.add_argument("--sementes", type=int, nargs="+", default=(42, 123, 2026))
    analisador.add_argument("--componentes-pca", type=int, default=64)
    analisador.add_argument("--quantidade-clusters", type=int, default=8)
    analisador.add_argument("--semente-pca", type=int, default=42)
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa a avaliação pela linha de comando."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    _, _, resumo = executar_avaliacao_fuzzificacao(
        arquivo_caracteristicas=opcoes.arquivo_caracteristicas,
        diretorio_saida=opcoes.diretorio_saida,
        expoentes=opcoes.expoentes,
        sementes=opcoes.sementes,
        componentes_pca=opcoes.componentes_pca,
        quantidade_clusters=opcoes.quantidade_clusters,
        semente_pca=opcoes.semente_pca,
    )
    print(resumo.to_json(orient="records", force_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
