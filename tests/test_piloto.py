from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from clustering_imagens.piloto import (
    calcular_xie_beni,
    exportar_pertinencias,
    fuzzy_c_means,
    preparar_representacao,
)


def criar_dados_separados() -> np.ndarray:
    gerador = np.random.default_rng(9)
    primeiro = gerador.normal(loc=(-3.0, -3.0), scale=0.15, size=(30, 2))
    segundo = gerador.normal(loc=(3.0, 3.0), scale=0.15, size=(30, 2))
    return np.vstack((primeiro, segundo))


def test_fcm_converge_e_pertinencias_somam_um() -> None:
    dados = criar_dados_separados()

    resultado = fuzzy_c_means(dados, quantidade_clusters=2, semente=11)

    assert resultado.convergiu
    assert resultado.centroides.shape == (2, 2)
    assert resultado.pertinencias.shape == (60, 2)
    np.testing.assert_allclose(resultado.pertinencias.sum(axis=1), 1.0)
    assert np.mean(np.max(resultado.pertinencias, axis=1)) > 0.99
    assert calcular_xie_beni(resultado, dados, 2.0) < 0.01


def test_fcm_e_reproduzivel() -> None:
    dados = criar_dados_separados()

    primeiro = fuzzy_c_means(dados, 2, semente=123)
    segundo = fuzzy_c_means(dados, 2, semente=123)

    np.testing.assert_allclose(primeiro.centroides, segundo.centroides)
    np.testing.assert_allclose(primeiro.pertinencias, segundo.pertinencias)


def test_fcm_aceita_inicializacao_por_pertinencias_aleatorias() -> None:
    dados = criar_dados_separados()

    resultado = fuzzy_c_means(
        dados, 2, semente=123, inicializacao="pertinencias_aleatorias"
    )

    assert resultado.convergiu
    np.testing.assert_allclose(resultado.pertinencias.sum(axis=1), 1.0)


def test_preparacao_aplica_pca_e_normalizacao_l2() -> None:
    gerador = np.random.default_rng(5)
    caracteristicas = gerador.normal(size=(20, 8))

    representacao, variancia = preparar_representacao(caracteristicas, 3, 42)

    assert representacao.shape == (20, 3)
    np.testing.assert_allclose(np.linalg.norm(representacao, axis=1), 1.0)
    assert variancia is not None and 0 < variancia <= 1


def test_rejeita_quantidade_invalida_de_clusters() -> None:
    with pytest.raises(ValueError, match="quantidade de clusters"):
        fuzzy_c_means(np.ones((3, 2)), quantidade_clusters=3)


def test_exporta_resumo_e_matriz_completa(tmp_path: Path) -> None:
    pertinencias = np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    destino = tmp_path / "pertinencias.csv"

    exportar_pertinencias(
        destino, ["a.jpg", "b.jpg"], pertinencias, np.array([2, 0])
    )

    with destino.open(encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert linhas[0]["cluster_principal"] == "1"
    assert linhas[0]["cluster_secundario"] == "2"
    assert float(linhas[0]["diferenca_top2"]) == pytest.approx(0.5)
    assert linhas[1]["pertinencia_cluster_003"] == "0.5"
    assert linhas[0]["cluster_kmeans"] == "3"
