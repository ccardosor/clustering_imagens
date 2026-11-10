from __future__ import annotations

import numpy as np
import pytest

from clustering_imagens.experimentos import comparar_com_referencia


def test_alinhamento_recupera_permutacao_de_clusters() -> None:
    centroides = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0]])
    pertinencias = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    permutacao = [2, 0, 1]

    estabilidade = comparar_com_referencia(
        centroides,
        pertinencias,
        centroides[permutacao],
        pertinencias[:, permutacao],
        componentes_pca=2,
        semente_referencia=42,
        semente_comparada=123,
    )

    assert estabilidade.concordancia_dominante == 1.0
    assert estabilidade.indice_rand_ajustado == 1.0
    assert estabilidade.diferenca_media_pertinencias == pytest.approx(0.0)
    assert estabilidade.distancia_media_centroides == pytest.approx(0.0)
