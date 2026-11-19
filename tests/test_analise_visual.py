from __future__ import annotations

import pandas as pd

from clustering_imagens.analise_visual import selecionar_ambiguas, selecionar_tipicas


def criar_pertinencias() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caminho_relativo": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "cluster_principal": [1, 1, 2, 2],
            "cluster_secundario": [2, 2, 1, 1],
            "pertinencia_principal": [0.9, 0.6, 0.8, 0.51],
            "pertinencia_secundaria": [0.1, 0.4, 0.2, 0.49],
            "diferenca_top2": [0.8, 0.2, 0.6, 0.02],
            "pertinencia_cluster_001": [0.9, 0.6, 0.2, 0.49],
            "pertinencia_cluster_002": [0.1, 0.4, 0.8, 0.51],
        }
    )


def test_seleciona_tipicas_dominantes_em_ordem() -> None:
    resultado = selecionar_tipicas(criar_pertinencias(), cluster=1, quantidade=2)

    assert resultado["caminho_relativo"].tolist() == ["a.jpg", "b.jpg"]


def test_seleciona_ambiguas_por_menor_diferenca() -> None:
    resultado = selecionar_ambiguas(criar_pertinencias(), quantidade=2)

    assert resultado["caminho_relativo"].tolist() == ["d.jpg", "b.jpg"]
