from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clustering_imagens.visualizacao import (
    resolver_caminho_imagem,
    selecionar_galeria,
)


def criar_tabela() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caminho_relativo": ["a.jpg", "b.jpg", "c.jpg"],
            "cluster_principal": [1, 2, 1],
            "cluster_secundario": [2, 1, 2],
            "pertinencia_principal": [0.8, 0.55, 0.51],
            "pertinencia_secundaria": [0.1, 0.4, 0.49],
            "diferenca_top2": [0.7, 0.15, 0.02],
            "pertinencia_cluster_001": [0.8, 0.4, 0.51],
            "pertinencia_cluster_002": [0.1, 0.55, 0.49],
        }
    )


def test_seleciona_imagens_tipicas_por_cluster() -> None:
    resultado = selecionar_galeria(criar_tabela(), "Típicas", cluster=2, limite=2)

    assert resultado["caminho_relativo"].tolist() == ["b.jpg", "c.jpg"]


def test_seleciona_imagens_ambiguas_por_diferenca() -> None:
    resultado = selecionar_galeria(criar_tabela(), "Ambíguas", cluster=None, limite=2)

    assert resultado["caminho_relativo"].tolist() == ["c.jpg", "b.jpg"]


def test_impede_caminho_fora_do_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inválido"):
        resolver_caminho_imagem(tmp_path, "../fora.jpg")
