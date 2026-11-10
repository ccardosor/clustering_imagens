from __future__ import annotations

from pathlib import Path

import pytest

from clustering_imagens.caracteristicas import (
    ler_caminhos_amostra,
    validar_caminhos_imagens,
)


def test_le_caminhos_na_ordem_da_amostra(tmp_path: Path) -> None:
    amostra = tmp_path / "amostra.csv"
    amostra.write_text(
        "ordem_sorteio,caminho_relativo\n1,b.jpg\n2,a.jpg\n", encoding="utf-8"
    )

    assert ler_caminhos_amostra(amostra) == ["b.jpg", "a.jpg"]


def test_rejeita_caminhos_duplicados(tmp_path: Path) -> None:
    amostra = tmp_path / "amostra.csv"
    amostra.write_text(
        "ordem_sorteio,caminho_relativo\n1,a.jpg\n2,a.jpg\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicados"):
        ler_caminhos_amostra(amostra)


def test_informa_imagens_ausentes(tmp_path: Path) -> None:
    (tmp_path / "presente.jpg").touch()

    with pytest.raises(FileNotFoundError, match="ausente.jpg"):
        validar_caminhos_imagens(tmp_path, ["presente.jpg", "ausente.jpg"])
