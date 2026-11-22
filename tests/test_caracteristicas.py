from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clustering_imagens.caracteristicas import (
    EstadoExtracao,
    carregar_estado_extracao,
    caminhos_extracao_parcial,
    ler_caminhos_amostra,
    salvar_estado_extracao,
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


def test_checkpoint_de_extracao_pode_ser_retomado(tmp_path: Path) -> None:
    arquivo_saida = tmp_path / "caracteristicas.npz"
    caminho_matriz, caminho_estado = caminhos_extracao_parcial(arquivo_saida)
    matriz = np.lib.format.open_memmap(
        caminho_matriz, mode="w+", dtype=np.float32, shape=(5, 3)
    )
    matriz[:2] = 1
    matriz.flush()
    del matriz
    estado = EstadoExtracao(
        sha256_amostra="abc",
        diretorio_imagens=str(tmp_path),
        quantidade_total=5,
        quantidade_processada=2,
        dimensao_caracteristicas=3,
    )
    salvar_estado_extracao(caminho_estado, estado)

    carregado = carregar_estado_extracao(
        caminho_matriz, caminho_estado, "abc", tmp_path, 5
    )

    assert carregado == estado


def test_checkpoint_rejeita_outra_amostra(tmp_path: Path) -> None:
    arquivo_saida = tmp_path / "caracteristicas.npz"
    caminho_matriz, caminho_estado = caminhos_extracao_parcial(arquivo_saida)
    matriz = np.lib.format.open_memmap(
        caminho_matriz, mode="w+", dtype=np.float32, shape=(2, 3)
    )
    matriz.flush()
    del matriz
    salvar_estado_extracao(
        caminho_estado,
        EstadoExtracao("abc", str(tmp_path), 2, 1, 3),
    )

    with pytest.raises(RuntimeError, match="outra amostra"):
        carregar_estado_extracao(
            caminho_matriz, caminho_estado, "diferente", tmp_path, 2
        )
