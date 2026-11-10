from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from clustering_imagens.inventario import (
    criar_inventario,
    executar_inventario,
    listar_caminhos_jpeg,
    selecionar_amostra,
)


def criar_jpeg(caminho: Path, cor: tuple[int, int, int] = (10, 20, 30)) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), cor).save(caminho, format="JPEG")


def test_lista_jpegs_recursivamente_em_ordem_estavel(tmp_path: Path) -> None:
    criar_jpeg(tmp_path / "z.jpg")
    criar_jpeg(tmp_path / "subpasta" / "A.JPEG")
    (tmp_path / "ignorar.png").write_bytes(b"nao e jpeg")

    caminhos = listar_caminhos_jpeg(tmp_path)

    assert [caminho.relative_to(tmp_path).as_posix() for caminho in caminhos] == [
        "subpasta/A.JPEG",
        "z.jpg",
    ]


def test_inventario_registra_imagem_corrompida(tmp_path: Path) -> None:
    criar_jpeg(tmp_path / "valida.jpg")
    (tmp_path / "corrompida.jpg").write_bytes(b"conteudo invalido")

    registros = criar_inventario(tmp_path)

    assert [registro.status for registro in registros] == ["excluida", "valida"]
    assert registros[0].motivo_exclusao
    assert registros[1].largura == 8
    assert registros[1].altura == 6


def test_amostra_e_reproduzivel_e_preserva_ordem_do_sorteio(tmp_path: Path) -> None:
    for indice in range(8):
        criar_jpeg(tmp_path / f"imagem_{indice:02}.jpg", (indice, 20, 30))
    registros = criar_inventario(tmp_path)

    primeira = selecionar_amostra(registros, tamanho=4, semente=123)
    segunda = selecionar_amostra(registros, tamanho=4, semente=123)

    assert primeira == segunda
    assert len({registro.caminho_relativo for registro in primeira}) == 4


def test_amostra_rejeita_quantidade_maior_que_as_validas(tmp_path: Path) -> None:
    criar_jpeg(tmp_path / "unica.jpg")
    registros = criar_inventario(tmp_path)

    with pytest.raises(ValueError, match="solicitadas=2, disponíveis=1"):
        selecionar_amostra(registros, tamanho=2, semente=42)


def test_execucao_salva_artefatos_consistentes(tmp_path: Path) -> None:
    diretorio_imagens = tmp_path / "imagens"
    diretorio_saida = tmp_path / "saida"
    for indice in range(3):
        criar_jpeg(diretorio_imagens / f"imagem_{indice}.jpg")
    (diretorio_imagens / "quebrada.jpg").write_bytes(b"invalida")

    resumo = executar_inventario(
        diretorio_imagens, diretorio_saida, tamanho_amostra=2, semente=7
    )

    assert resumo.quantidade_encontrada == 4
    assert resumo.quantidade_valida == 3
    assert resumo.quantidade_excluida == 1
    assert resumo.tamanho_amostra == 2
    assert {caminho.name for caminho in diretorio_saida.iterdir()} == {
        "amostra_piloto.csv",
        "exclusoes.csv",
        "inventario.csv",
        "resumo.json",
    }

    with (diretorio_saida / "amostra_piloto.csv").open(encoding="utf-8") as arquivo:
        amostra = list(csv.DictReader(arquivo))
    assert [linha["ordem_sorteio"] for linha in amostra] == ["1", "2"]

    with (diretorio_saida / "resumo.json").open(encoding="utf-8") as arquivo:
        resumo_salvo = json.load(arquivo)
    assert resumo_salvo["semente"] == 7
