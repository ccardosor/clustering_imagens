"""Inventário e seleção reproduzível das imagens do projeto."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, UnidentifiedImageError

EXTENSOES_JPEG = {".jpg", ".jpeg"}
CAMPOS_INVENTARIO = (
    "caminho_relativo",
    "status",
    "motivo_exclusao",
    "largura",
    "altura",
    "modo_cor",
    "tamanho_bytes",
)


@dataclass(frozen=True, slots=True)
class RegistroImagem:
    """Resultado da validação de um arquivo de imagem."""

    caminho_relativo: str
    status: str
    motivo_exclusao: str
    largura: int | None
    altura: int | None
    modo_cor: str
    tamanho_bytes: int


@dataclass(frozen=True, slots=True)
class ResumoInventario:
    """Metadados necessários para reproduzir o inventário e a amostra."""

    diretorio_imagens: str
    quantidade_encontrada: int
    quantidade_valida: int
    quantidade_excluida: int
    tamanho_amostra: int
    semente: int


def listar_caminhos_jpeg(diretorio_imagens: Path) -> list[Path]:
    """Lista JPEGs recursivamente em ordem estável pelo caminho relativo."""

    if not diretorio_imagens.is_dir():
        raise NotADirectoryError(
            f"Diretório de imagens não encontrado: {diretorio_imagens}"
        )

    caminhos = (
        caminho
        for caminho in diretorio_imagens.rglob("*")
        if caminho.is_file() and caminho.suffix.casefold() in EXTENSOES_JPEG
    )
    return sorted(
        caminhos,
        key=lambda caminho: caminho.relative_to(diretorio_imagens).as_posix().casefold(),
    )


def validar_imagem(caminho: Path, diretorio_imagens: Path) -> RegistroImagem:
    """Decodifica uma imagem e registra dimensões ou a causa da exclusão."""

    caminho_relativo = caminho.relative_to(diretorio_imagens).as_posix()
    tamanho_bytes = caminho.stat().st_size

    try:
        with Image.open(caminho) as imagem:
            largura, altura = imagem.size
            modo_cor = imagem.mode
            formato = imagem.format
            imagem.verify()

        if formato != "JPEG":
            raise ValueError(f"formato detectado inesperado: {formato or 'desconhecido'}")
        if largura <= 0 or altura <= 0:
            raise ValueError(f"dimensões inválidas: {largura}x{altura}")

        # verify() valida a estrutura; load() confirma que os pixels são decodificáveis.
        with Image.open(caminho) as imagem:
            imagem.load()
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as erro:
        return RegistroImagem(
            caminho_relativo=caminho_relativo,
            status="excluida",
            motivo_exclusao=f"{type(erro).__name__}: {erro}",
            largura=None,
            altura=None,
            modo_cor="",
            tamanho_bytes=tamanho_bytes,
        )

    return RegistroImagem(
        caminho_relativo=caminho_relativo,
        status="valida",
        motivo_exclusao="",
        largura=largura,
        altura=altura,
        modo_cor=modo_cor,
        tamanho_bytes=tamanho_bytes,
    )


def criar_inventario(diretorio_imagens: Path) -> list[RegistroImagem]:
    """Valida todos os JPEGs encontrados no diretório informado."""

    return [
        validar_imagem(caminho, diretorio_imagens)
        for caminho in listar_caminhos_jpeg(diretorio_imagens)
    ]


def selecionar_amostra(
    registros: Sequence[RegistroImagem], tamanho: int, semente: int
) -> list[RegistroImagem]:
    """Sorteia uma amostra de imagens válidas e preserva a ordem do sorteio."""

    if tamanho <= 0:
        raise ValueError("O tamanho da amostra deve ser maior que zero.")

    registros_validos = [registro for registro in registros if registro.status == "valida"]
    if len(registros_validos) < tamanho:
        raise ValueError(
            "Não há imagens válidas suficientes para a amostra: "
            f"solicitadas={tamanho}, disponíveis={len(registros_validos)}."
        )

    gerador = random.Random(semente)
    return gerador.sample(registros_validos, tamanho)


def escrever_csv(
    destino: Path, registros: Iterable[RegistroImagem], campos: Sequence[str]
) -> None:
    """Escreve registros em CSV UTF-8 com cabeçalho estável."""

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(asdict(registro) for registro in registros)


def executar_inventario(
    diretorio_imagens: Path,
    diretorio_saida: Path,
    tamanho_amostra: int = 500,
    semente: int = 42,
) -> ResumoInventario:
    """Executa o primeiro marco do projeto e salva seus quatro artefatos."""

    diretorio_imagens = diretorio_imagens.resolve()
    registros = criar_inventario(diretorio_imagens)
    if not registros:
        raise ValueError(f"Nenhum arquivo JPEG encontrado em {diretorio_imagens}.")

    excluidos = [registro for registro in registros if registro.status == "excluida"]
    amostra = selecionar_amostra(registros, tamanho_amostra, semente)
    quantidade_valida = len(registros) - len(excluidos)

    escrever_csv(diretorio_saida / "inventario.csv", registros, CAMPOS_INVENTARIO)
    escrever_csv(diretorio_saida / "exclusoes.csv", excluidos, CAMPOS_INVENTARIO)

    campos_amostra = ("ordem_sorteio", "caminho_relativo")
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    with (diretorio_saida / "amostra_piloto.csv").open(
        "w", encoding="utf-8", newline=""
    ) as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos_amostra)
        escritor.writeheader()
        escritor.writerows(
            {
                "ordem_sorteio": ordem,
                "caminho_relativo": registro.caminho_relativo,
            }
            for ordem, registro in enumerate(amostra, start=1)
        )

    resumo = ResumoInventario(
        diretorio_imagens=str(diretorio_imagens),
        quantidade_encontrada=len(registros),
        quantidade_valida=quantidade_valida,
        quantidade_excluida=len(excluidos),
        tamanho_amostra=len(amostra),
        semente=semente,
    )
    with (diretorio_saida / "resumo.json").open("w", encoding="utf-8") as arquivo:
        json.dump(asdict(resumo), arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")

    return resumo


def criar_analisador_argumentos() -> argparse.ArgumentParser:
    """Configura a interface de linha de comando."""

    analisador = argparse.ArgumentParser(
        description="Valida JPEGs e sorteia a amostra reproduzível do piloto."
    )
    analisador.add_argument(
        "--diretorio-imagens",
        type=Path,
        default=Path("102flowers/jpg"),
        help="pasta que contém os JPEGs (padrão: 102flowers/jpg)",
    )
    analisador.add_argument(
        "--diretorio-saida",
        type=Path,
        default=Path("artefatos/inventario"),
        help="pasta para os CSVs e o resumo (padrão: artefatos/inventario)",
    )
    analisador.add_argument(
        "--tamanho-amostra",
        type=int,
        default=500,
        help="quantidade de imagens no piloto (padrão: 500)",
    )
    analisador.add_argument(
        "--semente", type=int, default=42, help="semente do sorteio (padrão: 42)"
    )
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa a interface de linha de comando."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    resumo = executar_inventario(
        diretorio_imagens=opcoes.diretorio_imagens,
        diretorio_saida=opcoes.diretorio_saida,
        tamanho_amostra=opcoes.tamanho_amostra,
        semente=opcoes.semente,
    )
    print(json.dumps(asdict(resumo), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
