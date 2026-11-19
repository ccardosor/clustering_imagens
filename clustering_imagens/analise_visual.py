"""Geração de pranchas reproduzíveis para inspeção dos agrupamentos."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from clustering_imagens.visualizacao import resolver_caminho_imagem


def selecionar_tipicas(
    pertinencias: pd.DataFrame, cluster: int, quantidade: int
) -> pd.DataFrame:
    """Seleciona imagens dominantes com maior pertinência a um cluster."""

    coluna = f"pertinencia_cluster_{cluster:03}"
    if coluna not in pertinencias.columns:
        raise ValueError(f"A coluna {coluna} não foi encontrada.")
    candidatas = pertinencias[pertinencias["cluster_principal"] == cluster]
    return candidatas.sort_values(coluna, ascending=False).head(quantidade).copy()


def selecionar_ambiguas(
    pertinencias: pd.DataFrame, quantidade: int
) -> pd.DataFrame:
    """Seleciona as menores diferenças entre as duas maiores pertinências."""

    return (
        pertinencias.sort_values("diferenca_top2", ascending=True)
        .head(quantidade)
        .copy()
    )


def carregar_fonte(
    tamanho: int, negrito: bool = False
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Carrega uma fonte legível com fallback portável."""

    nomes = (
        ("arialbd.ttf", "DejaVuSans-Bold.ttf")
        if negrito
        else ("arial.ttf", "DejaVuSans.ttf")
    )
    for nome in nomes:
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def criar_prancha(
    selecao: pd.DataFrame,
    diretorio_imagens: Path,
    destino: Path,
    titulo: str,
    colunas: int = 4,
) -> None:
    """Monta uma prancha JPEG com miniaturas, arquivos e pertinências."""

    if selecao.empty:
        raise ValueError("Não há imagens para montar a prancha.")

    largura_imagem, altura_imagem = 260, 190
    altura_legenda, altura_titulo = 66, 54
    margem, espacamento = 16, 12
    linhas = (len(selecao) + colunas - 1) // colunas
    largura = margem * 2 + colunas * largura_imagem + (colunas - 1) * espacamento
    altura = (
        margem * 2
        + altura_titulo
        + linhas * (altura_imagem + altura_legenda)
        + (linhas - 1) * espacamento
    )
    prancha = Image.new("RGB", (largura, altura), "#f7f7f5")
    desenho = ImageDraw.Draw(prancha)
    fonte_titulo = carregar_fonte(24, negrito=True)
    fonte_legenda = carregar_fonte(14)
    desenho.text((margem, margem), titulo, fill="#202020", font=fonte_titulo)

    for posicao, (_, registro) in enumerate(selecao.iterrows()):
        linha, coluna = divmod(posicao, colunas)
        esquerda = margem + coluna * (largura_imagem + espacamento)
        topo = margem + altura_titulo + linha * (
            altura_imagem + altura_legenda + espacamento
        )
        caminho = resolver_caminho_imagem(
            diretorio_imagens, str(registro["caminho_relativo"])
        )
        with Image.open(caminho) as imagem_aberta:
            imagem = ImageOps.exif_transpose(imagem_aberta).convert("RGB")
            miniatura = ImageOps.contain(
                imagem, (largura_imagem, altura_imagem), Image.Resampling.LANCZOS
            )
        fundo = Image.new("RGB", (largura_imagem, altura_imagem), "#dededb")
        deslocamento = (
            (largura_imagem - miniatura.width) // 2,
            (altura_imagem - miniatura.height) // 2,
        )
        fundo.paste(miniatura, deslocamento)
        prancha.paste(fundo, (esquerda, topo))

        nome = str(registro["caminho_relativo"])
        principal = int(registro["cluster_principal"])
        secundaria = int(registro["cluster_secundario"])
        mu = float(registro["pertinencia_principal"])
        diferenca = float(registro["diferenca_top2"])
        legenda = (
            f"{nome}\nG{principal} / G{secundaria}  "
            f"mu={mu:.3f}  delta={diferenca:.3f}"
        )
        desenho.multiline_text(
            (esquerda, topo + altura_imagem + 6),
            legenda,
            fill="#252525",
            font=fonte_legenda,
            spacing=3,
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    prancha.save(destino, format="JPEG", quality=92, optimize=True)


def gerar_pranchas(
    arquivo_pertinencias: Path,
    diretorio_imagens: Path,
    diretorio_saida: Path,
    quantidade_tipicas: int = 12,
    quantidade_ambiguas: int = 20,
) -> pd.DataFrame:
    """Gera uma prancha por grupo e uma prancha global de fronteiras."""

    pertinencias = pd.read_csv(arquivo_pertinencias)
    quantidade_clusters = sum(
        coluna.startswith("pertinencia_cluster_")
        for coluna in pertinencias.columns
    )
    if quantidade_clusters < 2:
        raise ValueError("A tabela não contém pertinências completas dos clusters.")

    selecoes: list[pd.DataFrame] = []
    for cluster in range(1, quantidade_clusters + 1):
        tipicas = selecionar_tipicas(pertinencias, cluster, quantidade_tipicas)
        tipicas = tipicas.assign(tipo="tipica", cluster_referencia=cluster)
        selecoes.append(tipicas)
        criar_prancha(
            tipicas,
            diretorio_imagens,
            diretorio_saida / f"grupo_{cluster:02}_tipicas.jpg",
            f"Grupo {cluster} - imagens tipicas",
        )

    ambiguas = selecionar_ambiguas(pertinencias, quantidade_ambiguas)
    ambiguas = ambiguas.assign(tipo="ambigua", cluster_referencia=pd.NA)
    selecoes.append(ambiguas)
    criar_prancha(
        ambiguas,
        diretorio_imagens,
        diretorio_saida / "fronteiras_ambiguas.jpg",
        "Imagens de fronteira - menor diferenca entre pertinencias",
        colunas=5,
    )

    tabela_selecao = pd.concat(selecoes, ignore_index=True)
    tabela_selecao.to_csv(diretorio_saida / "selecao_pranchas.csv", index=False)
    return tabela_selecao


def criar_analisador_argumentos() -> argparse.ArgumentParser:
    """Configura a interface de linha de comando."""

    analisador = argparse.ArgumentParser(
        description="Gera pranchas com imagens típicas e de fronteira."
    )
    analisador.add_argument("--arquivo-pertinencias", type=Path, required=True)
    analisador.add_argument("--diretorio-imagens", type=Path, required=True)
    analisador.add_argument(
        "--diretorio-saida",
        type=Path,
        default=Path("artefatos/analise_visual"),
    )
    analisador.add_argument("--quantidade-tipicas", type=int, default=12)
    analisador.add_argument("--quantidade-ambiguas", type=int, default=20)
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa a geração de pranchas pela linha de comando."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    tabela = gerar_pranchas(
        arquivo_pertinencias=opcoes.arquivo_pertinencias,
        diretorio_imagens=opcoes.diretorio_imagens,
        diretorio_saida=opcoes.diretorio_saida,
        quantidade_tipicas=opcoes.quantidade_tipicas,
        quantidade_ambiguas=opcoes.quantidade_ambiguas,
    )
    print(f"{len(tabela)} imagens registradas nas pranchas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
