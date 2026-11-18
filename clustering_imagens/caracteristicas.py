"""Extração reproduzível de características visuais com ResNet-50."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, cast

if TYPE_CHECKING:
    from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class MetadadosExtracao:
    """Proveniência da matriz de características."""

    arquivo_amostra: str
    sha256_amostra: str
    diretorio_imagens: str
    arquivo_caracteristicas: str
    diretorio_cache_modelos: str
    quantidade_imagens: int
    dimensao_caracteristicas: int
    pesos: str
    preprocessamento: str
    dispositivo: str
    tamanho_lote: int
    versao_torch: str
    versao_torchvision: str


class ConjuntoImagens:
    """Dataset simples e serializável para carregadores com múltiplos processos."""

    def __init__(
        self, diretorio_imagens: Path, caminhos: Sequence[str], transformacao
    ) -> None:
        self.diretorio_imagens = diretorio_imagens
        self.caminhos = caminhos
        self.transformacao = transformacao

    def __len__(self) -> int:
        return len(self.caminhos)

    def __getitem__(self, indice: int):
        from PIL import Image

        with Image.open(self.diretorio_imagens / self.caminhos[indice]) as imagem:
            return self.transformacao(imagem.convert("RGB"))


def calcular_sha256(caminho: Path) -> str:
    """Calcula o SHA-256 de um arquivo sem carregá-lo inteiro na memória."""

    resumo = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def ler_caminhos_amostra(arquivo_amostra: Path) -> list[str]:
    """Lê e valida os caminhos relativos na ordem registrada pelo inventário."""

    with arquivo_amostra.open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))

    if not linhas:
        raise ValueError(f"A amostra está vazia: {arquivo_amostra}")
    if "caminho_relativo" not in linhas[0]:
        raise ValueError("A amostra não contém a coluna 'caminho_relativo'.")

    caminhos = [linha["caminho_relativo"].strip() for linha in linhas]
    if any(not caminho for caminho in caminhos):
        raise ValueError("A amostra contém um caminho relativo vazio.")
    if len(caminhos) != len(set(caminhos)):
        raise ValueError("A amostra contém caminhos duplicados.")
    return caminhos


def validar_caminhos_imagens(diretorio_imagens: Path, caminhos: Sequence[str]) -> None:
    """Falha cedo quando uma imagem da amostra não está disponível."""

    ausentes = [caminho for caminho in caminhos if not (diretorio_imagens / caminho).is_file()]
    if ausentes:
        exemplos = ", ".join(ausentes[:3])
        raise FileNotFoundError(
            f"{len(ausentes)} imagem(ns) da amostra não encontrada(s). Exemplos: {exemplos}"
        )


def extrair_caracteristicas(
    diretorio_imagens: Path,
    arquivo_amostra: Path,
    arquivo_saida: Path,
    tamanho_lote: int = 16,
    dispositivo: str = "auto",
    trabalhadores: int = 0,
    diretorio_cache_modelos: Path = Path(".cache/torch"),
) -> MetadadosExtracao:
    """Extrai o vetor anterior à classificação para cada imagem da amostra."""

    if tamanho_lote <= 0:
        raise ValueError("O tamanho do lote deve ser maior que zero.")
    if trabalhadores < 0:
        raise ValueError("A quantidade de trabalhadores não pode ser negativa.")

    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet50_Weights, resnet50

    diretorio_imagens = diretorio_imagens.resolve()
    arquivo_amostra = arquivo_amostra.resolve()
    arquivo_saida = arquivo_saida.resolve()
    diretorio_cache_modelos = diretorio_cache_modelos.resolve()
    diretorio_cache_modelos.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(diretorio_cache_modelos))
    caminhos = ler_caminhos_amostra(arquivo_amostra)
    validar_caminhos_imagens(diretorio_imagens, caminhos)

    if dispositivo == "auto":
        dispositivo_efetivo = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dispositivo_efetivo = dispositivo
    if dispositivo_efetivo.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível.")

    pesos = ResNet50_Weights.IMAGENET1K_V2
    transformacao = pesos.transforms()

    conjunto = cast(
        "Dataset[object]",
        ConjuntoImagens(diretorio_imagens, caminhos, transformacao),
    )
    carregador = DataLoader(
        conjunto,
        batch_size=tamanho_lote,
        shuffle=False,
        num_workers=trabalhadores,
        pin_memory=dispositivo_efetivo.startswith("cuda"),
    )
    modelo = resnet50(weights=pesos)
    modelo.fc = torch.nn.Identity()
    modelo.eval().to(dispositivo_efetivo)

    lotes: list[np.ndarray] = []
    with torch.inference_mode():
        for lote in carregador:
            vetores = modelo(lote.to(dispositivo_efetivo, non_blocking=True))
            lotes.append(vetores.cpu().numpy().astype(np.float32, copy=False))

    matriz = np.concatenate(lotes, axis=0)
    if matriz.shape[0] != len(caminhos):
        raise RuntimeError("A quantidade de vetores não corresponde à quantidade de imagens.")
    if not np.isfinite(matriz).all():
        raise RuntimeError("A matriz de características contém valores não finitos.")

    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arquivo_saida,
        caracteristicas=matriz,
        caminhos=np.asarray(caminhos),
    )

    metadados = MetadadosExtracao(
        arquivo_amostra=str(arquivo_amostra),
        sha256_amostra=calcular_sha256(arquivo_amostra),
        diretorio_imagens=str(diretorio_imagens),
        arquivo_caracteristicas=str(arquivo_saida),
        diretorio_cache_modelos=str(diretorio_cache_modelos),
        quantidade_imagens=matriz.shape[0],
        dimensao_caracteristicas=matriz.shape[1],
        pesos="ResNet50_Weights.IMAGENET1K_V2",
        preprocessamento=repr(transformacao),
        dispositivo=dispositivo_efetivo,
        tamanho_lote=tamanho_lote,
        versao_torch=torch.__version__,
        versao_torchvision=torchvision.__version__,
    )
    destino_metadados = arquivo_saida.with_suffix(".json")
    with destino_metadados.open("w", encoding="utf-8") as arquivo:
        json.dump(asdict(metadados), arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    return metadados


def criar_analisador_argumentos() -> argparse.ArgumentParser:
    """Configura a interface de linha de comando."""

    analisador = argparse.ArgumentParser(
        description="Extrai características ResNet-50 das imagens da amostra."
    )
    analisador.add_argument("--diretorio-imagens", type=Path, required=True)
    analisador.add_argument("--arquivo-amostra", type=Path, required=True)
    analisador.add_argument(
        "--arquivo-saida",
        type=Path,
        default=Path("artefatos/piloto/caracteristicas.npz"),
    )
    analisador.add_argument("--tamanho-lote", type=int, default=16)
    analisador.add_argument("--dispositivo", default="auto")
    analisador.add_argument("--trabalhadores", type=int, default=0)
    analisador.add_argument(
        "--diretorio-cache-modelos",
        type=Path,
        default=Path(".cache/torch"),
        help="cache local dos pesos do modelo (padrão: .cache/torch)",
    )
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa a extração pela linha de comando."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    metadados = extrair_caracteristicas(
        diretorio_imagens=opcoes.diretorio_imagens,
        arquivo_amostra=opcoes.arquivo_amostra,
        arquivo_saida=opcoes.arquivo_saida,
        tamanho_lote=opcoes.tamanho_lote,
        dispositivo=opcoes.dispositivo,
        trabalhadores=opcoes.trabalhadores,
        diretorio_cache_modelos=opcoes.diretorio_cache_modelos,
    )
    print(json.dumps(asdict(metadados), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
