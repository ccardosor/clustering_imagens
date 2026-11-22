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


@dataclass(frozen=True, slots=True)
class EstadoExtracao:
    """Estado mínimo necessário para retomar uma extração interrompida."""

    sha256_amostra: str
    diretorio_imagens: str
    quantidade_total: int
    quantidade_processada: int
    dimensao_caracteristicas: int


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


def caminhos_extracao_parcial(arquivo_saida: Path) -> tuple[Path, Path]:
    """Retorna os arquivos usados para a matriz parcial e seu estado."""

    prefixo = arquivo_saida.with_suffix("")
    return (
        prefixo.with_name(f"{prefixo.name}.parcial.npy"),
        prefixo.with_name(f"{prefixo.name}.parcial.json"),
    )


def salvar_estado_extracao(caminho: Path, estado: EstadoExtracao) -> None:
    """Grava o progresso de forma atômica depois de persistir cada lote."""

    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(f"{caminho.suffix}.tmp")
    temporario.write_text(
        json.dumps(asdict(estado), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def carregar_estado_extracao(
    caminho_matriz: Path,
    caminho_estado: Path,
    sha256_amostra: str,
    diretorio_imagens: Path,
    quantidade_total: int,
) -> EstadoExtracao | None:
    """Valida um checkpoint existente antes de permitir sua retomada."""

    if not caminho_matriz.exists() and not caminho_estado.exists():
        return None
    if not caminho_matriz.is_file() or not caminho_estado.is_file():
        raise RuntimeError(
            "Checkpoint de extração incompleto. Remova os arquivos '.parcial' "
            "ou restaure o par .npy/.json antes de continuar."
        )
    try:
        conteudo = json.loads(caminho_estado.read_text(encoding="utf-8"))
        estado = EstadoExtracao(**conteudo)
    except (OSError, json.JSONDecodeError, TypeError) as erro:
        raise RuntimeError(f"Checkpoint de extração inválido: {erro}") from erro

    esperado = (sha256_amostra, str(diretorio_imagens), quantidade_total)
    observado = (
        estado.sha256_amostra,
        estado.diretorio_imagens,
        estado.quantidade_total,
    )
    if observado != esperado:
        raise RuntimeError(
            "O checkpoint pertence a outra amostra ou diretório de imagens."
        )
    if not 0 < estado.quantidade_processada <= quantidade_total:
        raise RuntimeError("O checkpoint informa uma quantidade processada inválida.")
    if estado.dimensao_caracteristicas <= 0:
        raise RuntimeError("O checkpoint informa uma dimensão inválida.")
    return estado


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
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
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

    sha256_amostra = calcular_sha256(arquivo_amostra)
    caminho_matriz_parcial, caminho_estado_parcial = caminhos_extracao_parcial(
        arquivo_saida
    )
    estado = carregar_estado_extracao(
        caminho_matriz_parcial,
        caminho_estado_parcial,
        sha256_amostra,
        diretorio_imagens,
        len(caminhos),
    )
    quantidade_processada = estado.quantidade_processada if estado else 0
    caminhos_pendentes = caminhos[quantidade_processada:]
    conjunto = cast(
        "Dataset[object]",
        ConjuntoImagens(diretorio_imagens, caminhos_pendentes, transformacao),
    )
    carregador = DataLoader(
        conjunto,
        batch_size=tamanho_lote,
        shuffle=False,
        num_workers=trabalhadores,
        pin_memory=dispositivo_efetivo.startswith("cuda"),
    )
    matriz = None
    if estado is not None:
        matriz = np.lib.format.open_memmap(
            caminho_matriz_parcial,
            mode="r+",
            dtype=np.float32,
            shape=(len(caminhos), estado.dimensao_caracteristicas),
        )
        if matriz.shape != (len(caminhos), estado.dimensao_caracteristicas):
            raise RuntimeError("A matriz parcial não corresponde ao checkpoint.")

    if caminhos_pendentes:
        modelo = resnet50(weights=pesos)
        modelo.fc = torch.nn.Identity()
        modelo.eval().to(dispositivo_efetivo)

        with torch.inference_mode():
            for indice_lote, lote in enumerate(carregador, start=1):
                vetores = modelo(lote.to(dispositivo_efetivo, non_blocking=True))
                vetores_numpy = vetores.cpu().numpy().astype(np.float32, copy=False)
                if matriz is None:
                    matriz = np.lib.format.open_memmap(
                        caminho_matriz_parcial,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(caminhos), vetores_numpy.shape[1]),
                    )
                limite = quantidade_processada + vetores_numpy.shape[0]
                matriz[quantidade_processada:limite] = vetores_numpy
                matriz.flush()
                quantidade_processada = limite
                salvar_estado_extracao(
                    caminho_estado_parcial,
                    EstadoExtracao(
                        sha256_amostra=sha256_amostra,
                        diretorio_imagens=str(diretorio_imagens),
                        quantidade_total=len(caminhos),
                        quantidade_processada=quantidade_processada,
                        dimensao_caracteristicas=matriz.shape[1],
                    ),
                )
                if indice_lote % 10 == 0 or quantidade_processada == len(caminhos):
                    print(
                        f"Extração: {quantidade_processada}/{len(caminhos)} imagens.",
                        flush=True,
                    )

    if matriz is None:
        raise RuntimeError("A extração não produziu nem encontrou uma matriz parcial.")
    if matriz.shape[0] != len(caminhos):
        raise RuntimeError("A quantidade de vetores não corresponde à quantidade de imagens.")
    if not np.isfinite(matriz).all():
        raise RuntimeError("A matriz de características contém valores não finitos.")

    dimensao_caracteristicas = matriz.shape[1]
    metadados = MetadadosExtracao(
        arquivo_amostra=str(arquivo_amostra),
        sha256_amostra=sha256_amostra,
        diretorio_imagens=str(diretorio_imagens),
        arquivo_caracteristicas=str(arquivo_saida),
        diretorio_cache_modelos=str(diretorio_cache_modelos),
        quantidade_imagens=len(caminhos),
        dimensao_caracteristicas=dimensao_caracteristicas,
        pesos="ResNet50_Weights.IMAGENET1K_V2",
        preprocessamento=repr(transformacao),
        dispositivo=dispositivo_efetivo,
        tamanho_lote=tamanho_lote,
        versao_torch=torch.__version__,
        versao_torchvision=torchvision.__version__,
    )
    destino_metadados = arquivo_saida.with_suffix(".json")
    saida_temporaria = arquivo_saida.with_suffix(f"{arquivo_saida.suffix}.tmp")
    metadados_temporarios = destino_metadados.with_suffix(
        f"{destino_metadados.suffix}.tmp"
    )
    with saida_temporaria.open("wb") as arquivo:
        np.savez_compressed(
            arquivo,
            caracteristicas=matriz,
            caminhos=np.asarray(caminhos),
        )
    metadados_temporarios.write_text(
        json.dumps(asdict(metadados), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    saida_temporaria.replace(arquivo_saida)
    metadados_temporarios.replace(destino_metadados)
    matriz.flush()
    del matriz
    caminho_matriz_parcial.unlink()
    caminho_estado_parcial.unlink()
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
