"""Identificação das imagens que mudam de grupo entre sementes do FCM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from clustering_imagens.experimentos import alinhar_pertinencias


def carregar_modelo_com_caminhos(
    caminho: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega centroides, pertinências e caminhos de um modelo salvo."""

    with np.load(caminho) as dados:
        return (
            dados["centroides"].copy(),
            dados["pertinencias"].copy(),
            dados["caminhos"].astype(str),
        )


def identificar_imagens_instaveis(
    diretorio_modelos: Path,
    sementes: Sequence[int] = (42, 123, 2026),
) -> pd.DataFrame:
    """Mede, por imagem, mudanças de grupo após alinhar os centroides."""

    if len(sementes) < 2:
        raise ValueError("Informe ao menos duas sementes para comparar.")
    if len(set(sementes)) != len(sementes):
        raise ValueError("As sementes não podem se repetir.")

    semente_referencia = sementes[0]
    caminho_referencia = (
        diretorio_modelos
        / f"semente_{semente_referencia:04}"
        / "modelo_fcm.npz"
    )
    centroides_referencia, pertinencias_referencia, caminhos = (
        carregar_modelo_com_caminhos(caminho_referencia)
    )
    rotulos_referencia = pertinencias_referencia.argmax(axis=1)
    resultado = pd.DataFrame(
        {
            "caminho_relativo": caminhos,
            "grupo_referencia": rotulos_referencia + 1,
            "pertinencia_principal_referencia": pertinencias_referencia.max(axis=1),
        }
    )

    colunas_mudanca: list[str] = []
    colunas_diferenca: list[str] = []
    for semente in sementes[1:]:
        caminho_modelo = (
            diretorio_modelos / f"semente_{semente:04}" / "modelo_fcm.npz"
        )
        centroides, pertinencias, caminhos_comparados = carregar_modelo_com_caminhos(
            caminho_modelo
        )
        if not np.array_equal(caminhos, caminhos_comparados):
            raise ValueError(
                f"A ordem das imagens da semente {semente} difere da referência."
            )
        _, pertinencias_alinhadas = alinhar_pertinencias(
            centroides_referencia, centroides, pertinencias
        )
        rotulos_alinhados = pertinencias_alinhadas.argmax(axis=1)
        coluna_grupo = f"grupo_semente_{semente:04}"
        coluna_mudanca = f"mudou_grupo_semente_{semente:04}"
        coluna_diferenca = f"diferenca_pertinencias_semente_{semente:04}"
        resultado[coluna_grupo] = rotulos_alinhados + 1
        resultado[coluna_mudanca] = rotulos_alinhados != rotulos_referencia
        resultado[coluna_diferenca] = np.mean(
            np.abs(pertinencias_referencia - pertinencias_alinhadas), axis=1
        )
        colunas_mudanca.append(coluna_mudanca)
        colunas_diferenca.append(coluna_diferenca)

    resultado["mudancas_grupo"] = resultado[colunas_mudanca].sum(axis=1)
    resultado["taxa_mudanca"] = resultado["mudancas_grupo"] / len(colunas_mudanca)
    resultado["diferenca_media_pertinencias"] = resultado[colunas_diferenca].mean(
        axis=1
    )
    return resultado.sort_values(
        ["mudancas_grupo", "diferenca_media_pertinencias"],
        ascending=False,
    ).reset_index(drop=True)


def criar_analisador_argumentos() -> argparse.ArgumentParser:
    """Configura a interface de linha de comando."""

    analisador = argparse.ArgumentParser(
        description="Lista imagens que mudam de grupo entre sementes do FCM."
    )
    analisador.add_argument("--diretorio-modelos", type=Path, required=True)
    analisador.add_argument("--sementes", type=int, nargs="+", default=(42, 123, 2026))
    analisador.add_argument(
        "--arquivo-saida",
        type=Path,
        default=Path("artefatos/experimentos/estabilidade_imagens.csv"),
    )
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa a análise e salva a tabela ordenada por instabilidade."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    tabela = identificar_imagens_instaveis(opcoes.diretorio_modelos, opcoes.sementes)
    opcoes.arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(opcoes.arquivo_saida, index=False)
    quantidade_instaveis = int((tabela["mudancas_grupo"] > 0).sum())
    print(
        f"{quantidade_instaveis} de {len(tabela)} imagens mudaram de grupo; "
        f"tabela salva em {opcoes.arquivo_saida}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
