from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clustering_imagens.visualizacao import (
    descobrir_fontes_dados,
    filtrar_grupo_visual,
    preparar_distribuicao_grupos,
    resolver_caminho_imagem,
    selecionar_expoente_candidato,
    selecionar_galeria,
    selecionar_galeria_instavel,
    selecionar_quantidade_candidata,
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


def test_selecionar_galeria_instavel_filtra_e_ordena() -> None:
    tabela = pd.DataFrame(
        {
            "caminho_relativo": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "grupo_referencia": [1, 1, 1, 2],
            "mudancas_grupo": [0, 1, 2, 2],
            "diferenca_media_pertinencias": [0.01, 0.30, 0.10, 0.50],
        }
    )

    resultado = selecionar_galeria_instavel(tabela, grupo_referencia=1, limite=2)

    assert resultado["caminho_relativo"].tolist() == ["c.jpg", "b.jpg"]


def test_selecionar_quantidade_candidata_ignora_colapso() -> None:
    resumo = pd.DataFrame(
        {
            "quantidade_clusters": [8, 12],
            "convergencias": [3, 0],
            "execucoes": [3, 3],
            "grupos_ocupados_minimo": [8, 12],
            "menor_distancia_centroides_quadrada": [0.13, 1e-8],
            "xie_beni_mediano": [4.6, 1e6],
        }
    )

    assert selecionar_quantidade_candidata(resumo) == 8


def test_selecionar_expoente_candidato_prioriza_estabilidade_valida() -> None:
    resumo = pd.DataFrame(
        {
            "expoente_fuzzificacao": [1.1, 1.2, 1.25],
            "quantidade_clusters": [8, 8, 8],
            "convergencias": [3, 3, 0],
            "execucoes": [3, 3, 3],
            "grupos_ocupados_minimo": [8, 8, 8],
            "menor_distancia_centroides_quadrada": [0.25, 0.13, 1e-7],
            "concordancia_dominante_media": [0.64, 0.786, 0.99],
            "xie_beni_mediano": [2.45, 4.67, 5e5],
        }
    )

    assert selecionar_expoente_candidato(resumo) == 1.2


def test_descobrir_fontes_dados_encontra_piloto_e_completo(tmp_path: Path) -> None:
    piloto = tmp_path / "experimentos"
    completo = tmp_path / "completo" / "experimentos"
    piloto.mkdir(parents=True)
    completo.mkdir(parents=True)
    (piloto / "resultados.csv").touch()
    (completo / "resultados.csv").touch()

    fontes = descobrir_fontes_dados(piloto)

    assert fontes == {"piloto": piloto, "completo": completo}
    assert descobrir_fontes_dados(completo) == fontes


def test_filtrar_grupo_visual_preserva_tabela_original() -> None:
    tabela = criar_tabela()

    resultado = filtrar_grupo_visual(tabela, 1)

    assert resultado["caminho_relativo"].tolist() == ["a.jpg", "c.jpg"]
    assert len(tabela) == 3
    assert filtrar_grupo_visual(tabela, None).equals(tabela)


def test_preparar_distribuicao_grupos_usa_ordem_numerica_e_percentual() -> None:
    resultado = preparar_distribuicao_grupos(criar_tabela(), 3)

    assert resultado["grupo"].tolist() == ["Grupo 1", "Grupo 2", "Grupo 3"]
    assert resultado["imagens"].tolist() == [2, 1, 0]
    assert resultado["percentual"].tolist() == pytest.approx([2 / 3, 1 / 3, 0])
    assert resultado["rotulo_quantidade"].tolist() == ["2", "1", "0"]
