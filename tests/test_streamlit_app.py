from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest


def preparar_artefatos(tmp_path: Path) -> tuple[Path, Path]:
    diretorio_experimentos = tmp_path / "experimentos"
    diretorio_execucao = diretorio_experimentos / "pca_002" / "semente_0042"
    diretorio_execucao.mkdir(parents=True)
    diretorio_imagens = tmp_path / "imagens"
    diretorio_imagens.mkdir()

    caminhos = [f"flor_{indice}.jpg" for indice in range(4)]
    for indice, caminho in enumerate(caminhos):
        Image.new("RGB", (16, 12), (40 * indice, 80, 120)).save(
            diretorio_imagens / caminho
        )

    resultados = pd.DataFrame(
        {
            "componentes_pca": [2],
            "semente": [42],
            "quantidade_clusters": [2],
            "expoente_fuzzificacao": [1.2],
            "convergiu": [True],
            "indice_xie_beni": [0.8],
            "silhueta_atribuicao_dominante": [0.2],
            "pertinencia_principal_media": [0.75],
            "clusters_dominantes_ocupados": [2],
            "menor_distancia_centroides_quadrada": [0.3],
            "kmeans_silhueta": [0.21],
            "diretorio_execucao": ["pca_002/semente_0042"],
        }
    )
    resultados.to_csv(diretorio_experimentos / "resultados.csv", index=False)
    pd.DataFrame(
        {
            "componentes_pca": [2],
            "concordancia_dominante_media": [0.9],
        }
    ).to_csv(diretorio_experimentos / "resumo_representacoes.csv", index=False)

    projecao = pd.DataFrame(
        {
            "caminho_relativo": caminhos,
            "componente_1": [-1.0, -0.5, 0.5, 1.0],
            "componente_2": [0.1, -0.1, 0.2, -0.2],
            "cluster_principal": [1, 1, 2, 2],
            "pertinencia_principal": [0.9, 0.7, 0.8, 0.6],
            "cluster_secundario": [2, 2, 1, 1],
            "pertinencia_secundaria": [0.1, 0.3, 0.2, 0.4],
            "diferenca_top2": [0.8, 0.4, 0.6, 0.2],
            "cluster_kmeans": [1, 1, 2, 2],
        }
    )
    projecao.to_csv(diretorio_execucao / "projecao_2d.csv", index=False)
    pertinencias = projecao[
        [
            "caminho_relativo",
            "cluster_principal",
            "pertinencia_principal",
            "cluster_secundario",
            "pertinencia_secundaria",
            "diferenca_top2",
            "cluster_kmeans",
        ]
    ].copy()
    pertinencias["pertinencia_cluster_001"] = [0.9, 0.7, 0.2, 0.4]
    pertinencias["pertinencia_cluster_002"] = [0.1, 0.3, 0.8, 0.6]
    pertinencias.to_csv(diretorio_execucao / "pertinencias.csv", index=False)
    (diretorio_execucao / "metricas.json").write_text(
        json.dumps(
            {
                "quantidade_imagens": 4,
                "variancia_explicada_pca": 0.9,
                "pertinencia_principal_media": 0.75,
                "silhueta_atribuicao_dominante": 0.2,
                "convergiu": True,
                "quantidade_clusters": 2,
                "menor_distancia_centroides_quadrada": 0.3,
            }
        ),
        encoding="utf-8",
    )
    return diretorio_experimentos, diretorio_imagens


def test_app_carrega_dashboard_e_galeria(tmp_path: Path) -> None:
    diretorio_experimentos, diretorio_imagens = preparar_artefatos(tmp_path)
    caminho_app = Path(__file__).parents[1] / "streamlit_app.py"

    with patch.dict(
        os.environ,
        {
            "CLUSTERING_ARTEFATOS_DIR": str(diretorio_experimentos),
            "CLUSTERING_IMAGENS_DIR": str(diretorio_imagens),
        },
    ):
        app = AppTest.from_file(caminho_app, default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "Agrupamento visual de flores"
    assert len(app.metric) == 6
    assert len(app.dataframe[0].value) == 1
    assert str(app.selectbox(key="execucao_selecionada").value).startswith("PCA 2")
