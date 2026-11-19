from __future__ import annotations

import pandas as pd

from clustering_imagens.avaliacao_clusters import resumir_avaliacao


def test_resume_metricas_por_quantidade_de_clusters() -> None:
    resultados = pd.DataFrame(
        {
            "quantidade_clusters": [8, 8, 10, 10],
            "convergiu": [True, False, True, True],
            "clusters_dominantes_ocupados": [8, 8, 10, 9],
            "indice_xie_beni": [2.0, 4.0, 6.0, 10.0],
            "silhueta_atribuicao_dominante": [0.1, 0.2, 0.3, 0.4],
            "kmeans_silhueta": [0.2, 0.3, 0.4, 0.5],
            "pertinencia_principal_media": [0.5, 0.6, 0.7, 0.8],
            "menor_distancia_centroides_quadrada": [0.2, 0.1, 0.3, 0.05],
        }
    )
    estabilidade = pd.DataFrame(
        {
            "quantidade_clusters": [8, 8, 10, 10],
            "semente_referencia": [42, 42, 42, 42],
            "semente_comparada": [42, 123, 42, 123],
            "concordancia_dominante": [1.0, 0.8, 1.0, 0.6],
            "indice_rand_ajustado": [1.0, 0.7, 1.0, 0.5],
            "diferenca_media_pertinencias": [0.0, 0.1, 0.0, 0.2],
        }
    )

    resumo = resumir_avaliacao(resultados, estabilidade, (8, 10))

    linha_8 = resumo.set_index("quantidade_clusters").loc[8]
    assert linha_8["convergencias"] == 1
    assert linha_8["xie_beni_mediano"] == 3.0
    assert linha_8["concordancia_dominante_media"] == 0.8
    assert linha_8["grupos_ocupados_minimo"] == 8
