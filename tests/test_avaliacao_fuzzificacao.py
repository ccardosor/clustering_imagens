from __future__ import annotations

import pandas as pd

from clustering_imagens.avaliacao_fuzzificacao import (
    rotulo_expoente,
    resumir_fuzzificacao,
)


def test_rotulo_expoente_e_resumo() -> None:
    assert rotulo_expoente(1.15) == "m_1p150"
    resultados = pd.DataFrame(
        {
            "expoente_fuzzificacao": [1.1, 1.1, 1.2, 1.2],
            "quantidade_clusters": [8, 8, 8, 8],
            "convergiu": [True, True, True, False],
            "clusters_dominantes_ocupados": [8, 8, 8, 7],
            "iteracoes_realizadas": [100, 120, 150, 300],
            "indice_xie_beni": [2.0, 4.0, 6.0, 10.0],
            "silhueta_atribuicao_dominante": [0.1, 0.2, 0.3, 0.4],
            "pertinencia_principal_media": [0.8, 0.7, 0.6, 0.5],
            "menor_distancia_centroides_quadrada": [0.2, 0.1, 0.3, 0.05],
        }
    )
    estabilidade = pd.DataFrame(
        {
            "expoente_fuzzificacao": [1.1, 1.1, 1.2, 1.2],
            "semente_referencia": [42, 42, 42, 42],
            "semente_comparada": [42, 123, 42, 123],
            "concordancia_dominante": [1.0, 0.8, 1.0, 0.6],
            "indice_rand_ajustado": [1.0, 0.7, 1.0, 0.5],
            "diferenca_media_pertinencias": [0.0, 0.1, 0.0, 0.2],
        }
    )

    resumo = resumir_fuzzificacao(resultados, estabilidade, (1.1, 1.2))

    linha = resumo.set_index("expoente_fuzzificacao").loc[1.1]
    assert linha["convergencias"] == 2
    assert linha["xie_beni_mediano"] == 3.0
    assert linha["concordancia_dominante_media"] == 0.8
