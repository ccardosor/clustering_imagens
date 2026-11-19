from pathlib import Path

import numpy as np

from clustering_imagens.estabilidade_imagens import identificar_imagens_instaveis


def salvar_modelo(
    diretorio: Path,
    semente: int,
    centroides: np.ndarray,
    pertinencias: np.ndarray,
) -> None:
    pasta = diretorio / f"semente_{semente:04}"
    pasta.mkdir(parents=True)
    np.savez_compressed(
        pasta / "modelo_fcm.npz",
        centroides=centroides,
        pertinencias=pertinencias,
        caminhos=np.array(["a.jpg", "b.jpg", "c.jpg"]),
    )


def test_identifica_mudanca_apos_alinhar_rotulos(tmp_path: Path) -> None:
    centroides = np.array([[0.0, 0.0], [10.0, 10.0]])
    pertinencias = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    salvar_modelo(tmp_path, 42, centroides, pertinencias)

    # A ordem dos clusters foi trocada; somente b.jpg muda de grupo de fato.
    salvar_modelo(
        tmp_path,
        123,
        centroides[::-1],
        np.array([[0.1, 0.9], [0.8, 0.2], [0.9, 0.1]]),
    )
    salvar_modelo(tmp_path, 2026, centroides, pertinencias)

    resultado = identificar_imagens_instaveis(tmp_path)

    registro_b = resultado.set_index("caminho_relativo").loc["b.jpg"]
    assert registro_b["mudancas_grupo"] == 1
    assert registro_b["taxa_mudanca"] == 0.5
    assert resultado.iloc[0]["caminho_relativo"] == "b.jpg"
