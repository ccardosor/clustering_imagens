"""Preparação das características e execução do piloto Fuzzy C-Means."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


@dataclass(frozen=True, slots=True)
class ResultadoFCM:
    """Resultado numérico do algoritmo Fuzzy C-Means."""

    centroides: np.ndarray
    pertinencias: np.ndarray
    objetivo: float
    iteracoes: int
    convergiu: bool


@dataclass(frozen=True, slots=True)
class MetricasPiloto:
    """Configuração e métricas usadas para auditar o piloto."""

    quantidade_imagens: int
    dimensao_entrada: int
    dimensao_representacao: int
    componentes_pca: int | None
    variancia_explicada_pca: float | None
    quantidade_clusters: int
    expoente_fuzzificacao: float
    inicializacao: str
    semente: int
    semente_pca: int
    tolerancia: float
    maximo_iteracoes: int
    iteracoes_realizadas: int
    convergiu: bool
    funcao_objetivo: float
    indice_xie_beni: float
    silhueta_atribuicao_dominante: float | None
    menor_soma_pertinencias: float
    maior_soma_pertinencias: float
    pertinencia_principal_media: float
    clusters_dominantes_ocupados: int
    menor_distancia_centroides_quadrada: float
    kmeans_inercia: float
    kmeans_silhueta: float | None
    kmeans_menor_cluster: int
    kmeans_maior_cluster: int


def preparar_representacao(
    caracteristicas: np.ndarray,
    componentes_pca: int | None,
    semente: int,
) -> tuple[np.ndarray, float | None]:
    """Aplica PCA opcional e normalização L2, conforme o plano do projeto."""

    if caracteristicas.ndim != 2 or caracteristicas.shape[0] < 2:
        raise ValueError("A matriz de características deve ser bidimensional e ter 2+ linhas.")
    if not np.isfinite(caracteristicas).all():
        raise ValueError("A matriz de características contém valores não finitos.")

    representacao = caracteristicas.astype(np.float64, copy=False)
    variancia_explicada = None
    if componentes_pca is not None:
        limite = min(representacao.shape)
        if componentes_pca <= 0 or componentes_pca >= limite:
            raise ValueError(
                f"Os componentes de PCA devem estar entre 1 e {limite - 1}."
            )
        pca = PCA(
            n_components=componentes_pca,
            svd_solver="randomized",
            random_state=semente,
        )
        representacao = pca.fit_transform(representacao)
        variancia_explicada = float(pca.explained_variance_ratio_.sum())

    representacao = normalize(representacao, norm="l2", axis=1)
    if np.any(np.linalg.norm(representacao, axis=1) == 0):
        raise ValueError("A representação contém ao menos um vetor nulo.")
    return representacao, variancia_explicada


def calcular_distancias_quadradas(
    dados: np.ndarray, centroides: np.ndarray
) -> np.ndarray:
    """Calcula distâncias euclidianas quadradas sem criar tensor 3D."""

    distancias = (
        np.sum(dados * dados, axis=1, keepdims=True)
        + np.sum(centroides * centroides, axis=1)[None, :]
        - 2 * dados @ centroides.T
    )
    return np.maximum(distancias, 0.0)


def atualizar_pertinencias(
    distancias_quadradas: np.ndarray, expoente_fuzzificacao: float
) -> np.ndarray:
    """Atualiza pertinências, incluindo o caso de ponto sobre um centroide."""

    pertinencias = np.zeros_like(distancias_quadradas)
    distancias_nulas = distancias_quadradas <= np.finfo(float).eps
    linhas_com_nulos = distancias_nulas.any(axis=1)

    if linhas_com_nulos.any():
        contagens = distancias_nulas[linhas_com_nulos].sum(axis=1, keepdims=True)
        pertinencias[linhas_com_nulos] = distancias_nulas[linhas_com_nulos] / contagens

    linhas_regulares = ~linhas_com_nulos
    if linhas_regulares.any():
        potencia = -1.0 / (expoente_fuzzificacao - 1.0)
        inversos = distancias_quadradas[linhas_regulares] ** potencia
        pertinencias[linhas_regulares] = inversos / inversos.sum(axis=1, keepdims=True)
    return pertinencias


def fuzzy_c_means(
    dados: np.ndarray,
    quantidade_clusters: int,
    expoente_fuzzificacao: float = 2.0,
    semente: int = 42,
    maximo_iteracoes: int = 300,
    tolerancia: float = 1e-5,
    inicializacao: str = "kmeans++",
) -> ResultadoFCM:
    """Agrupa vetores com FCM e devolve todos os valores auditáveis."""

    if dados.ndim != 2 or not np.isfinite(dados).all():
        raise ValueError("Os dados devem formar uma matriz finita bidimensional.")
    if not 2 <= quantidade_clusters < dados.shape[0]:
        raise ValueError("A quantidade de clusters deve estar entre 2 e n_amostras - 1.")
    if expoente_fuzzificacao <= 1:
        raise ValueError("O expoente de fuzzificação deve ser maior que 1.")
    if maximo_iteracoes <= 0 or tolerancia <= 0:
        raise ValueError("Iterações e tolerância devem ser maiores que zero.")
    if inicializacao not in {"kmeans++", "pertinencias_aleatorias"}:
        raise ValueError(
            "A inicialização deve ser 'kmeans++' ou 'pertinencias_aleatorias'."
        )

    if inicializacao == "kmeans++":
        centroides_iniciais, _ = kmeans_plusplus(
            dados, n_clusters=quantidade_clusters, random_state=semente
        )
        pertinencias = atualizar_pertinencias(
            calcular_distancias_quadradas(dados, centroides_iniciais),
            expoente_fuzzificacao,
        )
    else:
        gerador = np.random.default_rng(semente)
        pertinencias = gerador.dirichlet(
            np.ones(quantidade_clusters), size=dados.shape[0]
        )
    centroides_anteriores: np.ndarray | None = None
    convergiu = False

    for iteracao in range(1, maximo_iteracoes + 1):
        pesos = pertinencias**expoente_fuzzificacao
        centroides = (pesos.T @ dados) / pesos.sum(axis=0)[:, None]
        distancias = calcular_distancias_quadradas(dados, centroides)
        pertinencias = atualizar_pertinencias(distancias, expoente_fuzzificacao)

        if centroides_anteriores is not None:
            deslocamento = np.max(np.linalg.norm(centroides - centroides_anteriores, axis=1))
            if deslocamento <= tolerancia:
                convergiu = True
                break
        centroides_anteriores = centroides.copy()

    distancias = calcular_distancias_quadradas(dados, centroides)
    objetivo = float(np.sum((pertinencias**expoente_fuzzificacao) * distancias))
    return ResultadoFCM(
        centroides=centroides,
        pertinencias=pertinencias,
        objetivo=objetivo,
        iteracoes=iteracao,
        convergiu=convergiu,
    )


def calcular_xie_beni(
    resultado: ResultadoFCM, dados: np.ndarray, expoente_fuzzificacao: float
) -> float:
    """Calcula o índice Xie–Beni; valores menores indicam melhor separação."""

    distancias_centroides = calcular_distancias_quadradas(
        resultado.centroides, resultado.centroides
    )
    np.fill_diagonal(distancias_centroides, np.inf)
    menor_separacao = float(np.min(distancias_centroides))
    if menor_separacao <= np.finfo(float).eps:
        return float("inf")

    distancias = calcular_distancias_quadradas(dados, resultado.centroides)
    numerador = np.sum(
        (resultado.pertinencias**expoente_fuzzificacao) * distancias
    )
    return float(numerador / (dados.shape[0] * menor_separacao))


def exportar_pertinencias(
    destino: Path,
    caminhos: Sequence[str],
    pertinencias: np.ndarray,
    rotulos_kmeans: np.ndarray | None = None,
) -> None:
    """Exporta a matriz completa e um resumo dos dois grupos mais fortes."""

    if len(caminhos) != pertinencias.shape[0]:
        raise ValueError("Caminhos e pertinências têm quantidades diferentes.")
    if rotulos_kmeans is not None and len(rotulos_kmeans) != len(caminhos):
        raise ValueError("Caminhos e rótulos do K-Means têm quantidades diferentes.")

    destino.parent.mkdir(parents=True, exist_ok=True)
    campos_clusters = [f"pertinencia_cluster_{indice:03}" for indice in range(1, pertinencias.shape[1] + 1)]
    campos = [
        "caminho_relativo",
        "cluster_principal",
        "pertinencia_principal",
        "cluster_secundario",
        "pertinencia_secundaria",
        "diferenca_top2",
        *campos_clusters,
    ]
    if rotulos_kmeans is not None:
        campos.append("cluster_kmeans")
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for indice_linha, (caminho, linha) in enumerate(
            zip(caminhos, pertinencias, strict=True)
        ):
            ordem = np.argsort(linha)[::-1]
            principal, secundario = int(ordem[0]), int(ordem[1])
            registro = {
                "caminho_relativo": caminho,
                "cluster_principal": principal + 1,
                "pertinencia_principal": linha[principal],
                "cluster_secundario": secundario + 1,
                "pertinencia_secundaria": linha[secundario],
                "diferenca_top2": linha[principal] - linha[secundario],
            }
            registro.update(dict(zip(campos_clusters, linha, strict=True)))
            if rotulos_kmeans is not None:
                registro["cluster_kmeans"] = int(rotulos_kmeans[indice_linha]) + 1
            escritor.writerow(registro)


def exportar_projecao_2d(
    destino: Path,
    caminhos: Sequence[str],
    representacao: np.ndarray,
    pertinencias: np.ndarray,
    rotulos_kmeans: np.ndarray,
    semente_pca: int,
) -> None:
    """Exporta uma projeção bidimensional para a visualização interativa."""

    if representacao.shape[0] != len(caminhos):
        raise ValueError("Caminhos e representação têm quantidades diferentes.")
    if representacao.shape[1] == 2:
        projecao = representacao
    else:
        projecao = PCA(
            n_components=2,
            svd_solver="randomized",
            random_state=semente_pca,
        ).fit_transform(representacao)

    campos = [
        "caminho_relativo",
        "componente_1",
        "componente_2",
        "cluster_principal",
        "pertinencia_principal",
        "cluster_secundario",
        "pertinencia_secundaria",
        "diferenca_top2",
        "cluster_kmeans",
    ]
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for indice, caminho in enumerate(caminhos):
            ordem = np.argsort(pertinencias[indice])[::-1]
            principal, secundario = int(ordem[0]), int(ordem[1])
            escritor.writerow(
                {
                    "caminho_relativo": caminho,
                    "componente_1": projecao[indice, 0],
                    "componente_2": projecao[indice, 1],
                    "cluster_principal": principal + 1,
                    "pertinencia_principal": pertinencias[indice, principal],
                    "cluster_secundario": secundario + 1,
                    "pertinencia_secundaria": pertinencias[indice, secundario],
                    "diferenca_top2": (
                        pertinencias[indice, principal]
                        - pertinencias[indice, secundario]
                    ),
                    "cluster_kmeans": int(rotulos_kmeans[indice]) + 1,
                }
            )


def executar_piloto(
    arquivo_caracteristicas: Path,
    diretorio_saida: Path,
    componentes_pca: int | None = 64,
    quantidade_clusters: int = 10,
    expoente_fuzzificacao: float = 1.2,
    semente: int = 42,
    semente_pca: int = 42,
    maximo_iteracoes: int = 300,
    tolerancia: float = 1e-5,
    inicializacao: str = "kmeans++",
) -> MetricasPiloto:
    """Executa preparação, FCM, avaliação e exportação do piloto."""

    with np.load(arquivo_caracteristicas) as dados:
        caracteristicas = dados["caracteristicas"]
        caminhos = dados["caminhos"].astype(str).tolist()
    if caracteristicas.shape[0] != len(caminhos):
        raise ValueError("Características e caminhos têm quantidades diferentes.")

    representacao, variancia = preparar_representacao(
        caracteristicas, componentes_pca, semente_pca
    )
    resultado = fuzzy_c_means(
        representacao,
        quantidade_clusters=quantidade_clusters,
        expoente_fuzzificacao=expoente_fuzzificacao,
        semente=semente,
        maximo_iteracoes=maximo_iteracoes,
        tolerancia=tolerancia,
        inicializacao=inicializacao,
    )
    somas = resultado.pertinencias.sum(axis=1)
    if not np.allclose(somas, 1.0, atol=1e-8):
        raise RuntimeError("As pertinências não somam 1 para todas as imagens.")

    rotulos = resultado.pertinencias.argmax(axis=1)
    quantidade_rotulos = np.unique(rotulos).size
    silhueta = None
    if 1 < quantidade_rotulos < representacao.shape[0]:
        silhueta = float(silhouette_score(representacao, rotulos))
    distancias_entre_centroides = calcular_distancias_quadradas(
        resultado.centroides, resultado.centroides
    )
    np.fill_diagonal(distancias_entre_centroides, np.inf)

    kmeans = KMeans(
        n_clusters=quantidade_clusters,
        init="k-means++",
        n_init=20,
        random_state=semente,
    ).fit(representacao)
    contagens_kmeans = np.bincount(
        kmeans.labels_, minlength=quantidade_clusters
    )
    quantidade_rotulos_kmeans = np.unique(kmeans.labels_).size
    silhueta_kmeans = None
    if 1 < quantidade_rotulos_kmeans < representacao.shape[0]:
        silhueta_kmeans = float(silhouette_score(representacao, kmeans.labels_))

    metricas = MetricasPiloto(
        quantidade_imagens=representacao.shape[0],
        dimensao_entrada=caracteristicas.shape[1],
        dimensao_representacao=representacao.shape[1],
        componentes_pca=componentes_pca,
        variancia_explicada_pca=variancia,
        quantidade_clusters=quantidade_clusters,
        expoente_fuzzificacao=expoente_fuzzificacao,
        inicializacao=inicializacao,
        semente=semente,
        semente_pca=semente_pca,
        tolerancia=tolerancia,
        maximo_iteracoes=maximo_iteracoes,
        iteracoes_realizadas=resultado.iteracoes,
        convergiu=resultado.convergiu,
        funcao_objetivo=resultado.objetivo,
        indice_xie_beni=calcular_xie_beni(
            resultado, representacao, expoente_fuzzificacao
        ),
        silhueta_atribuicao_dominante=silhueta,
        menor_soma_pertinencias=float(somas.min()),
        maior_soma_pertinencias=float(somas.max()),
        pertinencia_principal_media=float(resultado.pertinencias.max(axis=1).mean()),
        clusters_dominantes_ocupados=int(quantidade_rotulos),
        menor_distancia_centroides_quadrada=float(
            np.min(distancias_entre_centroides)
        ),
        kmeans_inercia=float(kmeans.inertia_),
        kmeans_silhueta=silhueta_kmeans,
        kmeans_menor_cluster=min(map(int, contagens_kmeans)),
        kmeans_maior_cluster=max(map(int, contagens_kmeans)),
    )

    diretorio_saida.mkdir(parents=True, exist_ok=True)
    exportar_pertinencias(
        diretorio_saida / "pertinencias.csv",
        caminhos,
        resultado.pertinencias,
        kmeans.labels_,
    )
    exportar_projecao_2d(
        diretorio_saida / "projecao_2d.csv",
        caminhos,
        representacao,
        resultado.pertinencias,
        kmeans.labels_,
        semente_pca,
    )
    np.savez_compressed(
        diretorio_saida / "modelo_fcm.npz",
        centroides=resultado.centroides,
        pertinencias=resultado.pertinencias,
        caminhos=np.asarray(caminhos),
    )
    with (diretorio_saida / "metricas.json").open("w", encoding="utf-8") as arquivo:
        json.dump(asdict(metricas), arquivo, ensure_ascii=False, indent=2)
        arquivo.write("\n")
    return metricas


def criar_analisador_argumentos() -> argparse.ArgumentParser:
    """Configura a interface de linha de comando."""

    analisador = argparse.ArgumentParser(
        description="Executa o piloto Fuzzy C-Means sobre características extraídas."
    )
    analisador.add_argument("--arquivo-caracteristicas", type=Path, required=True)
    analisador.add_argument(
        "--diretorio-saida", type=Path, default=Path("artefatos/piloto/fcm")
    )
    analisador.add_argument("--componentes-pca", type=int, default=64)
    analisador.add_argument("--quantidade-clusters", type=int, default=10)
    analisador.add_argument("--expoente-fuzzificacao", type=float, default=1.2)
    analisador.add_argument("--semente", type=int, default=42)
    analisador.add_argument(
        "--semente-pca",
        type=int,
        default=42,
        help="semente fixa da representação PCA (padrão: 42)",
    )
    analisador.add_argument("--maximo-iteracoes", type=int, default=300)
    analisador.add_argument("--tolerancia", type=float, default=1e-5)
    analisador.add_argument(
        "--inicializacao",
        choices=("kmeans++", "pertinencias_aleatorias"),
        default="kmeans++",
    )
    return analisador


def principal(argumentos: Sequence[str] | None = None) -> int:
    """Executa o piloto pela linha de comando."""

    opcoes = criar_analisador_argumentos().parse_args(argumentos)
    componentes_pca = opcoes.componentes_pca or None
    metricas = executar_piloto(
        arquivo_caracteristicas=opcoes.arquivo_caracteristicas,
        diretorio_saida=opcoes.diretorio_saida,
        componentes_pca=componentes_pca,
        quantidade_clusters=opcoes.quantidade_clusters,
        expoente_fuzzificacao=opcoes.expoente_fuzzificacao,
        semente=opcoes.semente,
        semente_pca=opcoes.semente_pca,
        maximo_iteracoes=opcoes.maximo_iteracoes,
        tolerancia=opcoes.tolerancia,
        inicializacao=opcoes.inicializacao,
    )
    print(json.dumps(asdict(metricas), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
