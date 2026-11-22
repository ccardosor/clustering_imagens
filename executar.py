"""Prepara o ambiente e executa as etapas configuradas do projeto."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib
import venv

RAIZ = Path(__file__).resolve().parent

MODULOS_POR_PERFIL = {
    "base": ("numpy", "pandas", "PIL", "scipy", "sklearn", "streamlit"),
    "extracao": (
        "numpy", "pandas", "PIL", "scipy", "sklearn", "streamlit",
        "torch", "torchvision",
    ),
    "dev": (
        "numpy", "pandas", "PIL", "scipy", "sklearn", "streamlit",
        "pytest", "ty",
    ),
}


def executar(comando: list[str], simular: bool, ambiente=None) -> None:
    print(shlex.join(comando), flush=True)
    if not simular:
        subprocess.run(comando, cwd=RAIZ, env=ambiente, check=True)


def dependencias_disponiveis(python: Path, perfil: str) -> bool:
    """Verifica os módulos do perfil sem depender de marcadores locais."""

    modulos = MODULOS_POR_PERFIL[perfil]
    codigo = (
        "import importlib.util,sys; "
        f"m={modulos!r}; "
        "sys.exit(0 if all(importlib.util.find_spec(x) for x in m) else 1)"
    )
    verificacao = subprocess.run(
        [str(python), "-c", codigo],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    return verificacao.returncode == 0


def principal() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument(
        "acao", nargs="?", default="pipeline",
        choices=["pipeline", "inventario", "extracao", "piloto",
                 "experimentos", "avaliacao-clusters", "dashboard",
                 "avaliacao-fuzzificacao", "instalar", "testes"],
    )
    analisador.add_argument("--config", type=Path, default=RAIZ / "configuracao.toml")
    analisador.add_argument("--simular", action="store_true",
                            help="mostra os comandos sem instalar ou executar")
    analisador.add_argument("--reinstalar", action="store_true",
                            help="atualiza a instalação do ambiente local")
    opcoes = analisador.parse_args()
    try:
        arquivo = opcoes.config.resolve()
        with arquivo.open("rb") as entrada:
            config = tomllib.load(entrada)
        base = arquivo.parent
        imagens = (base / config["projeto"]["diretorio_imagens"]).resolve()
        artefatos = (base / config["projeto"]["diretorio_artefatos"]).resolve()
        etapas = (
            ["inventario", "extracao", "piloto", "experimentos",
             "avaliacao-clusters", "avaliacao-fuzzificacao"]
            if opcoes.acao == "pipeline" else [opcoes.acao]
        )
        amostra = artefatos / "inventario" / "amostra_piloto.csv"
        caracteristicas = artefatos / "piloto" / "caracteristicas.npz"
        argumentos = {
            "inventario": ["--diretorio-imagens", str(imagens),
                           "--diretorio-saida", str(amostra.parent)],
            "extracao": ["--diretorio-imagens", str(imagens),
                         "--arquivo-amostra", str(amostra),
                         "--arquivo-saida", str(caracteristicas)],
            "piloto": ["--arquivo-caracteristicas", str(caracteristicas),
                       "--diretorio-saida", str(artefatos / "piloto" / "fcm"),
                       "--arquivo-cache-representacao",
                       str(artefatos / "piloto" / "representacao_pca.npz")],
            "experimentos": ["--arquivo-caracteristicas", str(caracteristicas),
                             "--diretorio-saida", str(artefatos / "experimentos")],
            "avaliacao-clusters": [
                "--arquivo-caracteristicas", str(caracteristicas),
                "--diretorio-saida", str(artefatos / "avaliacao_clusters"),
            ],
            "avaliacao-fuzzificacao": [
                "--arquivo-caracteristicas", str(caracteristicas),
                "--diretorio-saida", str(artefatos / "avaliacao_fuzzificacao"),
            ],
        }
        # Mantém os parâmetros na interface já oferecida por cada módulo.
        comandos = []
        for etapa in etapas:
            if etapa not in argumentos:
                continue
            args = argumentos[etapa].copy()
            chave_configuracao = etapa.replace("-", "_")
            for chave, valor in config[chave_configuracao].items():
                if chave == "diretorio_cache_modelos":
                    valor = str((base / valor).resolve())
                args.append("--" + chave.replace("_", "-"))
                valores = valor if isinstance(valor, list) else [valor]
                args.extend(str(item) for item in valores)
            comandos.append((etapa, args))
        if "inventario" in etapas and not imagens.is_dir():
            raise ValueError(f"Pasta de imagens ausente: {imagens}")
        python = RAIZ / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        perfil = (
            "dev" if opcoes.acao == "testes"
            else "extracao" if any(e in etapas for e in ("extracao", "instalar"))
            else "base"
        )
        marcador = RAIZ / ".venv" / f".instalado-{perfil}"
        if not python.exists() and not opcoes.simular:
            print("Criando ambiente .venv...", flush=True)
            venv.EnvBuilder(with_pip=True).create(RAIZ / ".venv")
        if opcoes.reinstalar or not marcador.exists():
            ambiente_pronto = (
                not opcoes.simular
                and not opcoes.reinstalar
                and dependencias_disponiveis(python, perfil)
            )
            if ambiente_pronto:
                print(
                    f"Dependências do perfil '{perfil}' já estão disponíveis.",
                    flush=True,
                )
            elif not opcoes.simular:
                verificacao_pip = subprocess.run(
                    [str(python), "-m", "pip", "--version"],
                    cwd=RAIZ,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if verificacao_pip.returncode != 0:
                    executar(
                        [str(python), "-m", "ensurepip", "--upgrade"],
                        simular=False,
                    )
            if not ambiente_pronto:
                pacote = "." if perfil == "base" else f".[{perfil}]"
                executar(
                    [
                        str(python),
                        "-m",
                        "pip",
                        "install",
                        "--no-build-isolation",
                        "-e",
                        pacote,
                    ],
                    opcoes.simular,
                )
            if not opcoes.simular:
                marcador.touch()
        for etapa, args in comandos:
            modulo = (
                "caracteristicas"
                if etapa == "extracao"
                else etapa.replace("-", "_")
            )
            executar([str(python), "-m", f"clustering_imagens.{modulo}", *args],
                     opcoes.simular)
        if opcoes.acao == "testes":
            executar([str(python), "-m", "pytest"], opcoes.simular)
        if opcoes.acao == "dashboard":
            ambiente = os.environ.copy()
            ambiente["CLUSTERING_IMAGENS_DIR"] = str(imagens)
            ambiente["CLUSTERING_ARTEFATOS_DIR"] = str(artefatos / "experimentos")
            executar([str(python), "-m", "streamlit", "run",
                      str(RAIZ / "streamlit_app.py")], opcoes.simular, ambiente)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as erro:
        print(f"Falha na execução: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
