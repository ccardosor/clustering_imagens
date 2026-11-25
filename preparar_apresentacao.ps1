[CmdletBinding()]
param(
    [switch]$NaoIniciar
)

$ErrorActionPreference = "Stop"
$ProgressoAnterior = $ProgressPreference
$ProgressPreference = "SilentlyContinue"

$RaizProjeto = Split-Path -Parent $MyInvocation.MyCommand.Path
$DiretorioDataset = Join-Path $RaizProjeto "102flowers"
$DiretorioImagens = Join-Path $DiretorioDataset "jpg"
$DiretorioCache = Join-Path $RaizProjeto ".cache\apresentacao"
$ArquivoDataset = Join-Path $DiretorioCache "102flowers.tgz"
$UrlDataset = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
$QuantidadeEsperada = 8189

function Obter-QuantidadeImagens {
    if (-not (Test-Path -LiteralPath $DiretorioImagens -PathType Container)) {
        return 0
    }
    return @(Get-ChildItem -LiteralPath $DiretorioImagens -File -Filter "*.jpg").Count
}

function Obter-Python {
    $PythonLocal = Join-Path $RaizProjeto ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $PythonLocal -PathType Leaf) {
        return $PythonLocal
    }

    $ComandoPy = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $ComandoPy) {
        return $ComandoPy.Source
    }

    $ComandoPython = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $ComandoPython) {
        return $ComandoPython.Source
    }

    throw "Python 3.11 ou mais recente nao foi encontrado. Instale-o antes de continuar."
}

try {
    Set-Location -LiteralPath $RaizProjeto

    $QuantidadeAtual = Obter-QuantidadeImagens
    if ($QuantidadeAtual -ne $QuantidadeEsperada) {
        Write-Host "Baixando o Oxford Flowers 102 da fonte oficial..."
        New-Item -ItemType Directory -Path $DiretorioCache -Force | Out-Null
        if (-not (Test-Path -LiteralPath $ArquivoDataset -PathType Leaf)) {
            Invoke-WebRequest -Uri $UrlDataset -OutFile $ArquivoDataset
        }

        $ComandoTar = Get-Command "tar.exe" -ErrorAction SilentlyContinue
        if ($null -eq $ComandoTar) {
            throw "O utilitario tar.exe nao foi encontrado no Windows."
        }

        New-Item -ItemType Directory -Path $DiretorioDataset -Force | Out-Null
        & $ComandoTar.Source -xzf $ArquivoDataset -C $DiretorioDataset
        if ($LASTEXITCODE -ne 0) {
            throw "A extracao do dataset terminou com o codigo $LASTEXITCODE."
        }

        $QuantidadeAtual = Obter-QuantidadeImagens
        if ($QuantidadeAtual -ne $QuantidadeEsperada) {
            throw "Foram encontradas $QuantidadeAtual imagens; eram esperadas $QuantidadeEsperada."
        }
    }
    Write-Host "Dataset validado: $QuantidadeAtual imagens."

    $Python = Obter-Python
    $PythonAmbiente = Join-Path $RaizProjeto ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $PythonAmbiente -PathType Leaf)) {
        Write-Host "Criando o ambiente virtual..."
        if ([System.IO.Path]::GetFileName($Python) -ieq "py.exe") {
            & $Python -3.11 -m venv ".venv"
        }
        else {
            & $Python -m venv ".venv"
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Nao foi possivel criar o ambiente virtual."
        }
    }

    Write-Host "Instalando as dependencias do dashboard..."
    & $PythonAmbiente -m pip install -e "."
    if ($LASTEXITCODE -ne 0) {
        throw "A instalacao das dependencias falhou."
    }

    if (-not $NaoIniciar) {
        Write-Host "Iniciando o dashboard..."
        & $PythonAmbiente -m streamlit run "streamlit_app.py"
    }
    else {
        Write-Host "Preparacao concluida. Execute .venv\Scripts\streamlit run streamlit_app.py"
    }
}
finally {
    $ProgressPreference = $ProgressoAnterior
}
