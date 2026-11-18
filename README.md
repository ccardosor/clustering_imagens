# Agrupamento fuzzy de imagens de flores

Projeto de agrupamento visual não supervisionado para as 8.189 imagens JPEG disponíveis em `102flowers/jpg/`. O dataset fica na pasta local e é excluído do versionamento.

- [Motivação para usar Fuzzy C-Means](MOTIVACAO.md)
- [Plano de implementação e avaliação](PLANO.md)
- [Hipóteses e resultados esperados](EXPECTATIVAS.md)
- [Resultados do piloto](RESULTADOS_PILOTO.md)
- [Guia de explicação do dashboard](EXPLICACAO_DASHBOARD.md)

O primeiro marco é validar o fluxo completo com uma amostra reproduzível de 500 imagens.

## Execução rápida

Com Python 3.11 ou mais recente e o dataset em `102flowers/jpg/`, execute:

```powershell
python executar.py
```

O lançador cria `.venv`, instala as dependências necessárias e executa
inventário, extração, piloto e experimentos em sequência. A primeira extração
precisa de internet para baixar os pesos da ResNet-50. Uma falha interrompe a
sequência com código de saída diferente de zero.

Para abrir o dashboard após gerar os resultados:

```powershell
python executar.py dashboard
```

Edite `configuracao.toml` para ajustar caminhos, amostra, dispositivo, lote,
PCA, grupos e sementes. Os caminhos são relativos à pasta do arquivo TOML.
Você também pode escolher outro arquivo com `--config minha_configuracao.toml`.

| Comando | Ação |
| --- | --- |
| `python executar.py --simular` | Mostra a sequência sem executar |
| `python executar.py instalar` | Prepara o ambiente com extração |
| `python executar.py inventario` | Valida imagens e sorteia a amostra |
| `python executar.py extracao` | Extrai os vetores da amostra existente |
| `python executar.py piloto` | Agrupa os vetores existentes |
| `python executar.py experimentos` | Executa a comparação configurada |
| `python executar.py testes` | Instala dependências de teste e roda a suíte |

Use `--reinstalar` para atualizar a instalação após mudar dependências.
Cada execução do pipeline recalcula as etapas e sobrescreve seus resultados.
Para repetir somente uma etapa, use seu comando específico.

## Comandos individuais

As interfaces abaixo continuam disponíveis para uso direto.

### Primeiro marco — inventário e amostra

Requisitos: Python 3.11 ou mais recente. Crie um ambiente virtual e instale o
projeto com as dependências de desenvolvimento:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,extracao]"
```

O comando abaixo valida a estrutura e a decodificação de todos os JPEGs e
sorteia 500 imagens válidas com a semente 42:

```powershell
.venv\Scripts\inventariar-imagens `
  --diretorio-imagens 102flowers\jpg `
  --diretorio-saida artefatos\inventario
```

São gerados quatro arquivos:

- `inventario.csv`: uma linha por JPEG encontrado, em ordem estável;
- `exclusoes.csv`: imagens inválidas e o motivo de cada exclusão;
- `amostra_piloto.csv`: ordem do sorteio e caminho relativo das 500 imagens;
- `resumo.json`: contagens, caminhos e parâmetros usados na execução.

Para executar os testes:

```powershell
.venv\Scripts\python -m pytest
```

## Piloto completo

A extração usa os pesos versionados `ResNet50_Weights.IMAGENET1K_V2` e salva o
vetor de 2.048 dimensões anterior à camada de classificação. O arquivo JSON ao
lado do `.npz` registra pesos, transformação, versões, dispositivo e SHA-256 da
amostra:

```powershell
.venv\Scripts\extrair-caracteristicas `
  --diretorio-imagens 102flowers\jpg `
  --arquivo-amostra artefatos\inventario\amostra_piloto.csv `
  --arquivo-saida artefatos\piloto\caracteristicas.npz
```

O piloto aplica PCA com 64 componentes, normalização L2 e Fuzzy C-Means com 10
grupos, expoente de fuzzificação 1,2 e inicialização k-means++. O valor 1,2 é o
padrão do piloto porque 1,5 e 2,0 produziram centroides coincidentes nessa
amostra; os resultados e a justificativa estão em
[`RESULTADOS_PILOTO.md`](RESULTADOS_PILOTO.md).

```powershell
.venv\Scripts\executar-piloto `
  --arquivo-caracteristicas artefatos\piloto\caracteristicas.npz `
  --diretorio-saida artefatos\piloto\fcm
```

O resultado inclui `metricas.json`, `modelo_fcm.npz` e `pertinencias.csv`. O
CSV contém todas as pertinências, os dois grupos mais fortes, a diferença
entre eles e o grupo atribuído pelo K-Means de referência para cada imagem.

## Comparação entre representações e sementes

O comando abaixo executa PCA-64 e PCA-128 com três sementes de agrupamento. A
semente do PCA permanece fixa para que a comparação meça o efeito da
inicialização do FCM, e não uma mudança da representação:

```powershell
.venv\Scripts\executar-experimentos `
  --arquivo-caracteristicas artefatos\piloto\caracteristicas.npz `
  --diretorio-saida artefatos\experimentos `
  --componentes-pca 64 128 `
  --sementes 42 123 2026
```

Além dos artefatos de cada execução, são gerados:

- `resultados.csv`: métricas por representação e semente;
- `estabilidade.csv`: concordância após alinhar os centroides;
- `resumo_representacoes.csv`: médias usadas no dashboard;
- `projecao_2d.csv`: coordenadas e grupos para a visualização.

## Visualização Streamlit

O dashboard permite trocar a execução, comparar métricas, explorar a projeção
bidimensional e filtrar galerias de imagens típicas ou ambíguas:

```powershell
.venv\Scripts\streamlit run streamlit_app.py
```

O app lê os artefatos já calculados e não executa ResNet, PCA ou FCM durante a
interação. Para instalar somente as dependências do dashboard, sem PyTorch:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e .
```
