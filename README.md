# Agrupamento fuzzy de imagens de flores

Projeto de agrupamento visual não supervisionado para as 8.189 imagens JPEG disponíveis em `102flowers/jpg/`. O dataset fica na pasta local e é excluído do versionamento.

- [Motivação para usar Fuzzy C-Means](MOTIVACAO.md)
- [Plano de implementação e avaliação](PLANO.md)
- [Hipóteses e resultados esperados](EXPECTATIVAS.md)
- [Resultados do piloto](RESULTADOS_PILOTO.md)
- [Análise visual dos grupos do piloto](ANALISE_VISUAL_PILOTO.md)
- [Comparação da quantidade de grupos](RESULTADOS_QUANTIDADE_GRUPOS.md)
- [Comparação do expoente de fuzzificação](RESULTADOS_FUZZIFICACAO.md)
- [Resultados sobre o dataset completo](RESULTADOS_DATASET_COMPLETO.md)
- [Guia de explicação do dashboard](EXPLICACAO_DASHBOARD.md)

O primeiro marco foi validar o fluxo completo com uma amostra reproduzível de
500 imagens; a linha de base sobre as 8.189 imagens também já foi concluída.

## Execução rápida

Com Python 3.11 ou mais recente e o dataset em `102flowers/jpg/`, execute:

```powershell
python executar.py
```

O lançador cria `.venv`, instala as dependências necessárias e executa
inventário, extração, piloto, experimentos e comparações da quantidade de
grupos e do expoente de fuzzificação em sequência. A primeira extração precisa
de internet para baixar os pesos da ResNet-50. Uma falha interrompe a sequência
com código de saída diferente de zero.

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
| `python executar.py avaliacao-clusters` | Compara 8, 10, 12 e 15 grupos |
| `python executar.py avaliacao-fuzzificacao` | Compara valores de `m` com 8 grupos |
| `python executar.py testes` | Instala dependências de teste e roda a suíte |

Use `--reinstalar` para atualizar a instalação após mudar dependências.
Para repetir somente uma etapa, use seu comando específico. A extração mantém
um checkpoint por lote e pode ser retomada após uma interrupção. As grades de
experimentos também reutilizam a representação PCA quando características e
parâmetros não mudaram.

## Execução sobre as 8.189 imagens

`configuracao_completa.toml` mantém a execução integral separada do piloto em
`artefatos/completo`. Para gerar somente a configuração candidata, sem repetir
toda a seleção de hiperparâmetros:

```powershell
python executar.py inventario --config configuracao_completa.toml
python executar.py extracao --config configuracao_completa.toml
python executar.py experimentos --config configuracao_completa.toml
python executar.py dashboard --config configuracao_completa.toml
```

A configuração completa usa PCA-64, oito grupos, `m=1,20` e sementes 42, 123 e
2026. Se a extração for interrompida, execute novamente o mesmo comando: o
processamento continuará depois do último lote persistido. O arquivo final
`.npz` só substitui o checkpoint após a validação de todas as características.
Os resultados observados estão em
[`RESULTADOS_DATASET_COMPLETO.md`](RESULTADOS_DATASET_COMPLETO.md).

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

O piloto aplica PCA com 64 componentes, normalização L2 e Fuzzy C-Means com 8
grupos, expoente de fuzzificação 1,2 e inicialização k-means++. Essa configuração
foi promovida após comparar quantidades de grupos e valores de `m` em três
sementes. Os resultados e as justificativas estão em
[`RESULTADOS_QUANTIDADE_GRUPOS.md`](RESULTADOS_QUANTIDADE_GRUPOS.md) e
[`RESULTADOS_FUZZIFICACAO.md`](RESULTADOS_FUZZIFICACAO.md). O primeiro ensaio,
com 10 grupos, permanece documentado em
[`RESULTADOS_PILOTO.md`](RESULTADOS_PILOTO.md) como histórico experimental.

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

## Comparação da quantidade de grupos

Para comparar 8, 10, 12 e 15 grupos mantendo PCA-64, `m=1,2` e as mesmas
sementes:

```powershell
python executar.py avaliacao-clusters
```

O resumo é salvo em `artefatos/avaliacao_clusters/resumo_clusters.csv` e
aparece automaticamente no dashboard. Os resultados observados no piloto
estão documentados em [`RESULTADOS_QUANTIDADE_GRUPOS.md`](RESULTADOS_QUANTIDADE_GRUPOS.md).

## Comparação do expoente de fuzzificação

Depois de selecionar oito grupos, compare valores de `m` próximos de 1,2:

```powershell
python executar.py avaliacao-fuzzificacao
```

O resumo é salvo em
`artefatos/avaliacao_fuzzificacao/resumo_fuzzificacao.csv` e também aparece no
dashboard. A interpretação está em
[`RESULTADOS_FUZZIFICACAO.md`](RESULTADOS_FUZZIFICACAO.md).

## Inspeção visual e estabilidade por imagem

As pranchas reproduzíveis reúnem os exemplos com maior pertinência de cada
grupo e as imagens mais ambíguas:

```powershell
.venv\Scripts\gerar-pranchas `
  --arquivo-pertinencias artefatos\experimentos\pca_064\semente_0042\pertinencias.csv `
  --diretorio-imagens 102flowers\jpg `
  --diretorio-saida artefatos\analise_visual\pca64_s42
```

A análise de estabilidade alinha os clusters de diferentes sementes e ordena
as imagens pela quantidade de mudanças do grupo dominante:

```powershell
.venv\Scripts\analisar-estabilidade-imagens `
  --diretorio-modelos artefatos\experimentos\pca_064 `
  --sementes 42 123 2026 `
  --arquivo-saida artefatos\analise_visual\pca64_s42\estabilidade_imagens.csv
```

As conclusões da inspeção das dez pranchas estão em
[`ANALISE_VISUAL_PILOTO.md`](ANALISE_VISUAL_PILOTO.md).

## Visualização Streamlit

O dashboard permite trocar a execução, comparar métricas, explorar a projeção
bidimensional e filtrar galerias de imagens típicas, ambíguas ou instáveis
entre sementes:

```powershell
.venv\Scripts\streamlit run streamlit_app.py
```

Quando os dois conjuntos de artefatos estão disponíveis, a barra lateral
oferece as opções **Piloto · 500** e **Dataset completo · 8.189**. O dataset
completo é selecionado inicialmente, e a escolha também pode ser compartilhada
pelos parâmetros da URL.

Na área principal, **Pontos** preserva a inspeção individual e pode contornar
imagens ambíguas segundo o limite de `Δ`. **Densidade** agrega regiões da
projeção para evitar a sobreposição visual das 8.189 imagens. O seletor
**Grupo em foco** filtra o mapa e a galeria e destaca o mesmo grupo no gráfico
de barras, que permanece em ordem numérica e mostra as quantidades diretamente.

O app lê os artefatos já calculados e não executa ResNet, PCA ou FCM durante a
interação. Para instalar somente as dependências do dashboard, sem PyTorch:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e .
```
