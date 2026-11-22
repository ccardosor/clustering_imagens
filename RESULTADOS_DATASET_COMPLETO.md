# Resultados sobre o dataset completo

Execução realizada em 22 de novembro de 2026 com todas as 8.189 imagens JPEG
válidas do conjunto 102 Flowers. Esta é a linha de base integral com a
configuração selecionada no piloto; ela não transforma os oito grupos em
espécies nem encerra a avaliação de cor e fundo.

## Configuração

- Extrator: ResNet-50 com `ResNet50_Weights.IMAGENET1K_V2`.
- Características por imagem: 2.048.
- Representação: PCA-64 seguida de normalização L2.
- Variância preservada pelo PCA: 77,68%.
- Agrupamento: Fuzzy C-Means com oito grupos e `m=1,20`.
- Inicialização: k-means++.
- Sementes: 42, 123 e 2026.
- Tolerância: `1 × 10⁻⁵`.
- Limite ampliado: 1.000 iterações.
- SHA-256 da lista integral:
  `4181cc7f8d2c411260530f23bbdb3dfcf802865cab4be5bd1b724cf531b6fbd0`.

A extração em CPU levou aproximadamente 6 minutos e 45 segundos. O arquivo
completo de características comprimidas ocupou cerca de 29,9 MB. A
representação PCA foi calculada uma vez e reutilizada entre as sementes.

## Resultados por semente

| Semente | Convergiu | Iterações | Xie–Beni | Silhueta FCM | Pertinência principal | Grupos ocupados | Menor distância² |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 42 | sim | 437 | 4,471 | 0,083 | 0,577 | 8 | 0,1350 |
| 123 | sim | 445 | 3,747 | 0,080 | 0,565 | 8 | 0,1611 |
| 2026 | sim | 416 | 3,747 | 0,080 | 0,565 | 8 | 0,1611 |

Com o limite anterior de 300 iterações, nenhuma semente atingia a tolerância,
embora todas mantivessem oito grupos e centroides separados. O limite foi
ampliado para 1.000 sem alterar PCA, quantidade de grupos, `m` ou tolerância.
As três execuções então convergiram entre 416 e 445 iterações.

## Resumo entre sementes

- Convergências: 3 de 3.
- Xie–Beni médio: 3,988.
- Silhueta FCM média: 0,081.
- Silhueta K-Means média: 0,085.
- Pertinência principal média: 0,569.
- Concordância dominante média: 75,0%.
- Índice Rand ajustado médio: 0,599.
- Diferença média entre pertinências alinhadas: 0,0377.
- Imagens que mudaram de grupo em pelo menos uma semente: 2.046 de 8.189
  (25,0%).

A estabilidade é moderada. Aproximadamente três quartos das imagens mantiveram
o grupo dominante, mas uma em cada quatro mudou em alguma inicialização. Isso
reforça a utilidade da galeria de imagens instáveis no dashboard.

## Tamanho dos grupos na semente 42

| Grupo | Imagens |
|---:|---:|
| 1 | 940 |
| 2 | 925 |
| 3 | 1.000 |
| 4 | 1.085 |
| 5 | 757 |
| 6 | 1.228 |
| 7 | 1.211 |
| 8 | 1.043 |

Os grupos variaram de 757 a 1.228 imagens. Não houve grupo vazio nem um único
grupo concentrando a maior parte do conjunto.

## Interpretação

O comportamento observado no piloto se manteve em linhas gerais: os
centroides permaneceram separados e a solução preservou pertinências fuzzy,
mas a silhueta continuou próxima de zero. Portanto, os grupos representam
padrões visuais sobrepostos, não classes naturais perfeitamente separadas.

PCA-64 preservou 77,68% da variância na base completa, abaixo dos 81,82% do
piloto. Essa diferença é esperada porque o conjunto integral apresenta maior
diversidade. A configuração continua tecnicamente válida, mas não deve ser
tratada como ótima sem novas comparações dirigidas.

## Próxima validação

Antes de considerar a análise encerrada, deve-se comparar a representação
atual com alternativas que reduzam a influência do cenário e da cor, por
exemplo escala de cinza, recorte central ou segmentação aproximada da flor. A
mesma avaliação deve repetir convergência, Xie–Beni, silhueta, estabilidade e
inspeção das galerias.
