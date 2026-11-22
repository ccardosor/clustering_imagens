# Comparação do expoente de fuzzificação

Este experimento compara valores de `m` próximos de 1,2 depois da seleção de
oito grupos. Foram mantidos fixos: amostra de 500 imagens, características
ResNet-50, PCA-64, normalização L2, inicialização k-means++ e sementes 42, 123 e
2026.

No Fuzzy C-Means, `m` controla o quanto as pertinências podem ser distribuídas
entre grupos. Valores muito próximos de 1 tornam a solução semelhante a um
agrupamento rígido; valores maiores aumentam a sobreposição, mas podem fazer os
centroides coincidirem.

## Resultado resumido

| m | Convergências | Grupos ocupados no pior caso | Xie–Beni mediano | Silhueta FCM média | Pertinência principal média | Concordância entre sementes | Rand ajustado médio | Menor distância² entre centroides |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,10 | 3 de 3 | 8 | 2,458 | 0,091 | 0,838 | 64,0% | 0,533 | 0,2497 |
| 1,15 | 3 de 3 | 8 | 3,094 | 0,090 | 0,721 | 63,0% | 0,526 | 0,1932 |
| 1,20 | 3 de 3 | 8 | 4,668 | 0,088 | 0,594 | 78,6% | 0,639 | 0,1301 |
| 1,25 | 0 de 3 | 8 | 589.321,455 | 0,071 | 0,427 | 99,3% | 0,987 | 6,19 × 10⁻⁷ |
| 1,30 | 0 de 3 | 7 | infinito | 0,055 | 0,304 | 96,9% | 0,953 | 0 |

## Interpretação

### `m=1,10`

Produziu a maior separação entre centroides, o menor Xie–Beni e a maior
silhueta. Entretanto, a pertinência principal média de 0,838 indica uma solução
próxima de um agrupamento rígido. A concordância dominante de apenas 64,0%
mostra que a boa separação dentro de cada execução não gerou a melhor
repetibilidade entre inicializações.

### `m=1,15`

Permitiu mais sobreposição que `m=1,10`, mas não trouxe ganho de estabilidade:
a concordância caiu para 63,0%. Como o Xie–Beni também aumentou, não apresentou
uma vantagem clara sobre os valores vizinhos.

### `m=1,20`

Convergiu nas três sementes, ocupou os oito grupos e manteve distância
mensurável entre os centroides. Sua pertinência principal média de 0,594
preserva sobreposição suficiente para justificar o uso do Fuzzy C-Means. Entre
as soluções numericamente válidas, teve a maior concordância entre sementes,
78,6%, e o maior Rand ajustado, 0,639.

A execução da semente 42 utilizou 296 das 300 iterações. Portanto, a
configuração converge, mas continua próxima do limite em uma das
inicializações.

### `m=1,25` e `m=1,30`

Nenhuma das seis execuções convergiu. Em `m=1,25`, a menor distância quadrada
entre centroides caiu para a ordem de `10⁻⁷`. Em `m=1,30`, houve centroides
coincidentes, apenas sete grupos dominantes ocupados e Xie–Beni infinito.

As concordâncias de 99,3% e 96,9% são artificiais: diferentes sementes
repetiram estados degenerados muito parecidos. Esse resultado reforça que
estabilidade nunca deve ser avaliada sem convergência, ocupação e distância
entre centroides.

## Decisão

**`m=1,20` permanece como expoente candidato.** Ele não venceu isoladamente em
separação, mas apresentou o melhor equilíbrio entre:

- convergência nas três sementes;
- ocupação dos oito grupos;
- centroides não coincidentes;
- pertinências genuinamente fuzzy;
- maior estabilidade entre as soluções válidas.

O Xie–Beni também não deve ser usado sozinho nesta comparação: o próprio
expoente `m` altera os pesos das pertinências usados no índice. A decisão
combina o índice com convergência, silhueta, estabilidade e inspeção visual.

## Configuração promovida e próxima etapa

A configuração principal do projeto foi atualizada para PCA-64, oito grupos e
`m=1,20`, e os artefatos principais foram regenerados. A execução sobre as
8.189 imagens também foi concluída e está em
[`RESULTADOS_DATASET_COMPLETO.md`](RESULTADOS_DATASET_COMPLETO.md). O próximo
experimento deve medir a influência da cor e do fundo.
