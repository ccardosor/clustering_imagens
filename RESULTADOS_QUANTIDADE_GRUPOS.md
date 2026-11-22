# Comparação da quantidade de grupos

Este experimento compara 8, 10, 12 e 15 grupos na mesma amostra de 500 imagens.
Todas as demais decisões foram mantidas fixas: características ResNet-50,
PCA-64, normalização L2, Fuzzy C-Means com `m=1,2`, inicialização k-means++ e
sementes 42, 123 e 2026. Assim, a comparação mede principalmente o efeito da
quantidade de grupos.

## Resultado resumido

| Grupos | Convergências | Grupos ocupados no pior caso | Xie–Beni mediano | Silhueta FCM média | Pertinência principal média | Concordância entre sementes | Rand ajustado médio | Menor distância² entre centroides |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 de 3 | 8 | 4,668 | 0,088 | 0,594 | 78,6% | 0,639 | 0,1301 |
| 10 | 2 de 3 | 10 | 8,882 | 0,084 | 0,555 | 77,7% | 0,660 | 0,0507 |
| 12 | 0 de 3 | 12 | 399.084,507 | 0,086 | 0,542 | 86,0% | 0,763 | 1,49 × 10⁻⁷ |
| 15 | 1 de 3 | 14 | 2.530.752,246 | 0,081 | 0,536 | 80,9% | 0,800 | 3,35 × 10⁻⁸ |

## Interpretação

### Oito grupos

Foi a única configuração que convergiu nas três sementes, preservou todos os
grupos e manteve os centroides claramente separados. Também apresentou o menor
Xie–Beni e a maior silhueta FCM média entre as opções comparadas. A execução da
semente 42 precisou de 296 das 300 iterações, portanto a convergência existe,
mas ainda está próxima do limite configurado.

A concordância dominante de 78,6% é moderada. Ela é semelhante à obtida com
dez grupos e mostra que reduzir a quantidade não elimina toda a sensibilidade
à inicialização.

### Dez grupos

Continua sendo uma configuração exploratória válida, mas uma das três sementes
não convergiu. Seu Xie–Beni mediano foi quase duas vezes maior que o de oito
grupos e a menor distância entre centroides caiu de 0,1301 para 0,0507. O Rand
ajustado foi ligeiramente maior, porém essa pequena vantagem não compensa a
piora de convergência e separação.

### Doze e quinze grupos

Essas configurações não são defensáveis no piloto. Em 12 grupos nenhuma
execução convergiu; em 15 grupos somente uma convergiu e outra ocupou apenas 14
dos 15 grupos. As distâncias mínimas entre centroides chegaram à ordem de
`10⁻⁷` e `10⁻⁸`, acompanhadas de Xie–Beni extremamente altos.

A concordância e o Rand ajustado relativamente elevados nessas linhas são
enganosos. Quando centroides colapsam ou ficam quase coincidentes, diferentes
sementes podem repetir uma solução degenerada. Por isso, estabilidade nunca é
interpretada isoladamente.

## Inspeção visual de oito grupos

Foram geradas pranchas com os 12 exemplos de maior pertinência de cada grupo na
semente 42. Os oito grupos apresentaram padrões visuais reconhecíveis:

1. flores saturadas, principalmente vermelhas, com formas tubulares ou em
   cachos;
2. inflorescências espinhosas, globulares ou com estruturas radiais marcadas;
3. flores delicadas em tons frios, com pétalas estreitas ou pequenas
   inflorescências;
4. flores radiais ou em camadas, em tons rosas, amarelos e pêssego;
5. flores claras em forma de taça, funil ou espata;
6. flores compostas com disco central, semelhantes visualmente a margaridas;
7. flores rosas ou lilases de pétalas largas, incluindo rosas, orquídeas e
   flores agrupadas;
8. flores amarelas ou laranjas muito saturadas e de contorno simples.

Os grupos 2, 6 e 8 são particularmente claros por morfologia ou cor. Os grupos
4 e 7 são mais amplos e ainda podem reunir subestruturas diferentes. Portanto,
oito grupos melhora a robustez numérica sem eliminar completamente grupos
guiados por cor ou categorias visualmente abrangentes.

## Decisão para a próxima etapa

Entre as opções testadas, **oito grupos passa a ser a configuração candidata
principal do piloto**. Isso não significa que oito seja o número verdadeiro de
espécies ou a solução final para as 8.189 imagens. A escolha significa apenas
que, nesta amostra e com esta representação, oito oferece o melhor equilíbrio
observado entre:

- convergência em múltiplas sementes;
- ocupação de todos os grupos;
- separação entre centroides;
- Xie–Beni e silhueta;
- estabilidade moderada;
- galerias visualmente interpretáveis.

As variações do expoente de fuzzificação já foram testadas e estão documentadas
em [`RESULTADOS_FUZZIFICACAO.md`](RESULTADOS_FUZZIFICACAO.md). `m=1,20`
permaneceu como candidato. PCA-64, oito grupos e `m=1,20` já foram promovidos
para os artefatos principais. O dataset completo também foi processado, como
registrado em [`RESULTADOS_DATASET_COMPLETO.md`](RESULTADOS_DATASET_COMPLETO.md).
O próximo passo é medir a influência da cor e do fundo.

