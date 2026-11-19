# Resultados do piloto

Execução realizada em 10 de novembro de 2026 com a amostra reproduzível de 500
imagens e semente 42.

## Proveniência

- Dataset encontrado: 8.189 JPEGs.
- Imagens válidas: 8.189.
- Exclusões: 0.
- SHA-256 de `amostra_piloto.csv`:
  `0a60c3c36ea8917946c0939930556c181e077194dc2a0fe0b8bb305fc81ee20a`.
- Extrator: ResNet-50 com `ResNet50_Weights.IMAGENET1K_V2`.
- Vetor por imagem: 2.048 características antes da camada de classificação.
- Representação do agrupamento: PCA-64 seguida de normalização L2.
- Variância explicada pelo PCA: 81,82%.
- Quantidade de grupos: 10.
- Inicialização do FCM: k-means++.

## Comparação do expoente de fuzzificação

| m | Convergiu | Iterações | Grupos dominantes ocupados | Pertinência principal média | Menor distância² entre centroides | Xie–Beni |
|---:|:---:|---:|---:|---:|---:|---:|
| 1,2 | sim | 169 | 10 | 0,5608 | 0,05068 | 11,38 |
| 1,5 | não | 300 | 4 | 0,1604 | 2,08 × 10⁻¹⁴ | 1,52 × 10¹³ |
| 2,0 | sim | 29 | 2 | 0,1000 | 2,13 × 10⁻¹⁵ | 4,70 × 10¹³ |

Todas as linhas de pertinência somaram 1 dentro da tolerância numérica. Apesar
de `m=2,0` ter atingido o critério de convergência, seus centroides ficaram
praticamente coincidentes e as pertinências quase uniformes em 0,1. Portanto,
essa convergência representa um estado degenerado e não dez grupos
interpretáveis. O mesmo processo já aparece em `m=1,5`, que também atingiu o
limite de 300 iterações.

O valor `m=1,2` foi acrescentado ao plano experimental porque foi o único dos
três a preservar dez grupos no piloto. Isso é uma hipótese para as próximas
rodadas, não uma seleção final do modelo.

## Referência K-Means

Na mesma representação, o K-Means produziu grupos com 35 a 67 imagens,
inércia 327,72 e silhueta 0,0906. A silhueta baixa indica que os grupos estão
pouco separados no espaço PCA-64. No FCM com `m=1,2`, a silhueta das
atribuições dominantes foi 0,0816.

Esses números sustentam uma leitura cautelosa: o fluxo técnico funciona e há
um resultado fuzzy não degenerado para `m=1,2`, mas uma única amostra e uma
única semente ainda não demonstram grupos visuais estáveis.

## PCA-64 e PCA-128 com múltiplas sementes

Foram executadas as sementes 42, 123 e 2026 mantendo a semente do PCA fixa em
42. Os clusters de cada repetição foram alinhados aos da semente 42 pelo menor
custo entre centroides.

| Representação | Variância explicada | Convergências | Concordância dominante média | Rand ajustado médio | Silhueta FCM média | Silhueta K-Means média |
|:---|---:|---:|---:|---:|---:|---:|
| PCA-64 | 81,82% | 2 de 3 | 77,7% | 0,660 | 0,084 | 0,091 |
| PCA-128 | 91,17% | 0 de 3 | 99,9% | 0,999 | 0,067 | 0,079 |

A concordância quase perfeita do PCA-128 não deve ser interpretada como
estabilidade de um bom modelo. Nenhuma execução convergiu em 300 iterações e a
menor distância quadrada entre centroides ficou entre `6,60 × 10⁻⁹` e
`1,09 × 10⁻⁶`, acompanhada de valores Xie–Beni muito altos. O resultado indica
trajetórias semelhantes em direção a centroides coincidentes.

No PCA-64, as sementes 42 e 123 convergiram. A semente 2026 manteve dez grupos
separados, mas atingiu o limite de iterações. A concordância das atribuições
dominantes com a referência foi 81,2% para a semente 123 e 74,2% para 2026.
Assim, PCA-64 é a representação mais defensável para a próxima inspeção
visual, ainda sem evidência suficiente para declarar uma configuração final.

## Verificações posteriores

A inspeção das galerias e a análise das mudanças por imagem foram realizadas e
estão documentadas em [`ANALISE_VISUAL_PILOTO.md`](ANALISE_VISUAL_PILOTO.md).
Elas encontraram núcleos morfológicos estáveis, grupos guiados principalmente
por cor e regiões sensíveis à semente. Os próximos experimentos devem:

1. comparar outras quantidades de grupos na amostra de 500 imagens;
2. medir separadamente os efeitos de cor, fundo e enquadramento;
3. escolher uma configuração por métricas, estabilidade e inspeção visual;
4. só então extrair características das 8.189 imagens e executar a grade final.
