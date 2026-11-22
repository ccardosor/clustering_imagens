# Guia de explicação do dashboard

## 1. Objetivo do trabalho

O projeto organiza imagens de flores por semelhança visual sem utilizar os
rótulos das espécies durante o agrupamento. O resultado não deve ser
interpretado como um classificador de espécies. Ele é uma análise exploratória
que procura responder a perguntas como:

- quais imagens compartilham cores, formas, texturas ou estruturas visuais;
- quais flores são exemplos representativos de um grupo;
- quais imagens estão na fronteira entre dois ou mais grupos;
- quanto o resultado muda quando a inicialização do algoritmo é alterada.

O fluxo usado no piloto é:

1. selecionar uma amostra reproduzível de 500 imagens válidas;
2. extrair 2.048 características de cada imagem com uma ResNet-50;
3. reduzir essas características com PCA;
4. aplicar normalização L2;
5. agrupar os vetores com Fuzzy C-Means;
6. comparar o resultado com K-Means e entre diferentes sementes;
7. inspecionar métricas e imagens no dashboard Streamlit.

Depois da seleção dos parâmetros, o mesmo fluxo foi executado sobre as 8.189
imagens. A linha de base completa usa PCA-64, oito grupos e `m=1,20`, convergiu
nas três sementes e está documentada em
[`RESULTADOS_DATASET_COMPLETO.md`](RESULTADOS_DATASET_COMPLETO.md). O dashboard
pode abrir o piloto ou a base completa conforme o arquivo de configuração.

## 2. Por que utilizar uma amostra de 500 imagens?

O conjunto possui 8.189 imagens válidas, mas o primeiro objetivo era validar o
fluxo completo antes de executar o experimento em toda a base. A amostra de
500 imagens foi escolhida como um **piloto computacional e metodológico**.

Ela é grande o suficiente para:

- testar leitura, pré-processamento e alinhamento entre arquivos e vetores;
- produzir aproximadamente 62 imagens por grupo quando são usados 8 grupos;
- observar exemplos típicos e de fronteira;
- comparar representações e sementes;
- identificar colapso de centroides e problemas de convergência;
- validar a exportação dos artefatos e o dashboard.

Ao mesmo tempo, é pequena o suficiente para permitir várias execuções do PCA,
FCM e K-Means com baixo tempo de espera. A extração com uma rede neural e a
repetição dos agrupamentos seriam mais caras sobre as 8.189 imagens, sobretudo
durante a fase em que parâmetros e código ainda estavam sendo corrigidos.

A amostra foi sorteada com semente fixa e contém caminhos únicos. Portanto, o
piloto pode ser reproduzido. Entretanto, 500 imagens não garantem que todos os
padrões da base completa estejam representados. Por isso, as conclusões do
piloto foram posteriormente verificadas no conjunto integral.

## 3. Algoritmos utilizados

### 3.1 ResNet-50

A ResNet-50 é uma rede neural convolucional com 50 camadas, originalmente
treinada para reconhecimento de imagens. A arquitetura utiliza conexões
residuais, que facilitam o treinamento de redes profundas ao permitir que a
informação atravesse blocos por caminhos diretos.

Neste trabalho foi usada a versão pré-treinada com os pesos
`ResNet50_Weights.IMAGENET1K_V2`. A camada final de classificação foi removida.
Assim, em vez de devolver uma categoria da ImageNet, a rede produz um vetor de
2.048 características para cada imagem.

Esse vetor funciona como uma representação visual compacta. Ele pode codificar
informações como:

- formatos e contornos;
- texturas;
- combinações de cores;
- partes e estruturas recorrentes;
- composição geral da fotografia.

#### Motivação para a escolha

- permite aplicar aprendizado por transferência sem treinar uma rede do zero;
- não exige rótulos de espécies para gerar as características;
- é uma arquitetura consolidada e reproduzível;
- oferece representações mais informativas que comparar diretamente os pixels;
- seus vetores podem ser reutilizados em vários experimentos de agrupamento.

A principal limitação é que a ResNet-50 foi treinada em imagens gerais, e não
especificamente em flores. Ela também pode representar fundo, iluminação e
enquadramento, fazendo com que esses fatores influenciem os grupos.

### 3.2 PCA

PCA, ou Análise de Componentes Principais, é uma técnica linear de redução de
dimensionalidade. Ela encontra novas direções ortogonais que concentram a maior
variação possível dos dados.

Os 2.048 atributos da ResNet são transformados em um número menor de
componentes. O primeiro componente explica a maior parcela possível da
variância; o segundo explica a maior parcela restante, mantendo-se ortogonal ao
primeiro; e assim sucessivamente.

#### Motivação para a escolha

- reduz o custo das distâncias usadas pelo agrupamento;
- remove direções de baixa variância que podem representar ruído;
- reduz redundância e correlação entre características;
- ajuda a combater a concentração de distâncias em espaços muito grandes;
- permite medir quanta variância foi preservada;
- é determinístico quando configuração e semente são fixadas.

Após o PCA, cada vetor recebe normalização L2. Dessa forma, todos passam a ter
norma igual a 1 e a distância deixa de ser dominada apenas pela magnitude do
vetor.

### 3.3 Fuzzy C-Means

O Fuzzy C-Means, ou FCM, é um algoritmo de agrupamento no qual cada observação
recebe um grau de pertinência para todos os grupos. Os graus ficam entre 0 e 1
e somam 1 para cada imagem.

Por exemplo, uma imagem pode receber:

- 0,46 de pertinência ao grupo 7;
- 0,31 ao grupo 1;
- valores menores para os demais grupos.

Isso indica que ela é mais associada ao grupo 7, mas possui características
visuais importantes do grupo 1. Esses valores representam afinidade relativa
aos centroides; não são probabilidades de espécie nem uma medida calibrada de
confiança.

#### Motivação para a escolha

Fotos de flores podem compartilhar características de vários padrões ao mesmo
tempo. Uma flor pode ter a cor predominante de um grupo e a forma das pétalas
de outro. O FCM preserva essa sobreposição, enquanto o K-Means obriga cada
imagem a pertencer exclusivamente a um grupo.

O expoente de fuzzificação controla quanto as pertinências podem se espalhar.
No piloto inicial, valores tradicionais como 1,5 e 2,0 fizeram os centroides
se aproximarem e as pertinências ficarem quase uniformes. Depois de selecionar
8 grupos, foram comparados `m=1,10`, `1,15`, `1,20`, `1,25` e `1,30` em três
sementes. O valor 1,2 preservou os 8 grupos, convergiu nas três execuções e
apresentou a maior estabilidade entre as soluções numericamente válidas. Na
base completa, a configuração convergiu nas três sementes após ampliar o limite
para 1.000 iterações, mas ainda precisa ser comparada com tratamentos de cor e
fundo.

O K-Means continua sendo executado sobre a mesma representação como uma
referência de agrupamento rígido.

## 4. Por que aplicar PCA antes do Fuzzy C-Means?

O FCM depende de distâncias euclidianas calculadas repetidamente entre cada
imagem e cada centroide. Aplicá-lo diretamente aos 2.048 atributos é possível,
mas traz dificuldades:

1. **Concentração das distâncias.** Em dimensões muito altas, as distâncias
   entre diferentes pontos tendem a ficar mais parecidas. Isso dificulta
   distinguir imagens próximas e distantes.
2. **Ruído e redundância.** Parte dos atributos pode carregar informação
   repetida ou pouco útil para separar as imagens.
3. **Custo computacional.** O custo de cada iteração cresce com o número de
   imagens, grupos e dimensões.
4. **Risco de colapso.** Se as distâncias ficarem pouco discriminantes, as
   pertinências podem se tornar quase uniformes e os centroides podem convergir
   para posições muito próximas.

O PCA cria um espaço menor e mais organizado antes do cálculo das distâncias.
Ele não garante, sozinho, um bom agrupamento e pode descartar informações
úteis, mas torna o FCM mais viável e auditável neste experimento. Os resultados
do PCA-128 mostram que preservar mais variância não significa necessariamente
produzir melhores grupos.

## 5. Diferença entre PCA-64 e PCA-128

As duas configurações começam com os mesmos vetores de 2.048 características.
A diferença é quantas direções principais são mantidas:

| Representação | Componentes | Variância preservada | Convergências | Interpretação do piloto |
|:---|---:|---:|---:|:---|
| PCA-64 | 64 | 81,82% | 3 de 3 | Melhor separação e comportamento mais estável |
| PCA-128 | 128 | 91,17% | 3 de 3 | Convergência numérica, mas uma semente degenerada |

O PCA-128 preserva cerca de nove pontos percentuais adicionais de variância,
mas também mantém mais dimensões de baixa contribuição. No experimento atual,
as três sementes convergiram numericamente, porém a semente 42 terminou com
centroides quase coincidentes e Xie–Beni de aproximadamente 1,16 milhão. Isso
mostra que convergência, isoladamente, não comprova a qualidade do resultado.

No PCA-64, as três sementes convergiram, ocuparam os 8 grupos e mantiveram os
centroides separados. A concordância dominante média foi 78,6% e o índice Rand
ajustado médio foi 0,639, indicando estabilidade moderada.

A estabilidade média do PCA-128 foi inferior à do PCA-64. Além disso, uma das
execuções convergiu para um estado degenerado. Por isso, estabilidade sempre
deve ser interpretada junto com convergência, separação dos centroides e
Xie–Beni.

## 6. Por que foram escolhidos 64 componentes e 8 grupos?

### 6.1 Escolha de 64 componentes

O valor 64 foi escolhido porque oferece um equilíbrio empírico entre retenção
de informação e simplificação:

- preservou 81,82% da variância da amostra;
- as três sementes convergiram com 8 grupos;
- manteve distância mensurável entre os centroides;
- apresentou Xie–Beni muito menor que o PCA-128;
- reduziu o vetor original em 32 vezes, de 2.048 para 64 dimensões;
- facilitou a repetição do experimento e a visualização dos resultados.

Portanto, 64 não é um número universalmente ideal. É a configuração mais
defensável entre as duas alternativas testadas no piloto.

### 6.2 Escolha de 8 grupos

O piloto inicial usou 10 grupos como uma quantidade pequena e interpretável.
Depois, foram comparados 8, 10, 12 e 15 grupos com PCA-64, `m=1,2` e três
sementes. Oito grupos foi a única configuração que convergiu nas três sementes,
ocupou todos os grupos e manteve separação mensurável entre os centroides. Com
500 imagens, isso produz em média 62,5 imagens por grupo e permite montar
galerias com exemplos suficientes.

Começar diretamente com 102 grupos seria arriscado: haveria menos de cinco
imagens por grupo, em média, na amostra. Isso aumentaria a sensibilidade à
inicialização e dificultaria distinguir grupos reais de divisões ocasionais.

O número 8 não representa oito espécies e não é uma quantidade universalmente
ideal. Ele é a melhor configuração entre as alternativas testadas neste
piloto. Na base completa, ainda será possível comparar quantidades maiores,
inclusive 25, 50 e 102 grupos. A configuração com 102 grupos é uma hipótese
relacionada ao número de espécies do conjunto, não uma obrigação para um método
não supervisionado.

## 7. Métricas apresentadas

### 7.1 Convergência

Indica se o deslocamento dos centroides ficou abaixo da tolerância antes de
atingir o limite de iterações. Convergir significa que o algoritmo estabilizou
numericamente, mas não garante que os grupos sejam úteis. Um resultado pode
convergir para centroides coincidentes e pertinências uniformes.

### 7.2 Pertinência principal

É a média do maior grau de pertinência de cada imagem. Na execução PCA-64 com
semente 42, o valor foi 0,600. Isso indica associações moderadas e sobreposição
relevante entre os grupos.

### 7.3 Diferença entre as duas maiores pertinências

O dashboard representa essa diferença por `Δ`:

- `Δ` alto: a imagem é muito mais associada ao primeiro grupo que ao segundo;
- `Δ` baixo: a imagem está em uma fronteira entre grupos.

Essa medida define a galeria de imagens ambíguas.

### 7.4 Silhueta

A silhueta compara a proximidade de uma imagem com o próprio grupo e com os
grupos vizinhos. Como o FCM é fuzzy, a métrica é calculada usando a atribuição
dominante de cada imagem.

Seu intervalo usual vai de -1 a 1:

- próximo de 1: grupos compactos e bem separados;
- próximo de 0: grupos sobrepostos;
- negativo: muitas imagens podem estar mais próximas de outro grupo.

A silhueta FCM de aproximadamente 0,092 na execução PCA-64 com semente 42 é
baixa. Isso confirma que
os padrões se sobrepõem e que os grupos não devem ser tratados como classes
naturais perfeitamente separadas.

### 7.5 Índice Xie–Beni

O índice Xie–Beni foi desenvolvido para avaliar agrupamentos fuzzy. Ele combina
compactação interna e separação entre centroides:

```text
             Σᵢ Σⱼ (uᵢⱼᵐ × ||xᵢ − vⱼ||²)
XB = ─────────────────────────────────────────
             n × minₚ≠q ||vₚ − vq||²
```

Onde:

- `uᵢⱼ` é a pertinência da imagem `i` ao grupo `j`;
- `m` é o expoente de fuzzificação;
- `xᵢ` é o vetor da imagem;
- `vⱼ` é o centroide do grupo;
- `n` é a quantidade de imagens.

Valores menores são preferíveis porque indicam grupos internamente compactos e
centroides separados. Se dois centroides quase coincidem, o denominador fica
próximo de zero e o índice cresce muito.

No PCA-64, os valores ficaram entre aproximadamente 3,80 e 4,67. No PCA-128,
ficaram entre cerca de 10,77 e 1,16 milhão. Essa diferença, especialmente na
semente 42, é uma evidência do risco de perda de separação no PCA-128.

O Xie–Beni deve ser comparado com cautela quando mudam representação,
quantidade de grupos ou escala. Ele não substitui a inspeção visual nem a
análise de estabilidade.

### 7.6 Concordância entre sementes e Rand ajustado

Os clusters de cada semente são alinhados aos da semente 42 pelo menor custo
entre centroides. Depois são calculados:

- a proporção de imagens com o mesmo grupo dominante;
- o índice Rand ajustado, que mede a concordância entre partições;
- a diferença média entre as matrizes de pertinência alinhadas.

No PCA-64, a concordância dominante média foi 78,6% e o Rand ajustado médio foi
0,639. Isso representa estabilidade moderada, não perfeita.

## 8. Como explicar as áreas do dashboard

### Seletor de conjunto analisado

Quando os artefatos estão disponíveis localmente, a barra lateral permite
alternar entre o piloto de 500 imagens e o dataset completo de 8.189 imagens.
O dataset completo é a opção inicial. A troca substitui métricas, projeções,
pertinências, estabilidade e galerias em conjunto, evitando misturar resultados
das duas execuções.

### Seletor de modelo

Na barra lateral é possível escolher a representação PCA e a semente. Essa
troca permite verificar se gráficos, métricas e galerias permanecem coerentes
entre execuções.

### Indicadores superiores

- **Imagens:** tamanho da amostra analisada.
- **Variância PCA:** informação preservada pela redução.
- **Pertinência principal:** afinidade dominante média.
- **Silhueta FCM:** separação das atribuições dominantes.
- **Concordância entre sementes:** estabilidade após alinhamento.
- **Convergência:** estabilização numérica da execução selecionada.

### Mapa visual da amostra

Cada ponto representa uma imagem. A cor indica o grupo FCM dominante e a
opacidade representa a pertinência principal. As duas coordenadas são uma nova
projeção para visualização; elas não mostram todas as 64 dimensões usadas pelo
agrupamento.

Pontos de cores diferentes misturados ajudam a explicar a silhueta baixa. Uma
separação aparente ou uma sobreposição no gráfico 2D deve ser tratada como
indício visual, não como reprodução exata do espaço do FCM.

### Tamanho dos grupos

Mostra quantas imagens têm cada grupo como atribuição dominante. Na execução
PCA-64 com semente 42, os grupos possuem entre 41 e 77 imagens. Não há um único
grupo concentrando quase toda a amostra.

### Galeria de imagens

- **Típicas:** ordena pelas maiores pertinências ao grupo selecionado.
- **Ambíguas:** ordena pela menor diferença entre as duas maiores
  pertinências.
- **Instáveis:** alinha os clusters das diferentes sementes, mostra como o
  grupo dominante de cada imagem mudou e permite baixar a tabela completa.

A galeria verifica se o algoritmo capturou características das flores ou
efeitos indesejados de fundo, iluminação, escala e enquadramento.

### Comparação experimental

A tabela reúne todas as sementes e representações. Nenhuma métrica deve ser
lida isoladamente. A configuração mais promissora deve combinar convergência,
centroides separados, Xie–Beni razoável, estabilidade entre sementes e
coerência visual nas galerias.

### Escolha da quantidade de grupos

Esta área compara 8, 10, 12 e 15 grupos mantendo PCA-64, `m=1,2` e as três
sementes. O gráfico de Xie–Beni usa escala logarítmica porque 12 e 15 grupos
produziram centroides quase coincidentes e valores extremamente altos.

Oito grupos foi a única configuração que convergiu nas três sementes, ocupou
todos os grupos e manteve separação clara entre centroides. Por isso, aparece
como candidata do piloto. A indicação não afirma que existam exatamente oito
espécies ou categorias naturais; ela identifica o melhor equilíbrio entre as
configurações efetivamente testadas.

### Escolha do expoente de fuzzificação

Depois de fixar oito grupos, esta área compara `m=1,10`, `1,15`, `1,20`, `1,25`
e `1,30`. Valores menores geraram grupos mais rígidos e separados, enquanto
`m=1,25` e `m=1,30` levaram a centroides quase coincidentes e falta de
convergência.

`m=1,20` aparece como candidato porque convergiu nas três sementes, preservou
os oito grupos e obteve a maior concordância entre as soluções numericamente
válidas. O dashboard destaca que as concordâncias ainda maiores de `m=1,25` e
`m=1,30` são artificiais e resultam de estados degenerados.

## 9. Como o agrupamento pode ser melhorado

### 9.1 Consolidar a execução sobre a base completa

A extração das 8.189 imagens foi concluída. A solução preservou os oito grupos,
convergiu nas três sementes e apresentou estabilidade dominante média de
75,0%. Como 25,0% das imagens mudaram de grupo em alguma semente, a inspeção de
instabilidade continua necessária.

### 9.2 Comparar mais configurações

- manter PCA-64, oito grupos e `m=1,20` como configuração candidata;
- repetir cada configuração com mais sementes;
- testar outras quantidades de componentes entre 64 e 128;
- selecionar configurações com várias métricas, não apenas uma.

### 9.3 Reduzir a influência do fundo

É possível segmentar a flor, recortar a região central ou usar um detector
antes da extração. Isso pode impedir que vegetação, céu, vasos e iluminação
definam clusters que deveriam representar a flor.

### 9.4 Comparar outros extratores de características

Representações de modelos como DINO, CLIP ou arquiteturas mais recentes podem
capturar relações semânticas diferentes das produzidas pela ResNet-50. Também
é possível ajustar um modelo com imagens de flores, caso existam rótulos para
treinamento ou validação.

### 9.5 Avaliar métodos não lineares com cuidado

UMAP ou autoencoders podem produzir representações mais adequadas a estruturas
não lineares. Entretanto, UMAP usado apenas no gráfico não deve substituir
automaticamente o espaço do agrupamento, pois pode distorcer distâncias. A
mudança precisa ser validada com estabilidade, métricas e galerias.

### 9.6 Usar rótulos somente para avaliação externa

Se os rótulos das 102 espécies forem disponibilizados, eles podem medir pureza,
Rand ajustado ou informação mútua após o agrupamento. Isso não transforma o
método em supervisionado, desde que os rótulos não sejam usados para formar os
clusters.

### 9.7 Investigar imagens instáveis e ambíguas

Imagens que mudam de grupo entre sementes ou apresentam `Δ` pequeno devem ser
inspecionadas. Elas podem revelar:

- transições visuais reais entre grupos;
- fotos com várias flores;
- enquadramentos atípicos;
- influência do fundo;
- problemas de qualidade ou pré-processamento.

## 10. Conclusão sugerida para a apresentação

O piloto demonstra que o fluxo técnico funciona e que o Fuzzy C-Means consegue
representar sobreposição entre padrões visuais. Depois de comparar 8, 10, 12 e
15 grupos e diferentes expoentes, a configuração PCA-64 com 8 grupos e
`m=1,20` é a candidata mais adequada para continuar a análise. Ela convergiu
nas três sementes, ocupou todos os grupos e manteve os centroides separados.
Entre os expoentes válidos, também apresentou a maior concordância.

Ao mesmo tempo, a silhueta baixa mostra que as imagens não formam grupos
naturalmente isolados. Portanto, o dashboard deve ser apresentado como uma
ferramenta de exploração e diagnóstico. A base completa já foi processada e
validada numericamente, com convergência nas três sementes. A próxima etapa é
medir a influência da cor e do fundo e comparar a coerência visual das novas
representações com esta linha de base.
