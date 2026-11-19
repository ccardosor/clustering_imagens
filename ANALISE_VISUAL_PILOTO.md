# Análise visual dos grupos do piloto

Esta análise verifica se os dez grupos do piloto PCA-64 representam padrões
visuais reconhecíveis ou apenas separações numéricas difíceis de interpretar.
Ela complementa as métricas de `RESULTADOS_PILOTO.md`: uma silhueta baixa pode
conviver com alguns núcleos visuais coerentes, enquanto uma pertinência alta em
uma execução não garante que a imagem permaneça no mesmo grupo em outra
semente.

## Procedimento

- Execução inspecionada: PCA-64, Fuzzy C-Means com `m=1,2`, dez grupos e
  semente 42.
- Para cada grupo, foram selecionadas as 12 imagens com maior pertinência entre
  aquelas cujo grupo dominante era o grupo analisado.
- Também foram selecionadas as 20 imagens com menor diferença entre a maior e
  a segunda maior pertinência. Elas representam as fronteiras mais ambíguas.
- Os grupos das sementes 123 e 2026 foram alinhados aos centroides da semente
  42. Para cada imagem, registrou-se se o grupo dominante mudou.

As pranchas e as tabelas são artefatos locais reproduzíveis. Elas não são
versionadas porque contêm imagens do dataset:

```powershell
.venv\Scripts\gerar-pranchas `
  --arquivo-pertinencias artefatos\experimentos\pca_064\semente_0042\pertinencias.csv `
  --diretorio-imagens 102flowers\jpg `
  --diretorio-saida artefatos\analise_visual\pca64_s42

.venv\Scripts\analisar-estabilidade-imagens `
  --diretorio-modelos artefatos\experimentos\pca_064 `
  --sementes 42 123 2026 `
  --arquivo-saida artefatos\analise_visual\pca64_s42\estabilidade_imagens.csv
```

Esta é uma leitura de aparência, não uma classificação botânica. Os nomes de
espécies abaixo são evitados deliberadamente: o modelo não recebeu rótulos, e
o que interessa é identificar cor, forma, textura, fundo e enquadramento.

## Leitura dos dez grupos

| Grupo | Imagens | Padrão visual predominante | Instáveis entre sementes | Leitura |
|---:|---:|:---|---:|:---|
| 1 | 50 | Flores muito saturadas, sobretudo vermelhas, laranjas e roxas; formas tubulares, alongadas ou em cachos | 48 (96,0%) | As imagens típicas são convincentes, mas o grupo se reorganiza fortemente entre sementes, principalmente em direção aos grupos 3 e 10. É um núcleo visual aparente, ainda não um cluster estável. |
| 2 | 49 | Flores claras, rosas, roxas ou azuis, com pétalas estreitas, estruturas finas ou inflorescências pequenas | 7 (14,3%) | Relativamente estável. Combina paleta fria com textura delicada; ainda é preciso testar quanto da separação vem da forma e quanto vem da cor. |
| 3 | 54 | Flores abertas, geralmente claras, com pétalas largas e centro reprodutivo evidente | 26 (48,1%) | Interpretação razoável, mas há sobreposição com flores rosadas do grupo 10 e com exemplares claros do grupo 4. |
| 4 | 56 | Flores brancas ou muito claras, frequentemente em forma de taça, funil ou espata, sobre fundo verde ou escuro | 20 (35,7%) | Grupo mais heterogêneo. Cor clara, fundo e formato parecem competir entre si. Muitas imagens de fronteira aparecem na divisão 4/5. |
| 5 | 42 | Estruturas exóticas, angulares ou alongadas, com presença frequente de laranja, vermelho e amarelo | 13 (31,0%) | Há um tema visual, mas ele mistura cor, orientação e morfologia. As pertinências típicas são menores que as dos núcleos mais fortes. |
| 6 | 44 | Inflorescências globulares ou espinhosas, em geral roxas e azuladas, com textura radial bem marcada | 5 (11,4%) | Um dos grupos mais coerentes morfologicamente. A estabilidade sugere que a textura e o contorno são sinais fortes no embedding. |
| 7 | 58 | Flores radiais de pétalas largas ou em camadas, em tons claros, amarelos, rosas e pêssego | 25 (43,1%) | As imagens mais típicas são nítidas, mas a categoria é ampla e faz transição para os grupos 3 e 10. |
| 8 | 36 | Flores amarelas ou laranjas muito saturadas, geralmente com poucas pétalas largas e contorno simples | 0 (0,0%) | É o grupo mais estável. A coesão é forte, embora aparentemente muito influenciada pela cor; um teste sem cor ajudará a verificar isso. |
| 9 | 61 | Flores compostas com disco central e pétalas radiais, semelhantes visualmente a margaridas | 3 (4,9%) | É o grupo morfologicamente mais claro e um dos mais estáveis. Indica que a ResNet capturou estrutura visual útil, não apenas paleta. |
| 10 | 50 | Flores rosas ou magenta, de pétalas largas, incluindo formas abertas e flores agrupadas | 13 (26,0%) | Coerente principalmente por cor e pétalas largas. A proximidade com os grupos 1, 3 e 7 explica parte das mudanças. |

## Imagens de fronteira

Nas 20 imagens mais ambíguas, a maior pertinência ficou aproximadamente entre
0,18 e 0,40, e a diferença entre as duas maiores pertinências ficou entre
0,000 e 0,012. Portanto, o modelo não tem base para tratá-las como exemplos
inequívocos de um único grupo.

As fronteiras mais recorrentes foram:

- grupos 4 e 5: flores claras ou angulares em que forma, cor e fundo fornecem
  sinais conflitantes;
- grupos 3 e 10: flores abertas e rosadas, com pétalas largas;
- grupos 7 e 10: flores radiais ou em camadas, também dominadas por rosa;
- grupos 2 e 5: estruturas finas ou alongadas que não seguem a paleta mais
  comum de nenhum dos dois grupos.

Essa é precisamente a informação que justifica o Fuzzy C-Means: em vez de
forçar certeza, a tabela de pertinências explicita que essas imagens ocupam
regiões intermediárias.

## Estabilidade por imagem

Após o alinhamento dos centroides, 160 das 500 imagens (32,0%) mudaram de grupo
em pelo menos uma das duas sementes comparadas. Destas, 97 mudaram em uma
semente e 63 mudaram nas duas. As 340 imagens estáveis tiveram pertinência
principal média de 0,617 e margem média entre os dois grupos mais fortes de
0,470. Entre as instáveis, esses valores caíram para 0,441 e 0,227.

O resultado confirma a hipótese geral: imagens próximas às fronteiras tendem a
ser menos estáveis. A relação, porém, não é perfeita. O grupo 1 contém exemplos
típicos de alta pertinência na semente 42, mas 96% de suas imagens mudam em ao
menos uma repetição. Isso indica uma reorganização global dos centroides, e não
somente dúvidas isoladas em imagens de fronteira.

## Conclusão

O piloto encontrou três tipos de resultado:

1. **Núcleos morfológicos fortes e estáveis:** grupos 6 e 9.
2. **Núcleo estável, mas possivelmente guiado por cor:** grupo 8.
3. **Regiões interpretáveis, porém sobrepostas ou sensíveis à semente:** os
   demais grupos, sobretudo 1, 3 e 7.

Assim, dez grupos é uma hipótese útil para exploração, mas ainda não uma
quantidade validada. A análise também mostra que a ResNet representa tanto a
flor quanto fatores como cor, fundo e enquadramento. O próximo experimento deve
medir esses efeitos antes de processar definitivamente as 8.189 imagens.

## Próximos passos recomendados

O filtro de **imagens instáveis** já foi incorporado ao dashboard, com os
grupos atribuídos nas três sementes, a variação média das pertinências e a
opção de baixar a tabela. A comparação de 8, 10, 12 e 15 grupos também foi
concluída e está em
[`RESULTADOS_QUANTIDADE_GRUPOS.md`](RESULTADOS_QUANTIDADE_GRUPOS.md). As
próximas etapas são:

1. Repetir o teste com pequenas variações de `m` próximas de 1,2 e oito grupos,
   sem aceitar
   configurações que colapsem centroides ou ocupem poucos grupos.
2. Medir a influência da cor: comparar o resultado atual com imagens em escala
   de cinza ou com uma representação que reduza a informação cromática.
3. Reduzir a influência do cenário por recorte da região central ou separação
   aproximada entre flor e fundo, e então comparar a estabilidade.
4. Só depois de definir critérios mínimos de qualidade, extrair e agrupar as
   8.189 imagens. Uma configuração candidata deve convergir em várias sementes,
   preservar todos os grupos, evitar centroides quase coincidentes e produzir
   galerias visualmente defensáveis.

