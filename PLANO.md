# Plano de implementação

1. **Inventário.** Ordenar os caminhos dos JPEGs, validar a leitura e registrar exclusões. Sortear 500 imagens válidas com uma semente fixa para o piloto.
2. **Extração de características.** Usar ResNet-50 pré-treinada para gerar um vetor por imagem antes da camada de classificação. Registrar pesos e pré-processamento e salvar os vetores junto aos caminhos relativos.
3. **Preparação.** Comparar vetores originais com PCA de 64 e 128 componentes. Aplicar normalização L2 após a transformação em cada configuração.
4. **Piloto.** Testar o fluxo completo com a amostra e um número pequeno de grupos. Conferir convergência, alinhamento entre vetores e arquivos, e soma das pertinências por imagem.
5. **Experimento completo.** Extrair os vetores das imagens válidas uma vez. Avaliar FCM com 10, 25, 50 e 102 grupos e expoente de fuzzificação de 1,5, 2,0 e 2,5, com múltiplas sementes. Tratar 102 como hipótese experimental. Rodar K-Means com as mesmas representações e quantidades de grupos como referência.
6. **Avaliação.** Comparar o índice Xie–Beni e a estabilidade entre sementes após alinhar os clusters. Usar silhueta nas atribuições dominantes como medida complementar. Inspecionar imagens típicas e de fronteira para avaliar o efeito de fundo, iluminação, cor e formato.
7. **Exportação.** Salvar centroides, configuração e métricas, além de um CSV com as pertinências completas. Exportar os dois grupos mais fortes por imagem e a diferença entre seus graus. Gerar galerias com exemplos representativos e ambíguos.

O experimento termina quando cada imagem válida tem uma linha de pertinências, cada exclusão tem um motivo registrado e a configuração selecionada pode ser reproduzida.
