# Por que usar Fuzzy C-Means?

Fotos de flores podem compartilhar características de vários grupos. Uma imagem pode ter a cor típica de um grupo e o formato das pétalas de outro. O Fuzzy C-Means (FCM) representa essa sobreposição atribuindo um grau de pertinência entre 0 e 1 a cada cluster. Os graus somam 1 para cada imagem e expressam afinidade relativa aos centroides, sem interpretação de probabilidade de espécie.

O K-Means oferece uma referência com atribuição única por imagem. Compará-lo com o FCM sobre os mesmos vetores ajuda a avaliar se os graus de pertinência acrescentam informação útil à análise. Por exemplo, pertinências de 0,44 e 0,41 para os dois primeiros grupos indicam uma fronteira que vale inspecionar visualmente.

O FCM depende da representação, da escala das características, da inicialização e da quantidade de grupos. A distância euclidiana favorece grupos compactos. Para verificar se os resultados capturam características das flores, vamos comparar sementes e inspecionar exemplos representativos e ambíguos, observando a influência de fundo e iluminação.
