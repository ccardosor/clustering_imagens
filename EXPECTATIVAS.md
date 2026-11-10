# Resultados esperados

- **Grupos visualmente reconhecíveis.** Esperamos encontrar agrupamentos com cores, formatos de pétalas e estruturas florais recorrentes. Vamos conferir essa hipótese nas imagens com maior pertinência a cada cluster.
- **Sobreposição entre grupos.** Flores com atributos mistos e fotos com enquadramentos diferentes podem ter pertinências próximas nos dois grupos principais. Vamos comparar essas imagens com os exemplos típicos dos dois grupos e verificar se a ambiguidade se repete entre sementes.
- **Influência da fotografia.** Esperamos que alguns grupos sejam explicados por fundo, iluminação ou escala da flor na imagem. A inspeção das galerias vai ajudar a separar esses efeitos das características da flor.
- **Correspondência parcial com espécies.** Espécies parecidas podem aparecer no mesmo grupo, enquanto fotos da mesma espécie podem se separar por cor, estágio de abertura ou ponto de vista. A pasta local não inclui rótulos para medir essa correspondência. Vamos tratar 102 clusters como uma das configurações experimentais.

O ganho esperado sobre o K-Means é poder distinguir imagens típicas de imagens de fronteira. Se as pertinências ficarem quase uniformes ou se os grupos mudarem muito entre sementes, teremos pouca evidência para interpretar essas fronteiras.
