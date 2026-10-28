# Documentação do projeto

Fidelidade de diacríticos na geração de escrita manuscrita (HTG) em português
brasileiro. Projeto de TCC, Ciência da Computação, PUCPR.

Esta pasta é a documentação organizada do projeto. Ela explica o problema, o
modelo usado, tudo o que foi tentado, os resultados de cada tentativa, os
motivos de cada escolha e os problemas que continuam abertos.

Os arquivos `LOG.md` (diário datado) e `ACHADOS.md` (achados na ordem em que
apareceram), na raiz, são o registro histórico bruto. Esta pasta reorganiza o
mesmo conteúdo por assunto.

---

## Resumo executivo (leia isto primeiro)

**O problema.** Modelos de geração de escrita manuscrita são treinados em
inglês (base IAM). Quando pedimos uma palavra com acento do português
("coração", "você", "pão"), o modelo omite o acento ou deforma a palavra. As
métricas usuais (FID, CER médio) não enxergam isso.

**O modelo.** Usamos o DiffusionPen (difusão latente condicionada por texto e
por estilo de escritor), com os pesos públicos treinados no IAM.

**O que já foi feito, em ordem:**

| fase | o que foi | resultado em uma linha |
|---|---|---|
| 1. Sonda | gerar palavras acentuadas com o modelo original | omite ou deforma o acento; o caractere chega ao modelo, mas ele não sabe desenhar |
| 2. Métrica E1/E2 | medir presença de acento e legibilidade | E1 fraco em dado real (AUC 0,68); o TrOCR não lê este corpus |
| 3. Fine-tune no BRESSAY | treinar em escrita real em português | piorou tudo: as palavras do BRESSAY têm ~26 px de altura, e o acento ocupa 1–2 px |
| 4. Acentos sintéticos | desenhar acentos em palavras reais do IAM | **o acento passa a aparecer**, mas também onde não deve |
| 5. Base portuguesa gerada | palavras portuguesas geradas pelo modelo e acentuadas | acento condicionado ao texto, mas fraco; a letra borra |
| 6. Peso no acento na loss | cobrar 5× mais o erro na região do acento | **+7 pontos** de acento certo, efeito causal; mais vazamento |
| 7. Controles | isolar a causa da piora da letra | o fine-tune em si não estraga a letra; os dados com acento sintético estragam |
| 8. Volta ao IAM real + pesos | evitar imagens geradas | traço nítido de novo, mas letra deformada e acento por toda parte |
| 9. Penalidade nas vogais | cobrar acento falso só onde acento caberia | −6,7 pontos de acento falso; a letra não volta |
| 10. Acento separado e professor | o sinal entra por um vetor próprio; o modelo original congelado vira alvo nas palavras sem acento | acento indevido e letra das palavras sem acento voltam ao nível do original |
| 11. Zona vogais com professor, 40 épocas | o mesmo peso para errar para mais e para menos | o modelo final: 72% das palavras acentuadas corretas pela métrica |
| 12. Métrica final | presença e letra do sinal (E1) e integridade da palavra (E2), validadas contra duas pessoas | marca em cima kappa 0,82, cedilha 0,91 |

**Onde estamos.** O modelo final desenha o sinal pedido e, na maior parte das
vezes, na letra certa: 72% das palavras acentuadas saem corretas pela métrica
final. O modelo original quase nunca desenha o sinal: põe marca em 6% das
palavras acentuadas, o mesmo que nos esqueletos. Nas palavras sem acento a
letra e a taxa de marca indevida ficam no nível do original (CER 0,19, marca
em 4 a 6%). O que continua aberto:
1. a letra das palavras acentuadas é pior que a das sem acento (CER 0,39
   contra 0,21);
2. ainda há acento a mais em parte das palavras com acento em cima;
3. a métrica não julga o tipo do sinal, e o E2 não tem validação humana.

O resumo com todas as tabelas está em
[09_resultados_consolidados.md](09_resultados_consolidados.md), e o roteiro
para o artigo em [16_roteiro_do_artigo.md](16_roteiro_do_artigo.md). Os
documentos 07 e 10 descrevem as fases anteriores ao acento separado.

---

## Ordem de leitura

| arquivo | conteúdo |
|---|---|
| [01_problema_e_contexto.md](01_problema_e_contexto.md) | o problema, as perguntas de pesquisa, as bases de dados e os termos usados |
| [02_diffusionpen_e_modificacoes.md](02_diffusionpen_e_modificacoes.md) | como o DiffusionPen funciona e **todas as mudanças que fizemos no treino oficial** |
| [03_ambiente_e_hardware.md](03_ambiente_e_hardware.md) | máquinas, GPUs, o defeito da RX 6600 XT, a RX 9060 XT, WSL e armadilhas |
| [04_sonda_e_metrica_inicial.md](04_sonda_e_metrica_inicial.md) | fases 1–2: a sonda do modelo original e a métrica E1/E2 |
| [05_finetune_bressay.md](05_finetune_bressay.md) | fase 3: fine-tune no BRESSAY e por que não funcionou |
| [05b_pre_processamento_bressay.md](05b_pre_processamento_bressay.md) | detalhe do pré-processamento v1/v2 do BRESSAY |
| [06_acentos_sinteticos.md](06_acentos_sinteticos.md) | o gerador de acentos sintéticos, etapa por etapa (fluxograma, alinhamento CTC, desenho) |
| [07_experimentos_com_acentos.md](07_experimentos_com_acentos.md) | fases 4–9: cada fine-tune com acentos, motivo, mudança, resultado e conclusão |
| [08_protocolo_de_avaliacao.md](08_protocolo_de_avaliacao.md) | como avaliamos: palavras, escritores, anti-vazamento, leitor, detector de marcas, posição, estatística |
| [09_resultados_consolidados.md](09_resultados_consolidados.md) | o modelo final, a métrica final e todas as tabelas de resultado, com links para as figuras |
| [10_problemas_em_aberto.md](10_problemas_em_aberto.md) | os problemas como estavam ao fim da fase 9, com hipóteses e testes; o estado atual está no 15 |
| [11_como_rodar.md](11_como_rodar.md) | guia prático: ambiente, dados, comandos de cada etapa, onde ficam as saídas |
| [12_mapa_do_repositorio.md](12_mapa_do_repositorio.md) | o que é cada pasta e cada script |
| [13_investigacao_do_vazamento_e_da_letra.md](13_investigacao_do_vazamento_e_da_letra.md) | testes de geração sem treino e treinos curtos por ingrediente |
| [14_acento_separado_e_professor.md](14_acento_separado_e_professor.md) | acento separado no condicionamento e professor na loss, com os resultados na GPU e como rodar |
| [15_acento_a_mais_e_variantes_do_professor.md](15_acento_a_mais_e_variantes_do_professor.md) | o que sobrou dentro da palavra acentuada, as variantes testadas do professor, o treino fora da máquina local e os trabalhos relacionados |
| [16_roteiro_do_artigo.md](16_roteiro_do_artigo.md) | o que entra no artigo, com qual número e figura, o que fica de fora e as referências |
