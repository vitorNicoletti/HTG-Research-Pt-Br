# 16 Roteiro do artigo

Para quem escreve o artigo. Diz o que entra em cada seção, com qual número e
qual figura, onde está o detalhe no repositório e o que fica de fora. O
repositório guarda a história inteira, com tentativas abandonadas, correções
de ambiente e uma placa de vídeo com defeito. O artigo conta só o caminho que
levou ao modelo final e as provas de que cada peça é necessária.

Todos os números são da validação. O split de teste do vocabulário nunca foi
aberto (ver "Decisões em aberto", no fim).

## A tese em três frases

1. Modelos de geração de escrita manuscrita treinados em inglês não desenham
   os diacríticos do português, e as métricas usuais (FID, KID, CER) não
   mostram isso.
2. Com dados sintéticos e três mudanças no fine-tune do DiffusionPen, o modelo
   passa a desenhar o sinal pedido na letra certa sem estragar a escrita das
   palavras sem acento.
3. Uma métrica geométrica simples, validada contra anotação humana, mede isso
   por sinal e por letra.

## Seções

### 1. Introdução

- O problema e a motivação: português tem `ã ç é ê ó õ ú`, e o IAM não.
- Figura: `resultados/figuras/artigo/original_e_melhor` (o modelo original e o
  final nas mesmas palavras, escritores e semente).
- Detalhe: [01](01_problema_e_contexto.md).

### 2. Trabalhos relacionados

- DiffusionPen, o modelo de partida.
- Destilação com professor em modelos de difusão (Masip et al.), base do
  professor.
- DiffusionPen retreinado em ucraniano (Ahitoliev e Berezin), em que o
  apóstrofo, marca pequena e rara, é a falha mais clara. É o trabalho mais
  próximo.
- Lista com os resumos em [14](14_acento_separado_e_professor.md) e
  [15](15_acento_a_mais_e_variantes_do_professor.md), seção "Trabalhos
  relacionados".

### 3. Dados

- IAM, a base do modelo de partida, com os splits do DiffusionPen. Os 20
  escritores de avaliação vêm do `iam_test` e nenhum está no treino.
- Vocabulário português (FrequencyWords, a partir do OpenSubtitles 2018),
  dividido em treino, validação e teste por grupo de palavra, e congelado.
  Detalhe em `vocabulario_pt/LEIAME.md`.
- BRESSAY, em um parágrafo, como resultado negativo. Escrita real em
  português, mas as palavras têm cerca de 26 px de altura e o acento ocupa 1 ou
  2 px; o fine-tune piorou tudo e não aprendeu acento. É o que motiva a base
  sintética. Detalhe em [05](05_finetune_bressay.md); figura
  `resultados/figuras/02_bressay_40ep.png`.
- Base sintética `iam_pt_alinhado`: palavras portuguesas geradas pelo próprio
  DiffusionPen, com o acento desenhado sobre a letra certa e um par idêntico
  sem o sinal. Figura `resultados/figuras/03_base_sintetica.png`. Detalhe em
  [06](06_acentos_sinteticos.md).
- O alinhador, que diz onde cada letra está para desenhar o acento (ver
  "Instrumentos", abaixo).

### 4. Método do treino

O DiffusionPen pré-treinado no IAM, fine-tune só do UNet, lr 2e-5, lote 32.
Três mudanças, cada uma com a sua prova:

| mudança | o que faz | prova | detalhe |
|---|---|---|---|
| acento separado | o CANINE lê a palavra sem acento e o sinal entra por um vetor somado na posição da letra | marca em palavra sem acento cai de 32% (peso 5) para 11% | [14](14_acento_separado_e_professor.md) |
| professor | nas amostras sem diacrítico, o alvo é a previsão do modelo original congelado | CER das palavras sem acento pedido volta de 0,28 para 0,19, o nível do original | [14](14_acento_separado_e_professor.md) |
| peso 5 no acento e na zona das vogais | o erro de pôr acento a mais custa o mesmo que o de não pôr | palavras corretas pela métrica, 16 épocas: 44% com professor, 55% com a zona | [15](15_acento_a_mais_e_variantes_do_professor.md) |

As variantes que não funcionaram (professor fora da máscara, pares sem
professor, só professor sem acento separado) aparecem como ablação, na figura
`categorias_por_modelo`. Elas mostram que cada peça faz falta.

Uma frase de reprodutibilidade sobre o hardware: as 16 primeiras épocas
rodaram numa NVIDIA T4 (Google Colab) e as épocas 17 a 40 numa AMD RX 9060 XT,
retomando do estado salvo, com o mesmo código e os mesmos parâmetros. Detalhe
em [09](09_resultados_consolidados.md), seção "Onde foi treinado".

### 5. Métrica

- Por que uma métrica própria, em uma ou duas frases: os HTRs prontos
  (TrOCR, EasyOCR, PaddleOCR) não leem a escrita gerada, e o alfabeto do IAM
  não tem acentos.
- E1, presença e letra. Acento é uma marca solta inteira acima do corpo da
  palavra. Cedilha é tinta abaixo da linha de base, na coluna do c. A marca vai
  para a vogal sob ela, pelas colunas do alinhador. Cada palavra sai como
  correto, a mais, letra errada ou ausente.
- E2, integridade. A palavra acentuada e o seu esqueleto (mesmo escritor,
  mesma semente) são lidos pelo leitor com os sinais apagados. Degradada se a
  diferença de CER passa de 0,2. Cruzado com o E1, dá as quatro categorias do
  planejamento: acerto, omissão, degradação com e sem diacrítico.
- Validação: dois anotadores, com desempate, em 212 imagens do modelo de 40
  épocas. Marca em cima com kappa 0,82, cedilha com 0,91, presença com
  precisão 0,97 e recall 0,90. Categoria da palavra com kappa 0,33.
- Figuras: `como_a_metrica_olha` e `por_classe`; a folha de anotação em
  `resultados/figuras/07_folha_de_anotacao.png`.
- Detalhe: [avaliacao_diacriticos/README.md](../avaliacao_diacriticos/README.md)
  e [resultados/README.md](../resultados/README.md), seções 7 e 8.

#### Instrumentos

Duas redes auxiliares, treinadas pela equipe no IAM com o mesmo código
(`scripts/treinar_alinhador.py`). Não são modelos publicados; usam a perda CTC
e a receita CRNN da literatura. Cada uma merece um parágrafo curto no método.

| | alinhador | leitor |
|---|---|---|
| o que faz | acha onde cada letra está na horizontal, por alinhamento forçado | lê a palavra (HTR) |
| rede | convoluções e CTC, sem LSTM, para cada letra disparar perto de onde está | as mesmas convoluções, 3 camadas Transformer e CTC |
| treino | `iam_training`, 47.981 palavras, 15 épocas, validação no `iam_val` | o mesmo |
| validação | CER 0,127, mas acerta a letra em 98,0% de 2.969 letras de val e teste com posição conhecida (letras soltas, o caso fácil) | CER 0,084, 76% das palavras certas no `iam_val` |
| uso | desenhar o acento na base sintética; dar a letra de cada marca no E1 | o E2 |
| detalhe | [06](06_acentos_sinteticos.md), seções 4a e "Avaliação da localização das letras" | [08](08_protocolo_de_avaliacao.md), seção 3 |

O leitor é separado do alinhador para a avaliação não ser circular, já que o
alinhador ajudou a montar a base.

### 6. Resultados

| figura | mostra |
|---|---|
| `categorias_por_modelo` | E1 por modelo; o final tem 72% das palavras acentuadas corretas |
| `quatro_categorias` | E1 cruzado com E2 |
| `evolucao_epocas` | o treino melhora até 30 épocas e estaciona em 40 |
| `por_classe` | por classe de sinal, pessoas contra métrica |

Tabelas em [09](09_resultados_consolidados.md), seção "Modelo final". As
figuras ficam em `resultados/figuras/artigo/`, com PNG e PDF, refeitas por
`scripts/figuras_artigo.py`.

### 7. Limitações

- A letra das palavras acentuadas continua pior que a das sem acento (CER
  0,39 contra 0,21), e o E2 marca 41% das palavras acentuadas como degradadas.
- Acento a mais nas palavras com acento em cima: 15% pelo gabarito humano.
- A métrica é mais severa que as pessoas (75% de corretas contra 87%),
  sobretudo no til. Não julga o tipo do sinal. Deixa de fora, no acento em
  cima, as palavras com i ou j, porque o pingo parece acento.
- O E2 não tem validação humana direta.
- Escritores e estilo do IAM só; nada foi avaliado em escrita de brasileiros.

## O que fica de fora

| assunto | por quê | onde está |
|---|---|---|
| métrica inicial (limiares do E1 antigo, eixo de diferenças, TrOCR como E2) | abandonada; basta a frase sobre os HTRs prontos | [04](04_sonda_e_metrica_inicial.md), "Histórico" no README da métrica |
| smoke test e detalhes da sonda | a sonda entra só como motivação, numa frase ou na figura | `sonda/`, [04](04_sonda_e_metrica_inicial.md) |
| correções do código do DiffusionPen e do leitor de dados | engenharia, não muda o método | [02](02_diffusionpen_e_modificacoes.md) |
| a RX 6600 XT com defeito | nenhum modelo dela é usado | [03](03_ambiente_e_hardware.md) |
| fases 4 a 9 em detalhe (base sobre o IAM real, peso 5, zona vazia) | caminho até o acento separado; no máximo uma frase | [07](07_experimentos_com_acentos.md) |
| investigação com testes pequenos na CPU | levou às mudanças, mas o artigo mostra o resultado delas | [13](13_investigacao_do_vazamento_e_da_letra.md) |

## Cuidados

- A palavra de validação "rapé" não aparece em figuras, tabelas nem exemplos.
- Os nomes dos modelos nas figuras são os dos docs. Se o artigo usar outros,
  trocar `ROTULOS` em `scripts/figuras_artigo.py` e rodar de novo.
- Em `quatro_categorias`, "com diacrítico" quer dizer sinal presente, sem
  olhar a letra. O acerto dali não é subconjunto do correto do E1.
- A diferença pareada do protocolo antigo conta a cedilha solta como marca.
  Para acento a mais, usar só a marca em cima, ou a métrica final.

## Decisões em aberto para a equipe

- Rodar o protocolo uma vez no split de teste, com o modelo final, para os
  números do artigo. Até hoje tudo é validação, e o teste foi guardado para
  isso.
- Se o E2 terá validação humana (umas 100 duplas lado a lado bastariam).

## Referências

Conferidas: Cohen; Landis e Koch; FrequencyWords; OpenSubtitles 2018 (Lison e
Tiedemann); Masip et al.; Ahitoliev e Berezin. As demais devem ser conferidas
antes de entrar no artigo.

- Nikolaidou, Retsinas, Sfikas e Liwicki. DiffusionPen: Towards Controlling
  the Style of Handwritten Text Generation. ECCV 2024.
- Marti e Bunke. The IAM-database: an English sentence database for offline
  handwriting recognition. IJDAR, 2002.
- BRESSAY, base de escrita manuscrita em português brasileiro (ICDAR 2024).
  Conferir autores e título.
- Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models.
  CVPR 2022. (VAE do Stable Diffusion.)
- Song, Meng e Ermon. Denoising Diffusion Implicit Models. ICLR 2021. (DDIM,
  usado na geração.)
- Clark, Garrette, Turc e Wieting. CANINE: Pre-training an Efficient
  Tokenization-Free Encoder for Language Representation. TACL 2022.
- Graves, Fernández, Gomez e Schmidhuber. Connectionist Temporal
  Classification. ICML 2006.
- Shi, Bai e Yao. An End-to-End Trainable Neural Network for Image-based
  Sequence Recognition (CRNN). IEEE TPAMI, 2017.
- Vaswani et al. Attention Is All You Need. NeurIPS 2017. (Cabeça do leitor.)
- Li et al. TrOCR. AAAI 2023.
- Cohen. A Coefficient of Agreement for Nominal Scales. 1960.
- Landis e Koch. The Measurement of Observer Agreement for Categorical Data.
  Biometrics, 1977.
- Masip, Rodríguez, Tuytelaars e van de Ven. Continual Learning of Diffusion
  Models with Generative Distillation. CoLLAs 2024, arXiv 2311.14028.
- Ahitoliev e Berezin. Diffusion-Based Ukrainian Handwritten Text Generation
  with Cross-Domain Style Transfer. arXiv 2605.27487.
