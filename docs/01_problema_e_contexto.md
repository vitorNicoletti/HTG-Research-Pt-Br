# 01 — O problema e o contexto

## O que é HTG

*Handwritten Text Generation* (HTG) é gerar a imagem de uma palavra
manuscrita a partir de dois pedidos:

- **o conteúdo:** o texto a escrever, por exemplo "coração";
- **o estilo:** a caligrafia de um escritor, dada por algumas imagens de
  referência escritas por ele.

A saída é uma imagem que parece a palavra pedida escrita à mão por aquela
pessoa. Os usos típicos são aumentar dados de treino para reconhecimento de
manuscrito (HTR), personalizar fontes e estudar caligrafia.

## O problema estudado

Quase todo modelo de HTG é treinado na base **IAM**, em inglês. O inglês não
tem diacríticos. O português tem 12 letras acentuadas e a cedilha:

| sinal | letras | posição na imagem |
|---|---|---|
| agudo | á é í ó ú | acima da vogal |
| circunflexo | â ê ô | acima da vogal |
| til | ã õ | acima da vogal |
| grave | à | acima da vogal |
| cedilha | ç | abaixo do c, encostada nele |

Quando se pede "coração" a um modelo treinado no IAM, ele não sabe desenhar o
til nem a cedilha. As métricas usuais de HTG (FID, KID e o CER médio de um
reconhecedor) **não enxergam essa falha**:
- **FID e KID** comparam distribuições de imagens inteiras. Um sinal de poucos
  pixels quase não muda a distribuição.
- **O CER médio** dilui o erro: numa palavra de 7 letras, faltar o til é 1
  erro em 7. Muitos reconhecedores nem têm acentos no alfabeto.

Por isso o trabalho tem três frentes:
1. **Sonda:** medir como o modelo original falha nos diacríticos.
2. **Fine-tune:** ensinar o modelo a desenhá-los.
3. **Métrica:** criar uma medida específica para diacríticos, que as métricas
   usuais não oferecem.

## O modelo escolhido: DiffusionPen

Modelo de difusão latente para HTG, com código e pesos públicos treinados no
IAM ([github.com/koninik/DiffusionPen](https://github.com/koninik/DiffusionPen)).
O funcionamento está em
[02_diffusionpen_e_modificacoes.md](02_diffusionpen_e_modificacoes.md).

Motivos da escolha:
- **Pesos públicos** do IAM, o que dispensa treinar do zero (dias de GPU).
- **O texto entra pelo CANINE,** um codificador que trabalha com o número
  Unicode de cada caractere. Por isso o modelo **aceita** "ã" sem erro: o
  caractere chega até a rede. O problema é só a rede não ter aprendido a
  desenhá-lo, e é exatamente isso que queremos estudar e corrigir.
- **Um alternativo descartado na prática:** o VATr++ (análise estática em
  `ACHADOS.md`, seções 3 e 4). Ele **descarta em silêncio** qualquer caractere
  fora do alfabeto do IAM (`mão` vira `mo`) e exige PyTorch 1.13, sem suporte
  às GPUs AMD usadas. Uma comparação ingênua registraria "sucesso" quando o
  acento nem chegou ao modelo.

## As bases de dados

### IAM (inglês, escrita real)
- Recortes de palavras manuscritas de centenas de escritores. Os splits do
  DiffusionPen estão em `DiffusionPen/utils/splits_words/`.
- **`iam_train_val.txt`:** 55.535 palavras, 339 escritores. É o treino do
  modelo original e também o nosso.
- **`iam_test.txt`:** escritores que o modelo nunca viu no treino. Nas nossas
  avaliações, o estilo vem só desses escritores.

### BRESSAY (português, escrita real)
- Redações de estudantes brasileiros: 1.000 páginas e 416.826 recortes de
  palavras.
- As imagens vieram de várias plataformas, sem padrão de captura. **A altura
  mediana de uma palavra é 26 px**; no IAM é ~50 px. Esse é o motivo principal
  de o fine-tune nele ter falhado (fase 3).
- Nosso split, em `bressay_split/`, separa escritores entre treino, validação
  e teste: 647 escritores e 74.882 palavras de treino.

### Vocabulário português (`vocabulario_pt/`)
- 48.658 palavras da lista de frequência FrequencyWords (OpenSubtitles 2018,
  CC-BY-SA 4.0).
- Particionado **por grupo**: o esqueleto sem acento com o plural dobrado.
  "nação", "nações" e "nacao" caem sempre no mesmo split, e uma palavra do
  teste nunca aparece no treino só com outro acento.
- Treino: 4.952 acentuadas e 35.248 sem acento. Validação: 532 e 2.105.
  Teste: 1.536 e 4.285.
- Toda palavra que o repositório usa para avaliar foi **forçada** para o
  teste ou a validação.
- O arquivo foi congelado antes de gerar qualquer imagem. Os scripts conferem
  cada rótulo contra ele e param com erro se houver vazamento.

## Termos usados em toda a documentação

| termo | significado |
|---|---|
| **diacrítico / acento** | qualquer um dos sinais da tabela acima, cedilha incluída |
| **esqueleto** | a palavra sem acentos: "coração" → "coracao" |
| **par mínimo** | a mesma palavra com e sem acento ("nação"/"nacao"), gerada com o mesmo estilo e a mesma semente: a única diferença é o acento |
| **corpo da palavra** | a faixa vertical das letras baixas (a, c, e, o…) |
| **altura-x** | a altura do corpo, a altura de um "x" |
| **linha de base** | onde as letras "apoiam" |
| **haste / perna** | a parte de b, d, h, l, t que sobe acima do corpo, e a de g, p, q, y que desce abaixo |
| **marca solta** | um traço pequeno, fora do corpo e sem tocá-lo: é como detectamos acento na imagem gerada |
| **vazamento** | o modelo desenhar acento numa palavra que não pede acento |
| **diferença pareada** | marca na palavra acentuada menos marca no seu esqueleto, com o mesmo escritor e a mesma semente: mede se o acento depende do texto |
| **CER** | *character error rate*: distância de edição entre o que um leitor automático leu e o texto esperado, dividida pelo tamanho do texto |
| **latente** | a representação comprimida da imagem usada pelo modelo: 4×8×32 para uma imagem de 64×256 |
| **EMA** | média móvel exponencial dos pesos, a versão "suavizada" do modelo usada para gerar |
| **época / bloco** | uma passada por todos os dados de treino; os nossos treinos rodam em blocos de algumas épocas, com amostras e checkpoint no fim de cada bloco |
| **validação / teste** | a validação escolhe entre modelos; o teste só é aberto no fim, uma vez. **O teste ainda não foi aberto.** |
