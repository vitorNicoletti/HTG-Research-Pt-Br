# 10 — Problemas em aberto

Onde estamos travados. Para cada problema:
- o que se observa;
- as hipóteses, com a evidência a favor e contra;
- os testes que separam as hipóteses.

Os números vêm de [09_resultados_consolidados.md](09_resultados_consolidados.md).

---

## Problema 1 — A letra piora sempre que dados com acento sintético entram no treino

### O que se observa

| treino | CER sem acento | CER sem acento pedido e sem marca |
|---|---|---|
| IAM original | 0,22 | 0,19 |
| fine-tune **só com originais do IAM** | 0,21 | 0,18 |
| base pt gerada (qualquer variante) | 0,32–0,35 | 0,27–0,28 |
| base IAM real com pares e pesos | 0,46–0,47 | 0,35–0,36 |

- **O fine-tune em si não piora nada:** mesmo código, mesmo regime e mesmos
  hiperparâmetros. A piora aparece **inteira já com 4 épocas** e não aumenta
  com mais treino.
- **Mais escrita real não ajuda:** com 77% de palavras reais o CER é o mesmo,
  ou até um pouco pior.

### Hipóteses

| hipótese | a favor | contra | como testar |
|---|---|---|---|
| **H1. Imagens geradas pelo DiffusionPen** (cópia da cópia) | a base pt borra o traço (0,68 de tinta cinza contra 0,59) | a base IAM real tem traço nítido e a letra piora ainda mais | **explica a base pt, não a IAM** |
| **H2. Os traços desenhados** (o acento sintético, o apagamento do pingo do i) | a acentuada é lida pior (+0,08 de CER) | só as acentuadas têm o traço desenhado, e a piora vale também para palavras sem acento | treinar só com pares e originais, sem acentuadas |
| **H3. O rótulo "barulhento"** (acento em letra sorteada de palavra inglesa; a mesma imagem com dois rótulos, acentuada e par) | a base IAM real (acento sorteado) piora mais que a pt (acento ortográfico) | a base pt também piora | treinar a base IAM sem os pares |
| **H4. O peso 5 no acento** domina o gradiente em poucas amostras | as duas bases com pesos pioram | o controle da base pt (peso 1) já piorava igual (0,32) | **avaliar o `model_iam_acentuado_teto25`**: base IAM, sem pares e sem pesos, modelo já treinado. Custa ~35 min de GPU |
| **H5. Perda de generalização de estilo:** a base só tem escritores do treino, e a avaliação usa escritores novos | nas amostras de bloco (escritores do treino) as palavras parecem melhores | o fine-tune só com originais também só tem escritores do treino e não piora | avaliar os mesmos modelos com escritores do treino |

**Primeiro passo recomendado:** avaliar o `model_iam_acentuado_teto25` (H4)
e o `model_iam_acentuado` sem teto. É barato e separa "o dado estraga" de
"o peso estraga".

```bash
python scripts/avaliar_pt.py --split val \
    --modelo teto25_10ep=model_iam_acentuado_teto25/models/ema_bloco_10ep.pt \
    --bases iam_acentuado_teto25 --aceitar_grupos central pas \
    --leitor modelos/leitor_iam_transformer.pt --saida avaliacao_teto25_val
```

O nome exato do bloco e os grupos a aceitar devem ser conferidos antes de
rodar.

---

## Problema 2 — Acento onde o texto não pede (vazamento)

### O que se observa

O melhor modelo em acento (peso 5) põe marca em ~32–34% das palavras sem
acento; os da base IAM, em 48–56%. O IAM original fica em 5% (o ruído do
detector).

### Causas identificadas

1. **Custo assimétrico na loss (confirmado).** Com peso 5 no acento, faltar o
   acento custa 5× e sobrar custa 1× na maior parte da imagem. Na dúvida,
   desenhar é mais barato. O peso 5 aumentou o vazamento em +6,7 pp.
2. **Sinal de texto fraco.** "provável" e "provavel" ficam quase iguais
   depois do CANINE (cosseno 0,935). O UNet recebe pouca diferença entre os
   dois pedidos.
3. **Base em inglês com acento sorteado.** O rótulo não segue regra
   nenhuma, então o modelo aprende "palavra pode ter sinal" em vez de "esta
   letra tem sinal".

### O que já foi tentado

| tentativa | efeito |
|---|---|
| teto de 25% por palavra | pequeno |
| zona vazia com peso 2 | não reduziu, e apagou hastes |
| zona vogais com peso 5 (custo simétrico) | **−6,7 pp**, real mas pequeno |

### Ideias ainda não testadas

- **Treinar com *classifier-free guidance*:** às vezes sem o texto, para
  depois reforçar o texto na geração. O DiffusionPen original tem a linha
  para isso (`labels = None`), mas ela não faz nada. Exige mudar o
  `train.py` e treinar de novo.
- **Reforçar a diferença no condicionamento de texto:** por exemplo, uma
  entrada extra que marque quais letras têm acento.
- **Dados com regra ortográfica e escrita real:** um subconjunto do BRESSAY
  com resolução aceitável, ou palavras do IAM que também são palavras
  portuguesas ("pais" → "país", "nos" → "nós"). São poucas, mas corretas.

---

## Problema 3 — O acento cai na letra errada

### O que se observa

No melhor modelo (peso 5), com palavras de um único sinal:
- **agudo:** 14% na letra certa, 28% em outra letra (7% na vizinha, 21% mais
  longe) e 57% sem marca;
- **til:** 50% certo;
- **circunflexo:** quase nunca aparece.

### Hipóteses

- **O til fica quase sempre no fim da palavra (-ão)**, uma posição fácil de
  aprender. O agudo pode cair em qualquer sílaba ("módulo", "rústica",
  "haverá"), e a posição tem de vir do texto.
- **A base em inglês** põe o acento em letra sorteada, sem regra, então não
  ensina posição.
- **Poucos exemplos por palavra:** ~2 imagens por palavra na base pt.

### Ideias

Mais exemplos por palavra, com amostragem pela raiz quadrada da frequência,
e dados com regra ortográfica. O peso na máscara do acento já é localizado,
mas não foi suficiente.

---

## Problema 4 — Lacunas na medição

1. **A cedilha não é medida.** O detector só conta traços soltos, e a cedilha
   nasce encostada no c (4–7% detectadas na validação). Falta um detector,
   por exemplo tinta abaixo da linha de base na fatia do c.
2. **Não há validação humana** de nenhuma métrica. Uma anotação de ~200
   imagens (tem acento? na letra certa? palavra legível?) com kappa daria
   credibilidade aos números. `avaliacao_diacriticos/anotacao.py` serve de
   ponto de partida.
3. **O CER mistura marca e letra.** Por isso reportamos também o CER sem
   marca. Um leitor com acentos no alfabeto mediria melhor.
4. **O detector conta pedaços de palavra deformada como marca.** Parte do
   "vazamento" pode ser deformação.

---

## Decisões de projeto pendentes

- **A narrativa do TCC.** O plano original era o fine-tune no BRESSAY. Hoje a
  contribuição mais forte é: a análise do BRESSAY (resolução), a base de
  acentos sintéticos com os pesos na loss (efeito causal medido) e o
  protocolo de avaliação. Falta decidir se o BRESSAY volta, por exemplo
  combinado com a base sintética.
- **Critério de sucesso** a fixar **antes** de abrir o teste. Por exemplo:
  - diferença pareada ≥ X;
  - marca falsa ≤ Y;
  - CER ≤ Z.
- **O teste nunca foi aberto.** Deve ser rodado uma vez, no modelo final, com
  `avaliar_pt.py --split teste`.

---

## Próximos passos sugeridos, em ordem de custo

| passo | custo | responde |
|---|---|---|
| 1. Avaliar `model_iam_acentuado_teto25` e `model_iam_acentuado` pelo protocolo | ~35 min de GPU cada | o peso estraga a letra? (H4) |
| 2. Avaliar 2–3 modelos com escritores do **treino** | ~35 min cada | é perda de generalização de estilo? (H5) |
| 3. Detector de cedilha + anotação humana de ~200 imagens | CPU e trabalho manual | a medição é confiável? |
| 4. Treino da base IAM **sem pares** e sem pesos, mais um com pares e sem pesos | ~3,5 h cada | H2 × H3 |
| 5. Treino com *classifier-free guidance* (texto descartado em ~10% dos passos) | mudança no `train.py` + ~3,5 h | dá para reforçar o texto na geração? |
| 6. Dados com regra ortográfica (BRESSAY de boa resolução, palavras PT dentro do IAM) | dias | posição do acento e vazamento |
