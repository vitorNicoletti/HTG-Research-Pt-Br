# 08 — Protocolo de avaliação

Como medimos se um modelo desenha o acento, se desenha **só quando deve**, se
o desenha **na letra certa** e se a palavra continua legível. Também cobre
como evitamos vazamento entre treino e avaliação e como comparamos modelos
estatisticamente.

## Visão geral

```
modelo ──> avaliar_pt.py ──> 3.200 imagens + paineis.tsv + resumo.md
                                   │
                                   ├── comparar_avaliacoes.py   tabela + bootstrap por palavra
                                   ├── medir_posicao_acento.py  o acento caiu na letra certa?
                                   ├── comparar_paineis.py      figuras lado a lado
                                   └── diagnostico/nitidez.py   o traço está borrado?
```

## 1. A geração (`scripts/avaliar_pt.py`)

### Que palavras

Do vocabulário português (`vocabulario_pt/`), **só do split pedido**: `val`
para escolher entre modelos, `teste` uma única vez no fim.

| tipo | quantas | exemplo |
|---|---|---|
| acentuada | 30 | "prestação" |
| esqueleto (a mesma sem acento) | 30 | "prestacao" |
| sem acento (palavra que não tem acento) | 20 | "tarde" |

As palavras são sorteadas com semente fixa (0), com 3 a 10 letras, **sem "i"
nem "j"**: o pingo seria contado como marca.

**Por que o esqueleto.** Ele forma o **par mínimo**: mesma palavra, mesmo
escritor, mesma semente. A única diferença é o acento no texto. Se o modelo
desenha marca nos dois, ele não está respondendo ao texto.

**Por que as palavras sem acento.** Elas medem o vazamento em palavras que
nunca têm acento.

### Que escritores e sementes

- **20 escritores do `iam_test`**, sorteados com semente 0. Nenhum aparece no
  treino, então estilos nunca vistos.
- **2 sementes de ruído** (42 e 43).
- Cada palavra: 20 × 2 = **40 imagens**. Cada modelo: 80 × 40 = **3.200
  imagens**.
- **A mesma semente e as mesmas referências** para todas as palavras e todos
  os modelos. Por isso dá para comparar imagem a imagem.

### Travas contra vazamento (o script para com erro)

1. Toda palavra avaliada é do split pedido.
2. **Nenhuma palavra avaliada aparece, pelo grupo, nos `split.txt` das bases
   de treino** (`--bases`). O grupo é o esqueleto com o plural dobrado.
   Exceções só com `--aceitar_grupos`, conferidas à mão e registradas no
   `resumo.json`. Isso aconteceu uma vez: "central" e "pás" na base em inglês,
   que tem a palavra real inglesa "central" e "pass"/"pãss". Excluir as três
   palavras muda os números em menos de 0,01.
3. Os escritores de avaliação não estão no treino.

### Saídas

Em `avaliacao_<nome>/`:
- `paineis/<modelo>/<k>_<escritor>_<semente>.png`, onde `k` é o índice da
  palavra em `resumo.json["palavras"]`: as 30 acentuadas, depois os 30
  esqueletos, depois as 20 sem acento;
- `paineis.tsv`: uma linha por imagem, com modelo, palavra, tipo, escritor,
  semente, marca, leitura e CER;
- `resumo.md` e `resumo.json`.

## 2. O detector de marcas (`scripts/medir_marcas.py`)

**O que mede:** se há uma **marca solta** acima ou abaixo do corpo da palavra.

1. **Corpo da palavra** (`acentos_sinteticos/geometria.analisar`):
   - a tinta é separada por Otsu;
   - o corpo é a faixa contínua em torno da linha horizontal com mais tinta,
     crescendo enquanto as linhas têm pelo menos uma fração da tinta dela;
   - isso dá o topo da altura-x e a linha de base.
2. **Marca:** um componente de tinta que:
   - termina acima de `topo_x + 0,15·altura_x` (acento) ou começa abaixo de
     `base − 0,15·altura_x` (cedilha);
   - tem área de pelo menos 12 px e altura menor que a altura-x. Mais alto
     que isso é haste de letra.

**Ruído do detector:** 5% no IAM original, que não desenha acento.

**Limitações conhecidas:**
- **Não detecta a cedilha.** A cedilha que desenhamos nasce **encostada** no
  c, então não é um componente solto. Na validação nas bases, só 4–7% das
  cedilhas desenhadas são achadas. **Os números de "marca" das palavras com ç
  vêm quase todos do til do -ção.** Falta um detector próprio.
- **Palavra deformada se parte em pedaços,** e um pedaço solto conta como
  marca. Parte do "vazamento" pode ser deformação.
- **Não sabe onde a marca caiu:** "hávera" conta como acerto para "haverá". A
  métrica de posição (seção 4) resolve isso.

## 3. O leitor independente (CER)

- **Modelo:** reconhecedor CTC com cabeça Transformer, treinado no IAM
  (`scripts/treinar_alinhador.py --cabeca transformer` →
  `modelos/leitor_iam_transformer.pt`, CER 0,084 na validação do IAM).
- **Por que um leitor próprio:** o TrOCR não lê este corpus (fase 2).
- **Por que separado do alinhador:** o alinhador (`modelos/alinhador_iam.pt`,
  cabeça convolucional) filtrou e posicionou os acentos da base. Usar o mesmo
  modelo para avaliar seria circular.
- **Como lê:** o recorte justo da tinta, com leitura gulosa. O CER é
  calculado **contra o esqueleto**, porque o alfabeto do leitor não tem
  acentos.
- **Limitação:** uma marca a mais é lida como letra a mais e sobe o CER.
  Por isso reportamos também o **CER só nas imagens sem marca**, que mede a
  letra em si.

## 4. A posição do acento (`scripts/medir_posicao_acento.py`)

1. O alinhador CTC alinha o **esqueleto** no recorte da palavra gerada e
   define a fatia de cada letra, com fronteiras ajustadas aos vales de tinta.
   É o mesmo localizador do gerador de acentos.
2. Cada marca do detector vai para a letra cuja fatia contém o centro dela.
3. Por imagem acentuada:
   - **certa:** toda letra acentuada tem marca do lado certo;
   - **parcial:** só parte delas;
   - **errada:** há marca, mas em nenhuma letra acentuada;
   - **nenhuma:** nenhuma marca.
4. Também por sinal: a fração das letras com aquele sinal que receberam marca
   do lado certo.

**Validação** nas imagens das bases, onde a posição é conhecida: o método
acha 80–94% do agudo, til e circunflexo na letra certa, e só 4–7% da cedilha
(o mesmo limite do detector).

**Não é circular:** o alinhador só diz onde fica cada letra; quem decide se há
acento é o detector.

## 5. Nitidez do traço (`diagnostico/nitidez.py`)

Entre os pixels de tinta, a fração que é cinza intermediário em vez de tinta
firme, e o gradiente médio na borda do traço. Mostrou que as bases geradas
borram o traço: 0,68 contra 0,59 do IAM original.

## 6. Estatística (`scripts/comparar_avaliacoes.py`)

**A unidade é a palavra, não a imagem.** As 40 imagens de uma palavra não são
independentes: o mesmo texto produz erros parecidos.

Para comparar dois modelos A e B:
1. calcula-se a medida **por palavra** em cada modelo;
2. toma-se a diferença A − B **palavra a palavra**, pareada;
3. reamostram-se as palavras com reposição, 10.000 vezes (bootstrap);
4. o IC95 vem dos percentis 2,5 e 97,5.

Uma diferença cujo IC95 não contém zero é tratada como real.

```bash
python scripts/comparar_avaliacoes.py \
    --modelo avaliacao_peso_val:controle_16ep --modelo avaliacao_peso_val:peso5_16ep \
    --comparar peso5_16ep controle_16ep
```

## 7. As medidas, em resumo

| medida | o que responde | bom é |
|---|---|---|
| marca na acentuada | o modelo desenha algum sinal quando pedimos? | alto |
| marca no esqueleto / sem acento | põe sinal quando **não** pedimos (vazamento)? | baixo, perto dos 5% do ruído |
| **diferença pareada** | o sinal depende do texto? | alto |
| acerto por sinal (posição) | o sinal certo cai na letra certa? | alto |
| CER sem acento pedido | a palavra está legível? | perto de 0,2 (o IAM original) |
| CER sem marca | a letra em si, descontando as marcas | perto de 0,19 |

## 8. O que ainda falta no protocolo

- **Detector de cedilha.**
- **Validação humana:** anotar uma amostra à mão e medir a concordância
  (kappa) com o detector e com a métrica de posição. A ferramenta de
  `avaliacao_diacriticos/anotacao.py` serve de ponto de partida.
- **O teste ainda não foi aberto.** Todas as escolhas foram feitas na
  validação, e o teste deve ser rodado uma única vez, no modelo final.
