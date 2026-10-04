# ACHADOS — Fase de setup e sonda exploratória

> **Status: PRELIMINAR — análise estática apenas.**
> A GPU alvo (RX 9060 XT / gfx1200) está em outra máquina, indisponível.
> Nenhum modelo foi executado. Tudo abaixo vem de leitura de código e de um
> teste de tokenização em CPU. As seções marcadas ⏳ dependem da GPU.
>
> Data: 2026-08-17

---

## 1. Ambiente

| Item | Situação |
|---|---|
| Máquina inspecionada | Samsung 550XBE (notebook), i5-8265U, 15 GiB RAM |
| GPU presente | Intel UHD 620 (iGPU) — **a RX 9060 XT não está nesta máquina** |
| SO | Arch Linux, kernel 6.19.8 (o plano previa Ubuntu/WSL2) |
| ROCm | não instalado (irrelevante sem hardware AMD) |
| PyTorch | não instalado |

**Fase 0 não pôde ser concluída.** `env/check_env.py` está escrito e pronto,
mas não foi executado.

### Wheels ROCm disponíveis (verificado em `download.pytorch.org`)

| Índice | torch |
|---|---|
| `rocm7.2` (mais recente) | 2.11.0+rocm7.2 |
| `rocm7.1` | 2.10.0+rocm7.1 |
| `rocm7.0` | 2.10.0+rocm7.0 |
| `rocm6.4` | 2.8.0+rocm6.4 |

`rocm7.2` atende o requisito de ROCm ≥ 7.0.2 do gfx1200. Wheels para cp310–cp315.

### Patches

| Patch | Status |
|---|---|
| `xformers` → SDPA (previsto no plano) | **desnecessário** — nenhum dos dois repos usa `xformers` ou `bitsandbytes` |
| `sampling_mode='sonda'` no DiffusionPen | aplicado, aditivo, `diffusionpen/patch_sonda_train.diff` |

---

## 2. Tabela de viabilidade ⏳

Depende inteiramente da GPU. Nada medido.

| Modelo | Roda? | Pico VRAM | s/imagem | 25k imagens |
|---|---|---|---|---|
| DiffusionPen | ⏳ | ⏳ | ⏳ | ⏳ |
| VATr++ | ⏳ | ⏳ | ⏳ | ⏳ |

Único dado com base sólida: o VATr++ é geração de **passada única** e o
DiffusionPen é **difusão iterativa** (DDIM). A diferença de tempo por imagem
deve ser de ordens de magnitude, o que muda qual dos dois pode gerar as 25k
localmente.

---

## 3. Achados de diacríticos — análise estática

A sonda visual não rodou. Mas a leitura de código já responde parte da SP1, e
com uma distinção que muda o desenho do experimento: **os dois modelos falham em
pontos diferentes do pipeline.**

### DiffusionPen — falha silenciosa no decoder, como previsto

| Etapa | Comportamento com `ã` |
|---|---|
| Tokenização (CANINE-C) | ✅ **aceita** — vira o codepoint 227, sem OOV |
| Normalização de acentos | ✅ **não existe** — nenhum `unidecode`/`unicodedata` |
| Charset ASCII (`letter2index`) | ⚠️ existe, mas **fora do caminho de sampling** |
| UNet | ❓ hipótese: gera algo plausível e errado |

Confirmado empiricamente (`transformers` 5.15.0, CPU):

```
nacao  → [110, 97, 99, 97, 111]     (5 tokens)
nação  → [110, 97, 231, 227, 111]   (5 tokens)
```

Os pares mínimos têm comprimento idêntico e diferem apenas nas posições do
diacrítico. **A hipótese do plano se sustenta:** o modelo vai aceitar `ã` sem
erro e a falha, se houver, será visual. Ausência de crash não é sucesso.

> Nota para a fase seguinte do TCC: `label_padding()` (`train.py:37`) faz
> `letter2index[c]` num dict ASCII-only. No **fine-tuning em português** isso
> levanta `KeyError`. É um bloqueio conhecido, ainda não um problema.

### VATr++ — falha antes do modelo ⚠️

`generate/writer.py:70`:

```python
text = "".join([c for c in text if c in self.model.args.alphabet])
```

O alfabeto padrão é o do IAM, sem diacríticos. Portanto:

| pedido | o que chega ao modelo |
|---|---|
| `mão` | `mo` |
| `coração` | `corao` |
| `março` | `maro` |
| `você` | `voc` |

Sem aviso e sem erro. **Isto não é "o modelo errou o diacrítico" — é o
caractere sendo descartado antes da geração.** Uma sonda que só verificasse
"gerou sem crash" registraria sucesso.

**Consequência metodológica:** comparar os dois modelos como estão seria
comparar coisas diferentes. O VATr++ precisa ter os caracteres portugueses
admitidos no alfabeto antes que a pergunta faça sentido para ele.

### O prior geométrico do VATr++ tem caminho concreto

`models/unifont_module.py:29-42` carrega os arquétipos de um pickle indexado por
`ord(char)` e os projeta com uma **`nn.Linear` compartilhada**. Não há embedding
por caractere.

Isso significa que **acrescentar um caractere ao alfabeto não cria parâmetros
novos** — basta o glifo existir no Unifont. O repo já usa isso: `special_alphabet`
(`util/misc.py:508`) contém o **alfabeto grego**, concatenado ao alfabeto em
`model.py:123` e `model.py:219`, e `model.py:331` o gera explicitamente para
demonstrar caracteres nunca vistos no treino.

É a prova de conceito, dentro do próprio repositório, do mecanismo que o TCC
quer testar — com grego em vez de português.

---

## 4. DiffusionPen vs. VATr++ — o contraste é mais nítido do que o plano supunha

| | DiffusionPen | VATr++ |
|---|---|---|
| Representação de conteúdo | CANINE-C (codepoint) | arquétipo visual 16×16 (Unifont) |
| Aceita `ã` na entrada? | sim | **não — descarta silenciosamente** |
| Prior de forma do glifo | nenhum | sim, o bitmap do caractere |
| Parâmetros por caractere | — | **nenhum** (projeção linear compartilhada) |
| Caminho para PT-BR | — | acrescentar ao `special_alphabet` |
| Precedente no repo | — | grego, já implementado |

A assimetria é a contribuição: o DiffusionPen sabe *qual* caractere foi pedido
mas nada sobre sua forma; o VATr++ recebe a forma pronta, mas hoje recusa o
caractere na porta de entrada. ⏳ Se o prior geométrico se traduz em til
corretamente desenhado é o que a sonda visual precisa responder.

---

## 5. Riscos e bloqueios

| Risco | Severidade | Situação |
|---|---|---|
| GPU alvo em outra máquina, indisponível | **bloqueante** | Fases 0–4 paradas |
| VATr++ fixa PyTorch 1.13.1 / CUDA 11.7 | **alto** | Não existe wheel ROCm 7.x para 1.13.1. Vai exigir PyTorch 2.x. Não forçado — regra 4 |
| Filtro silencioso do VATr++ | **alto** | Achado. Invalida sonda ingênua |
| `files/unifont.pickle` não está no repo | médio | Vem do Google Drive. Cobertura de U+00E0–U+00FA **não verificada** |
| `KeyError` em `label_padding` no fine-tuning | médio | Fase seguinte do TCC, não esta |
| SO é Arch, não Ubuntu | baixo | ROCm vem do repo `extra`, não do instalador da AMD |

---

## 6. Recomendação ⏳

Prematuro sem números de VRAM e tempo. O que já dá para dizer:

- **A questão do VATr++ deixou de ser só "roda?"** Passou a ser "como admitir os
  caracteres portugueses no alfabeto sem alterar a lógica generativa". O
  `special_alphabet` é o candidato natural, e é aditivo.
- **A primeira medição a fazer na GPU** é o tempo por imagem do DiffusionPen. É
  ele que decide se as 25k imagens do protocolo cabem localmente ou vão para
  Kaggle/Colab. O VATr++, de passada única, quase certamente cabe.

---

## Próximos passos, na ordem

1. Rodar `env/check_env.py` na máquina com a RX 9060 XT (Fase 0)
2. Baixar os artefatos do HF `konnik/DiffusionPen` e rodar `smoke_test.py` (Fase 1)
3. Rodar `sonda_diacriticos.py` + `folha_contato.py` (Fase 2) — **inspeção visual honesta**
4. `perfil_vram.py` (Fase 3) — ainda não escrito, depende de ver o modelo rodar
5. VATr++ (Fase 4), começando por decidir o tratamento do filtro de alfabeto

---

## 7. Fine-tune no BRESSAY: a resolução do dataset limita o que dá para ensinar

> Data: 2026-09-29. Máquina: RX 9060 XT (gfx1200), aprovada no `diagnostico/`
> (cosseno do gradiente GPU vs CPU 1,002). Run `model_bressay_25`, detalhes no
> `LOG.md`.

### O fine-tune piorou o modelo em tudo, inclusive no controle ASCII

Mesmas palavras, mesma semente (42), mesmo extrator de estilo (IAM), quatro
condições: modelo original do IAM e checkpoint de 26 épocas no BRESSAY (25% do
treino, lr 2e-5, deriva 2,198%), cada um com imagens de referência de estilo
do BRESSAY e do IAM. Figura:
`saidas/diffusionpen/fine_tune_25/comparacao_iam_vs_26ep.png`.

| | IAM original | 26 épocas no BRESSAY |
|---|---|---|
| controle `text`, `nacao`, `coracao` | legíveis nas duas referências | ilegíveis com ref. BRESSAY; degradados com ref. IAM |
| palavras acentuadas | legíveis, acento omitido (`avó`→"avo", `não`→"nao", `português`→"portugues") ou palavra degradada (`ação`→"alvo", `nação`→"natio") | ilegíveis; nenhum acento identificável |
| efeito da referência de estilo | quase nenhum | grande: ref. BRESSAY gera palavras minúsculas com traço embaixo |

A referência de estilo do BRESSAY **não** é o problema: o modelo original gera
bem com ela. O que mudou foi o que o fine-tune ensinou.

### Causa: as palavras do BRESSAY têm ~30 px

Varredura de todas as imagens do zip oficial
(`diagnostico/resolucao_bressay.py`):

| tipo | n | altura mediana | p90 | máx |
|---|---|---|---|---|
| páginas | 1.000 | 568 px | 1.115 px | 2.328 px |
| linhas | 30.090 | 21 px | 38 px | 452 px |
| palavras | 416.826 | 26 px | 38 px | 100 px |

Só 8.599 palavras (2%) têm 48 px ou mais; as do IAM têm ~50 px. O README do
BRESSAY explica: as redações vieram de várias plataformas online, sem captura
padronizada. Não há versão em resolução maior.

Depois do `load_image`, o que o modelo recebe é uma palavra pequena e
pixelada, muitas vezes com a linha da pauta embaixo
(`saidas/diffusionpen/fine_tune_25/bressay_como_o_modelo_ve.png`). Um til ou
uma cedilha ocupa 1–2 pixels da imagem original. O fine-tune aprendeu
fielmente essa aparência, e quase não há acento visível nos dados para
aprender.

### Filtrar pelas páginas de melhor resolução não é viável

Altura mediana das palavras por página, no split filtrado: p10 = 29 px,
mediana = 31 px, p90 = 32 px. A distribuição é estreita demais:

| corte (mediana da página) | páginas de treino | palavras de treino | com acento |
|---|---|---|---|
| nenhum | 647 | 74.882 | 7.560 |
| ≥ 30 px | 572 | 71.588 | 7.338 |
| ≥ 35 px | 12 | 3.471 | 531 |
| ≥ 40 px | 4 | 1.543 | 283 |

O split atual já é a metade "alta" do dataset (o `preparar_split.py` descarta
palavras com menos de 28 px; a mediana do dataset inteiro é 26 px).

### Consequências

- Mais épocas, lr menor ou busca de hiperparâmetros não resolvem: otimizariam
  a imitação de imagens de baixa resolução.
- O critério de deriva (faixa-alvo 2,5–4,0% no `medir_deriva.py`) veio de runs
  na placa defeituosa e não se sustentou aqui: a 2,2% as amostras estão piores
  que as do modelo de partida.
- Para o TCC, o próprio dado é um resultado: o maior dataset público de
  manuscrito em português tem palavras de ~30 px, em que o diacrítico ocupa
  1–2 pixels. Isso limita qualquer modelo treinado nos pixels dele.

### Caminhos em aberto

1. Pré-processamento que remova o que o modelo aprendeu de errado: recorte
   justo na tinta, remoção da pauta, escala como a do IAM. Não recupera
   resolução.
2. Ensinar os acentos sem depender dos pixels do BRESSAY: congelar parte do
   UNet, misturar IAM no treino, ou palavras acentuadas sintéticas a partir de
   recortes do IAM.

---

## 8. O pré-processamento v2 mudou o formato da saída, mas não ensinou diacríticos

> Data: 2026-09-30. Run `model_bressay_25_v2` (ver `LOG.md`). Figuras:
> `saidas/diffusionpen/fine_tune_25_v2/comparacao_ref_{iam,bressay}.png`.

- **O que o v2 corrigiu.** Com referência do BRESSAY, o run v1 gerava palavras
  minúsculas com a pauta embaixo; o v2 gera palavras que ocupam a imagem e sem
  a linha contínua. O defeito de formato que o v2 atacava sumiu.
- **O que não mudou.** Em todas as palavras, inclusive no controle `text`, os
  modelos com fine-tune continuam piores que o IAM original. Com referência do
  BRESSAY a saída é em blocos borrados quando as imagens de referência são de
  baixa resolução; com referência do IAM a legibilidade é parecida com a do v1.
- **Nenhum sinal de diacrítico aprendido.** No par mínimo, `nacao` e `nação`
  saem quase idênticos no mesmo estilo — o til é ignorado. Marcas soltas acima
  das letras aparecem também em palavras sem acento (`nacao`, `coracao`), então
  não são diacríticos.
- **Mais treino não ajuda.** 30, 35 e 40 épocas geram imagens quase iguais com a
  mesma semente, embora a deriva tenha ido de 2,51% a 2,92%.
- **A faixa-alvo de deriva não se sustenta.** Os três checkpoints estão na
  "FAIXA ALVO" do `medir_deriva.py` e nenhum gera BRESSAY legível com acento.
  A faixa foi calibrada na RX 6600 XT e não vale como critério nesta placa.

Conclusão: com os pixels do BRESSAY (~30 px por palavra), limpar o formato não
basta. O próximo passo coerente é ensinar os acentos sem depender desses
pixels (seção 7, caminho 2).

### Adendo: sonda da Fase 2 com escritor fixo do IAM

Sonda canônica (`comum/palavras.py`, 20 palavras × seeds 0/1/2) com o escritor
12 do IAM e as mesmas 5 referências do modo `sonda` da Fase 2
(`gerar_amostras.py --classes_iam 12`). Figura:
`saidas/diffusionpen/fine_tune_25_v2/sonda_estilo12_iam.png`.

- **IAM original:** controle ASCII perfeito nas 3 seeds; nas acentuadas, os
  mesmos dois modos de falha do README — omissão (`pão`→"pao", `café`→"cafe",
  `força`→"forca", `português`→"portugues") e degradação (`nação`, `coração`,
  `ação`, `três`, `irmã`, `põe`).
- **Depois do fine-tune o modelo deixa de seguir o estilo do escritor.** Com
  as mesmas referências, o v1 gera palavras minúsculas com pauta e o v2 gera
  traço grosso e borrado; nenhum dos dois reproduz a escrita do escritor 12.
  O controle ASCII, perfeito antes, fica ilegível ou quase.
- Nenhum acento aparece em nenhuma palavra, nas 3 seeds, nos dois modelos.

O fine-tune no BRESSAY não só deixou de ensinar os acentos: apagou parte do
que o modelo sabia (esquecimento catastrófico), inclusive o condicionamento
por estilo.

---

## 9. Treinar em resolução menor não ajuda: o VAE perde o acento em 32×128

> Data: 2026-09-30. `diagnostico/teste_vae_resolucao.py`, figura
> `saidas/diffusionpen/vae_resolucao.png`. Só o VAE do SD-1.5, ida e volta,
> sem treino.

O DiffusionPen gera o latente do VAE, que reduz 8× em cada eixo: 64×256 vira
um latente 8×32; 32×128 viraria 4×16.

- **Em 64×256 o VAE preserva os acentos.** Palavras do BRESSAY (as de tinta
  mais alta) e do RIMES voltam praticamente iguais, com til, cedilha,
  circunflexo e agudo intactos (`informação`, `tância`, `trágico`, `agréer`,
  `précédent`).
- **Em 32×128 o VAE já degrada.** `informação` perde letras, `agréer` perde o
  agudo, `déménager` e `précédent` borram. O acento ocupa menos de um pixel do
  latente.

Consequências:

1. Reduzir a resolução de treino e geração pioraria a representação do acento
   antes de o UNet entrar em cena, além de descartar o que o UNet e o extrator
   de estilo aprenderam na escala de 64×256 do IAM.
2. Em 64×256 o VAE **não** é o gargalo: o acento cabe no latente. A falha das
   seções 7 e 8 está no que o UNet aprende (ou esquece), e na qualidade da
   maior parte dos dados do BRESSAY, não na capacidade de representação.

## 10. Acentos sintéticos no IAM: o modelo passa a desenhar til e cedilha, mas também onde não deve

**Experimento.** `experimentos/iam_acentuado.json`:
- dados: 29.732 palavras do IAM com acento sintético (`docs/acentos_sinteticos.md`)
  mais as 55.535 originais, sem acento;
- 10 épocas, lr 2e-5, a partir dos pesos do IAM.

Amostras com a mesma semente (42) e os mesmos 4 estilos do IAM, comparadas com o modelo
original:
- `saidas/diffusionpen/fine_tune_iam_acentuado/comparacao_iam_vs_10ep.png`;
- `saidas/diffusionpen/fine_tune_iam_acentuado/evolucao.png` (2 a 10 épocas).

**O que funcionou.** O modelo original desenha `nação` igual a `nacao`, sem sinal
nenhum. Depois do fine-tune, `nação`, `coração` e `pão` saem com traços de til e de
cedilha nos lugares aproximados. É a primeira vez neste trabalho que um diacrítico
aparece por causa do texto pedido. No BRESSAY isso não aconteceu (seções 7 e 8).

**O que deu errado:**
1. **Vazamento para palavras sem acento.** O controle `the` sai como `thé` já com 2
   épocas, e `coracao` sai com sinal sobre o `ao`.
   - Hipótese para o `the`: toda ocorrência elegível de `the` no IAM ganhou um acento
     na base (o `e` é a única letra acentuável), então metade dos `the` que o modelo
     viu tinha sinal. A diferença entre `e` e `é` no texto não pesou o suficiente para
     separar os dois casos.
   - Os números confirmam: o treino viu 2.907 `the` sem acento e 2.858 com acento
     (1.693 `thé` e 1.165 `thê`). Sozinhos, `thé` e `thê` são 9,6% da base
     sintética. Depois deles vêm `ãnd` (663), `wíth` (353), `hís` (318) e `thãt`
     (309).
2. **Legibilidade.** As palavras com `ã`/`ç` ficaram piores que as versões sem acento
   (`nação` com letras deformadas). As sem acento continuam parecidas com as do modelo
   original.
3. **`ó` e `ê` não apareceram.** `avó` e `você` saem idênticos a `avo` e `voce` em
   todas as épocas.

**Deriva** (`medir_deriva.py`): 1,01% (2 ép.) · 1,38% (4) · 1,64% (6) · 1,84% (8) ·
2,02% (10). Fica abaixo da faixa "alvo" do script, mas o vazamento já aparece com 2
épocas, então o problema não é treinar demais.

**Implicação.** O dado sintético ensina a relação "diacrítico no texto → marca na
imagem", que o BRESSAY não conseguiu ensinar. Mas a forma como a base foi montada (todas
as ocorrências de uma palavra frequente acentuadas, 35% de amostras acentuadas) ensinou
também a pôr marca onde não há diacrítico. Próximos passos a testar:
- limitar a fração acentuada por palavra, para `the` continuar majoritariamente sem
  acento;
- medir com a métrica (E1) em vez de só olhar.

### Adendo: teto de 25% por palavra (`model_iam_acentuado_teto25`)

Base com no máximo `max(1, round(0,25 × ocorrências))` versões acentuadas por palavra
(9.892 acentuadas; `the` com 718 acentuadas contra 2.907 sem). Mesmo treino, 10
épocas. Comparação com os 4 estilos (semente 42):
`saidas/diffusionpen/fine_tune_iam_acentuado/comparacao_teto_4estilos.png`.

Contagem por estilo (sinal visível / 4):

| palavra | IAM | sem teto | teto 25% |
|---|---|---|---|
| `the` (sem acento) | 0 | 1 | 1 |
| `and` (sem acento) | 0 | 2 | **0** |
| `with` (sem acento) | 0 | 0 | 0 |
| `coracao` (sem acento) | 0 | 2 | 2 |
| `nação` | 0 | 4 | 4, mais deformados |
| `pão` | 0 | 3 | 1 |
| `avó`, `você` | 0 | 0 | 0 |

**Leitura:**
- O teto tirou o vazamento de `and`.
- Não mudou o de `the` (o mesmo estilo 2 nos dois treinos) nem o de `coracao`
  (estilos 1 e 2 nos dois).
- O acento ficou mais fraco em `pão`, e `nação` ficou ainda menos legível.
- O vazamento que sobra se repete **no mesmo estilo** nos dois treinos. Isso sugere
  que ele depende do escritor de referência, e não só da frequência da palavra
  acentuada.
- Com 4 estilos por palavra, as diferenças de 1 em 4 não são conclusivas. É preciso
  medir com muitas amostras por palavra antes de afirmar qualquer coisa.

Deriva: 1,10% (3 ép.) · 1,32% (5) · 1,49% (7) · 1,64% (9) · 1,70% (10), menor que a do
treino sem teto (2,02% com 10 épocas), já que há menos amostras por época.
