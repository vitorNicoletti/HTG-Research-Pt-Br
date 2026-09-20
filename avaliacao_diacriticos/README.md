# Métrica de avaliação de diacríticos

Instrumento para medir se um gerador de manuscrito desenha os diacríticos do
português. Não depende de o fine-tune ter dado certo: ele mede, e o resultado
pode perfeitamente ser "não deu".

Nada aqui toca o treino nem `DiffusionPen/train.py`.

## A ideia

A sonda gera **pares mínimos** ("nacao" / "nação") com estilo e semente fixos,
de modo que a única variável entre os gêmeos seja o diacrítico. Cada imagem é
pontuada em dois eixos **independentes**:

- **E1 — presença**: há tinta na faixa esperada do diacrítico? Acima da
  altura-x para til, agudo, grave e circunflexo; abaixo da linha de base para
  cedilha.
- **E2 — integridade da base**: a palavra continua legível descontando-se o
  diacrítico? Medido pelo CER de um reconhecedor, com alvo e predição dobrados
  para ASCII.

O cruzamento dos dois dá as quatro categorias:

|                     | base íntegra        | base degradada               |
|---------------------|---------------------|------------------------------|
| **acento presente** | acerto              | degradação com diacrítico    |
| **acento ausente**  | omissão do acento   | degradação sem diacrítico    |

## Arquivos

| arquivo | papel |
|---|---|
| `gerar_pares.py` | sonda: gera os pares mínimos com estilo/semente fixos |
| `metrica.py` | núcleo: E1 (duas variantes), CER, dobra ASCII |
| `reconhecedor.py` | E2: TrOCR + CER dobrado para ASCII |
| `preparar_reais.py` | controle: recortes REAIS do BRESSAY no mesmo formato |
| `avaliar.py` | aplica E1/E2 a um conjunto, classifica, IC por bootstrap |
| `calibrar.py` | Passo 4: valida nos dois casos de resposta conhecida |
| `anotacao.py` | Passo 6: planilha de anotação humana e Cohen's kappa |
| `testes_metrica.py` | testes sintéticos de resposta conhecida da geometria |
| `checar_alinhamento.py` | Passo 1: mede se os gêmeos saem alinhados |
| `controle_ascii.py` | controle negativo **no mesmo domínio**: palavra real sem acento, com acento fabricado |
| `teste_linha_pautada.py` | diagnóstico do efeito da pauta na geometria |
| `teste_folga.py` | sensibilidade da janela de coluna ao parâmetro `folga` |
| `teste_sensibilidade_diff.py` | curva de detecção do eixo por diferença |

## Passo 1 — o portão (resultado: PASSOU)

Antes de decidir como fazer o E1, foi preciso medir se os gêmeos saem
espacialmente alinhados. Se saem, o E1 pode ser feito por **diferença de
imagens**, que é mais robusto do que estimar linha de base e altura-x.

6 pares mínimos, mesmo escritor (174), mesma semente, `ema_ep11.pt`:

```
par                      dx   dy   IoU@0  corr_col
ação/acao                 0    0   0.879     0.992
àqueles/aqueles           0    0   0.967     1.000
doença/doenca             0    0   0.949     0.999
importância/importancia   0    0   0.649     0.853
não/nao                   0    0   0.915     0.999
país/pais                 0    0   0.957     0.999
MEDIANAS                  0    0   0.932     0.999
```

Deslocamento ótimo zero em todos os pares. **Os gêmeos saem alinhados**, então
o E1 por diferença é válido. Os números por par estão em
`resultados/passo1_alinhamento.json`.

Esta conclusão foi **revista depois**: ver a seção sobre alinhamento abaixo.
Em 348 pares, 60% saem com deslocamento zero, mas o máximo chega a 18 px — o
portão acertou na média e errou na cauda.

As duas variantes do E1 continuam existindo porque servem a coisas diferentes:

- `e1_por_diff` — usa o par. Mais limpo, mas exige o gêmeo.
- `e1_por_faixa` — usa só a imagem acentuada. É a **única** que funciona em
  recortes reais do BRESSAY, onde não existe gêmeo ASCII do mesmo punho. Toda
  comparação entre gerado e real usa esta variante nos dois lados.

## Achado de infraestrutura: lote > 1 corrompe a amostragem nesta GPU

Descoberto ao montar a sonda, e vale para qualquer geração neste repositório.
Mesma entrada, mesmo escritor, mesmo ruído, RX 6600 XT / ROCm:

```
batch 4, rep 0 : [0.0793, 0.0515, 0.0423, 0.0643]   std da imagem
batch 4, rep 1 : [nan, nan, nan, nan]
batch 4, rep 2 : [0.1229, 0.0596, 0.0756, 0.0691]
batch 2        : [0.1863, 0.0597, 0.0398, 0.0637]
batch 1, rep 0 : [0.1858, 0.1945, 0.2165, 0.2193]
batch 1, rep 1 : [0.1809, 0.1945, 0.2165, 0.2193]
```

Com lote > 1 a saída colapsa para cinza quase uniforme (ou vira NaN) e não é
reproduzível. Com lote 1 é estável e **igual à CPU**: nas 12 imagens do portão,
max|dif| GPU-vs-CPU = 0.0039, que é 1/255, a quantização do PNG.

As operações básicas da GPU estão sãs (matmul, conv e reduções bit-exatos, sem
NaN em stress) — o problema aparece só no caminho completo de amostragem.

Consequência para a métrica: uma imagem colapsada seria lida como "acento
omitido", ou seja, o artefato viraria resultado. Por isso:

- `gerar_pares.py` usa `--batch 1` por padrão;
- `--autoteste-lote` verifica a invariante e **falha** se o lote divergir do
  individual (medido: passa bit-exato em batch 1, falha com mean|dif| = 0.43 em
  batch 4);
- o manifest marca `colapsada` para imagens com NaN ou std < 0.02, e o
  `avaliar.py` as descarta em vez de pontuá-las.

Varrendo as pastas `amostras_*` já existentes: **12 de 332 sub-painéis (3,6%)
estão degenerados**, todos em `ação.png`, nas pastas `amostras_bressay_ep15`,
`amostras_bressay_ep15_sem_flag` e `amostras_v2_5ep` — os 4 estilos com
std = 0,000, imagem em branco. Não é o modelo falhando em "ação": é a
amostragem em lote.

## E2 satura no piso — e satura já no manuscrito humano

Este é o resultado do Passo 3/4 para o segundo eixo, e ele é negativo. O TrOCR
`microsoft/trocr-base-handwritten` aplicado aos recortes **REAIS** do BRESSAY:

| grupo | n | CER médio | IC95 | mediana |
|---|---|---|---|---|
| acentuadas | 148 | 0,826 | [0,760, 0,895] | 0,778 |
| ASCII (controle) | 120 | 1,035 | [0,922, 1,160] | 1,000 |

CER 1,0 quer dizer "tão errado quanto uma string vazia". **Apenas 1 de 120**
recortes acentuados foi lido perfeitamente; 12 ficaram com CER ≤ 0,3. Exemplos
de leitura em manuscrito humano perfeitamente legível: "psicólogos" →
`assimilance`, "vivência" → `preceivers .`, "pública" → `1million .`.

A diferença entre os dois grupos é **confundimento de comprimento**, não
qualidade: as acentuadas têm 8,32 caracteres de média contra 5,28 das ASCII, e
o CER normaliza pelo comprimento do alvo. Dentro de cada faixa os grupos
empatam:

| comprimento | CER acentuadas | CER ASCII |
|---|---|---|
| 3–4 | 1,550 (n=15) | 1,520 (n=58) |
| 5–6 | 0,773 (n=11) | 0,727 (n=33) |
| 7–9 | 0,747 (n=53) | 0,396 (n=20) |
| 10+ | 0,656 (n=41) | 0,452 (n=9) |

**A consequência é que o eixo não decide nada.** Em manuscrito humano real, onde
a base está íntegra por construção, a classificação vira de ponta-cabeça
conforme a escolha do limiar:

| limiar de E2 | "acerto" | "degradação" |
|---|---|---|
| 0,3 absoluto | 6,8% | **89,9%** |
| 0,5 absoluto | 21,6% | 74,3% |
| relativo ao grupo ASCII (q75) | **83,8%** | 6,8% |

Com o limiar relativo, "base íntegra" acaba significando "não é pior do que um
baseline ilegível" — vacuamente verdadeiro. Com limiar absoluto, manuscrito
humano legível é declarado degradado.

**Leitura para o TCC:** não é que o gerador degrade a base — é que *este
reconhecedor não lê este corpus*. O eixo E2 não está validado e não deve
sustentar conclusão sobre integridade de base enquanto não houver um
reconhecedor treinado em manuscrito português. Até lá, a classificação honesta
tem duas categorias (acento presente / ausente), não quatro. Isso está
implementado: sem `--com-e2`, o `avaliar.py` rotula `presente`/`ausente` em vez
de forçar as quatro.

O caminho para consertar o eixo é trocar o reconhecedor, não afrouxar o limiar.
A folha de contato `figuras/anotacao_folha.png` deixa isso visível: as palavras são
perfeitamente legíveis para uma pessoa, e o CER diz 0,83.

## A linha pautada do papel quebrava o E1 (corrigido)

73% dos alvos de treino do BRESSAY e 88% dos recortes de teste têm a linha
pautada do papel atravessando a imagem. Ela não é tinta do escritor e estragava
as duas coisas que o E1 faz:

1. **a geometria.** `linha_base_e_altura_x()` chama de "corpo" as fileiras com
   pelo menos metade da tinta da fileira mais cheia. A pauta atravessa a imagem
   inteira, então ela *é* a fileira mais cheia: o corpo estimado caía de 25 px
   para 12 px e a faixa ACIMA passava a engolir a letra em vez de só a zona do
   diacrítico.
2. **a contagem.** Para a cedilha a faixa é ABAIXO da linha de base — exatamente
   onde a pauta costuma estar.

Os dois efeitos são mensuráveis. Em teste sintético, sem remover a pauta o
escore de um til conhecido vai de **0,082 para 1,553** (19×). Nos recortes reais,
agudo ia a 1,070 e til a 1,035 nos que têm pauta, contra 0,265 e 0,328 nos que
não têm.

**O discriminador é espessura, não cobertura.** A primeira tentativa (zerar
fileiras que cobrem >85% da caixa de tinta) destruía palavras curtas: em "que",
"para" e "das" ela levava de 52% a 62% da tinta, porque num vocábulo de 3 letras
um traço cursivo horizontal já cobre 85% da própria caixa. Medido no conjunto
real, as bandas cheias se concentram em 2–4 px com queda brusca depois de 5 —
o esperado para uma linha de 1–2 px ampliada 2,06× (os recortes do BRESSAY têm
31 px de altura e são ampliados para 64). `remover_pauta()` só apaga bandas de
até 5 px.

Sobram casos difíceis: palavras onde a normalização de contraste deixou tudo
um bloco sólido continuam dando falso positivo ("para") ou falso negativo
("das"). Não foi tentada reconstrução do traço que cruza a pauta.

**O efeito no controle positivo foi grande e está reportado como tal:**

| | antes | depois |
|---|---|---|
| E1 médio dos recortes reais | 0,861 | **0,510** |
| E1 médio do IAM puro | 0,141 | 0,141 (sem pauta, não muda) |

**A correção vale para o gerado, não só para o controle.** Medindo a presença
de pauta com o mesmo detector:

| conjunto | painéis | com pauta |
|---|---|---|
| `ger_iam_puro/` (IAM puro) | 116 | **0%** |
| `images/*.jpg` (amostras IAM) | 8 | **0%** |
| `am_v2/*.png` (ajustado no BRESSAY) | 24 | **100%**, sempre 4 fileiras |
| `gate_cpu/` (ajustado, ep11) | 12 | 50% |

O modelo ajustado aprende a desenhar a pauta — e as 4 fileiras que ele desenha
caem dentro do teto de 5 px, então `remover_pauta()` as pega. Sem a correção, a
métrica leria a pauta aprendida como acento em praticamente toda amostra do
fine-tune.

## O controle positivo é mais fraco do que a comparação com o IAM sugeria

O negativo do Passo 4 é o DiffusionPen puro do IAM. Ele responde "o gerador que
nunca viu português omite o acento?", mas **não** responde "o detector inventa
acento onde não há?" — imagem gerada e recorte real diferem em traço, contraste
e pauta (0% no IAM, 88% no real), então parte da separação pode ser só domínio.

`controle_ascii.py` constrói o negativo no próprio domínio: pega palavras REAIS
do BRESSAY que comprovadamente não têm diacrítico e finge que têm. Mesmo papel,
mesma pauta, mesmo punho. Tudo que o E1 marcar ali é falso positivo por
construção.

| negativo usado | E1 médio do negativo | E1 médio do positivo | **AUC** |
|---|---|---|---|
| IAM puro (fora de domínio) | 0,141 | 0,510 | **0,849** |
| ASCII real com acento fabricado (mesmo domínio) | 0,386 | 0,510 | **0,680** |

Por marca, no mesmo domínio: cedilha 0,930 (n⁻=8), grave 0,755, til 0,723,
circunflexo 0,731, agudo 0,686.

**Conclusão honesta: o E1 por faixa, em material real, é um detector fraco
(AUC 0,68).** A barra do Passo 4 — "omissão e presença perto de 100%" — não é
atingida com limiar único: no melhor limiar global dá 45,0% de omissão no
negativo de domínio e 90,5% de presença no positivo. Isso limita quanto peso o
controle positivo de recortes reais pode sustentar, e precisa ser declarado.

Vale distinguir o que isso condena e o que não condena. A variante por faixa é
necessária só onde **não há gêmeo** — ou seja, no controle de recortes reais. A
medida no material gerado, que é o objetivo do trabalho, usa a variante por
diferença, caracterizada abaixo.

## O eixo por diferença: piso de ruído zero e curva de detecção

O eixo por diferença tem controle negativo validado (IAM puro = 0,003) mas
ainda não tem positivo real — não existe gerador nosso que desenhe o
diacrítico. `teste_sensibilidade_diff.py` constrói o positivo sem depender de
gerador: pinta um acento de tamanho conhecido sobre a própria imagem ASCII que
o modelo gerou, de modo que a única diferença entre os gêmeos seja o acento.

Acento nominal = 60% da largura do caractere × 18% da altura-x.

| escala do acento | delta_rel médio | detectado (>0,02) |
|---|---|---|
| 0,00 (gêmeos idênticos) | 0,0000 | 0,0% |
| 0,25 | 0,0103 | 5,2% |
| 0,50 | 0,0389 | 87,9% |
| 0,75 | 0,0947 | 100,0% |
| 1,00 (nominal) | 0,1657 | 100,0% |
| 1,50 | 0,3634 | 100,0% |
| **IAM puro, real** | **0,0029** | 25,9% |

O piso de ruído é exatamente zero, e meio acento nominal já é detectado em 88%
dos casos. O IAM puro produz 1,8% da massa de um acento nominal — é o controle
negativo se comportando como esperado. **É esta tabela que dá sentido aos
números do fine-tune quando ele existir**: 0,05 passa a ser "cerca de um terço
de acento", não um número solto.

## O IAM não é controle negativo puro para cedilha

Olhando `figuras/e1_falhas.png`, os maiores escores do controle negativo não
parecem todos ruído: em "presença" o final virou "ga" — um c com cauda abaixo
da linha de base, que é a forma de um ç —, enquanto a gêmea "presenca" tem um
"ca" limpo. Em "preço" acontece o mesmo.

Testado em vez de aceito. Para cada amostra, comparei o escore na **coluna do
acento** com o escore nas **outras colunas da mesma imagem**, com a mesma
faixa. Se fosse jitter de traço, as duas seriam iguais. Diferença pareada,
IC95 por bootstrap sobre 4.000 reamostragens:

| marca | diferença | IC95 | efeito real? |
|---|---|---|---|
| **cedilha** | **+0,0245** | [+0,0085, +0,0416] | **sim** |
| agudo | +0,0067 | [−0,0064, +0,0197] | não |
| grave | −0,0028 | [−0,0114, +0,0056] | não |
| circunflexo | −0,0123 | [−0,0289, +0,0040] | não |
| til | −0,0112 | [−0,0210, −0,0018] | sim, mas **negativo** |

E o extra de tinta da cedilha está na faixa certa, não no corpo da letra: na
coluna do ç o corpo **perde** tinta (−0,0245) enquanto a faixa abaixo da linha
de base **ganha** (+0,0338). O modelo desloca massa para baixo da linha de
base, que é o que desenhar uma cedilha faz.

**Consequência, e ela é metodológica.** Eu vinha dizendo que a métrica é pouco
confiável na cedilha porque o ruído (p95 = 0,238) supera um acento nominal
(0,165). A leitura correta é outra: **o controle não é negativo nessa marca**.
O que eu estava chamando de ruído é, em parte, sinal — o IAM produzindo algo
cedilhoide. Logo o p95 da cedilha calibrado no IAM está inflado e não serve
como limiar.

Para o agudo a hipótese **não** se sustenta: o "i maior" que aparece num caso
isolado de "país" não sobrevive ao teste agregado com n=72.

O que isso NÃO estabelece: que o ç gerado seja bem formado ou legível. O que
está medido é que há tinta a mais, na coluna certa e na faixa certa,
sistematicamente. Julgar a forma é trabalho da anotação humana (Passo 6).

Nada disso foi corrigido — é uma limitação declarada do controle, não um bug.
Til, grave e circunflexo continuam com controle negativo válido, e são as
marcas em que o instrumento está pronto para medir o fine-tune.

## Alinhar os gêmeos antes de subtrair (e o portão do Passo 1 revisto)

O portão do Passo 1 mediu 6 pares, deu deslocamento zero em todos, e eu
concluí que os gêmeos saem alinhados. Em 348 pares isso **não vale sempre**:
60% saem com deslocamento zero e o p90 é de 1 px, mas o máximo chega a 18 px.
Nesses casos a palavra inteira entra na subtração, e parte do resíduo cai
dentro da região medida — são exatamente os falsos positivos da figura
`figuras/e1_falhas.png`.

Alinhar por correlação cruzada antes de subtrair (6 linhas, `alinha()` em
`e1.py`) custa pouco e paga:

| | p95 do ruído | AUC |
|---|---|---|
| sem alinhar | 0,181 | 0,936 |
| alinhar pelo canto da caixa de tinta | 0,204 | 0,906 |
| **alinhar por correlação cruzada** | **0,103** | **0,966** |

Alinhar pelo canto da caixa, que seria mais simples, **piora** — o canto
depende de um pixel solto.

Limiares operacionais por marca, recalculados com o alinhamento
(`limiares_eixo_diff.json`):

| marca | p95 do ruído | acento nominal | AUC | decide por amostra? |
|---|---|---|---|---|
| grave | 0,036 | 0,151 | 1,000 | sim |
| til | 0,046 | 0,141 | 0,998 | sim |
| circunflexo | 0,067 | 0,149 | 0,986 | sim |
| agudo | 0,126 | 0,158 | 0,948 | sim |
| **cedilha** | **0,238** | 0,165 | 0,917 | **não** |

O alinhamento resgatou o agudo (p95 de 0,268 para 0,126, agora abaixo do
sinal). A cedilha continua sendo a marca em que o ruído supera um acento
nominal: só a média agregada com IC é confiável nela. Faz sentido — a faixa
da cedilha é abaixo da linha de base, que é onde ficam descendentes e a linha
pautada.

## O limiar do eixo por diferença sai do negativo REAL, não da curva sintética

A curva acima é o melhor caso possível: o acento é pintado sobre a *mesma*
imagem, então fora do acento os gêmeos são idênticos. Em par mínimo gerado de
verdade os dois gêmeos diferem em todo lugar por jitter de traço, e esse ruído
entra no delta. Medido no Passo 5 (IAM puro, 696 imagens, 348 diacríticos):
**28,4% das amostras cruzam o piso sintético de 0,02**, e 16,4% cruzam 0,0389 —
num gerador que comprovadamente não desenha diacrítico nenhum.

Portanto o limiar operacional se calibra no controle negativo real
(`limiares_eixo_diff.json`), não na curva sintética:

| marca | média | p90 | **p95** | p99 | acento nominal |
|---|---|---|---|---|---|
| grave | −0,0088 | 0,0276 | **0,0365** | 0,0562 | 0,1657 |
| til | −0,0116 | 0,0319 | **0,0455** | 0,0667 | 0,1657 |
| circunflexo | −0,0085 | 0,0350 | **0,0682** | 0,1213 | 0,1657 |
| cedilha | 0,0311 | 0,1643 | **0,2056** | 0,3534 | 0,1657 |
| agudo | 0,0526 | 0,1907 | **0,2636** | 0,3401 | 0,1657 |

Usar o p95 como limiar quer dizer "um escore assim aparece em menos de 5% das
amostras de um gerador que não desenha acento".

**Limitação que precisa estar no TCC:** para agudo e cedilha o ruído do p95
(0,264 e 0,206) **supera** o sinal de um acento nominal (0,166). Nessas duas
marcas a classificação amostra a amostra não é confiável — só a média agregada
com intervalo de confiança. Faz sentido: o agudo no "i" substitui o pingo, que
o gêmeo ASCII também tem, e a cedilha cai numa região já cheia de tinta. Para
til e grave, que são o alvo central do trabalho, o eixo separa por amostra com
folga de 3 a 4 vezes.

Til, grave e circunflexo dão média **negativa** no IAM puro: a versão acentuada
tem *menos* tinta na faixa do que a gêmea ASCII. Não é o modelo desenhando um
acento pequeno — é o modelo renderizando uma palavra diferente, sem acento
nenhum. É o controle negativo se comportando como deveria.

## A janela de coluna é sensível à folga, mas a decisão não é

`coluna_do_caractere()` divide a caixa de tinta em fatias iguais. No par mínimo
isso se cancela entre os gêmeos; em recorte real não. Medido:

| n de letras | folga=0,0 | folga=0,5 | folga=1,0 |
|---|---|---|---|
| 3 | 34% | 68% | 100% |
| 7 | 15% | 29% | 43% |
| 10 | 10% | 20% | 30% |

(percentual da palavra coberto pela janela). Em palavra de 3 letras com
folga=1,0 a janela cobre a palavra inteira e o E1 deixa de ser "tinta na coluna
do caractere".

O escore absoluto varia bastante com a folga (0,58 → 0,40 entre folga 0 e 2),
**mas a separação quase não se move**: AUC entre 0,790 e 0,880 em toda a faixa.
Ou seja: o escore E1 não pode ser reportado como grandeza absoluta sem declarar
a folga, e o limiar tem que ser calibrado na mesma folga da medição — mas a
decisão presente/ausente é estável. O padrão continua 0,5, que foi a escolha a
priori; 0,25 dá AUC marginalmente melhor e não foi adotado para não ajustar
parâmetro no próprio controle.

## Limitação do corpus: til e cedilha quase não têm gêmeo real

Dos 10.691 tipos do BRESSAY, 2.296 têm diacrítico e 188 têm o gêmeo ASCII
também como palavra real. Mas a distribuição por marca é muito desigual:

| marca | pares com gêmeo ASCII real |
|---|---|
| agudo | 148 |
| circunflexo | 23 |
| grave | 8 |
| cedilha | 5 |
| til | **3** (`sã`, `fã`, `dã`) |

Til e cedilha são justamente as marcas que o IAM puro apaga. Por isso
`pares_sonda30.tsv` é estratificado por marca (6 por marca) e traz uma coluna
`gemeo_e_palavra_real`: vale 1 para agudo/grave/circunflexo e 0 para
til/cedilha. Nesses últimos o gêmeo ASCII é uma não-palavra, e qualquer
diferença observada pode ser efeito de não-palavra em vez de efeito de
diacrítico. **Isso tem que ser declarado ao reportar.**

## Como rodar

Os caminhos do repositório já foram reorganizados uma vez durante o trabalho,
então os scripts procuram o código do DiffusionPen, o extrator de estilo e o
split numa lista de candidatos, e falham com mensagem útil em vez de
`ImportError`. `--ckpt` aceita o caminho do `.pt` direto.

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
PY=/nix/store/98rpw6g3y3j5vc2xiyhwdqgy7xl1qyix-python3-3.14.7-env/bin/python
IAM=DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt

# 0. testes do instrumento (não usam o gerador nem GPU)
$PY avaliacao_diacriticos/testes_metrica.py

# 1. invariante de amostragem do checkpoint que for usar
$PY avaliacao_diacriticos/gerar_pares.py --ckpt $IAM --out-dir /tmp/x \
    --pares-tsv avaliacao_diacriticos/pares_gate.tsv \
    --n-styles 1 --seeds 0 --batch 1 --autoteste-lote

# 2. controles do Passo 4
$PY avaliacao_diacriticos/preparar_reais.py \
    --out-dir avaliacao_diacriticos/amostras/reais_test
$PY avaliacao_diacriticos/controle_ascii.py \
    --dir avaliacao_diacriticos/amostras/reais_test \
    --out-dir avaliacao_diacriticos/amostras/reais_ascii_negativo

# 3. gerar a sonda (N declarado: 29 pares x 4 estilos x 3 sementes = 696)
$PY avaliacao_diacriticos/gerar_pares.py --ckpt <ckpt.pt> --out-dir <saida> \
    --pares-tsv avaliacao_diacriticos/pares_sonda30.tsv \
    --n-styles 4 --seeds 0 1 2 --batch 1

# 4. avaliar. --eixo1 auto usa diferença quando há gêmeo e faixa quando não há.
#    O limiar de E2, se omitido, sai do quantil 0.75 do grupo ASCII do próprio
#    conjunto, que é o que o planejamento pede. --reusar-e2 evita re-rodar o
#    TrOCR (~20 min em CPU para 240 imagens) quando só o E1 mudou.
$PY avaliacao_diacriticos/avaliar.py --dir <saida> --csv-out <saida>.csv \
    --com-e2 --device cpu --eixo1 auto --limiar-e1 <da calibração>

# 5. calibrar contra os DOIS negativos e comparar
$PY avaliacao_diacriticos/calibrar.py \
    --negativo avaliacao_diacriticos/resultados/res_ascii_negativo.csv \
    --positivo avaliacao_diacriticos/resultados/res_reais_e1.csv

# 6. planilha de anotação humana e kappa
$PY avaliacao_diacriticos/anotacao.py preparar --csv <saida>.csv \
    --dir <saida> --out anotacao.csv --contato anotacao.png
$PY avaliacao_diacriticos/anotacao.py kappa --csv anotacao_preenchida.csv
```

## O que ainda falta para medir o fine-tune

O instrumento está pronto e caracterizado; o que falta é material gerado que
tenha diacrítico. Quando houver:

1. rodar `--autoteste-lote` no checkpoint novo (a amostragem em lote corrompe
   nesta GPU, e uma imagem colapsada seria lida como "acento omitido");
2. gerar com `pares_sonda30.tsv`, `--n-styles 4 --seeds 0 1 2 --batch 1`;
3. ler o eixo por diferença contra a tabela de sensibilidade acima —
   0,0000 é nada, 0,0389 é meio acento, 0,1657 é acento nominal;
4. só então aplicar o kappa, sorteando a amostra estratificada pelas quatro
   categorias.
