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
