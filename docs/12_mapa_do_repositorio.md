# 12 — Mapa do repositório

O que é cada pasta e cada arquivo, agrupado pela etapa do trabalho.

## Raiz

| arquivo | o que é |
|---|---|
| `README.md` | apresentação curta, com link para esta pasta |
| `docs/` | **esta documentação** |
| `ACHADOS.md` | achados na ordem em que apareceram (seções 1–17): o registro histórico detalhado |
| `LOG.md` | diário datado de execução: comandos, tempos, incidentes |
| `CLAUDE.md` | instruções para o assistente de código usado no projeto; também é um resumo técnico |
| `requirements.txt`, `flake.nix`, `flake.lock` | dependências (venv ou Nix) |
| `.gitattributes`, `.gitignore` | o `.gitignore` deixa fora do git o clone, os pesos, os dados e as pastas de modelos |

## Dados versionados

| pasta | o que é |
|---|---|
| `vocabulario_pt/` | vocabulário português particionado (treino/val/teste por grupo), **congelado**; fonte FrequencyWords, CC-BY-SA 4.0 |
| `bressay_split/` | split do BRESSAY por escritor (`splits/*.tsv`, `writers_dict.json`); as imagens não são versionadas |
| `bressay_split_25/`, `bressay_split_25_v2/` | splits reduzidos (25%) do BRESSAY; o v2 ainda filtrado por altura de tinta |
| `base_vazia/` | "base" sem nenhuma amostra, para o controle só com originais do IAM |
| `experimentos/` | um JSON por treino, com todos os parâmetros (ver [02](02_diffusionpen_e_modificacoes.md)) |

## Modificações do DiffusionPen

| caminho | o que é |
|---|---|
| `diffusionpen_mods/train.py` | o `train.py` modificado: fine-tune, robustez, leitores novos, pesos na loss |
| `diffusionpen_mods/style_encoder_train.py` | correções no treino do extrator de estilo |
| `diffusionpen_mods/utils/bressay_dataset.py` | leitor do BRESSAY (pré-processamento v1/v2) |
| `diffusionpen_mods/utils/iam_acentuado_dataset.py` | leitor das bases de acentos sintéticos, com as máscaras do acento e da zona |
| `diffusionpen_mods/README.md` | cada correção, explicada |
| `scripts/aplicar_mods.sh` | copia os arquivos acima para o clone `DiffusionPen/` |
| `sonda/patches_diffusionpen.diff` | patch aditivo, só para a sonda (modo `sonda`) |

## Fase 1 — Sonda

| arquivo | o que é |
|---|---|
| `env/check_env.py` | confere GPU e ROCm; aborta se cair para CPU |
| `sonda/smoke_test.py` | 5 palavras em inglês, para ver se o modelo roda |
| `sonda/sonda_diacriticos.py` | a sonda de diacríticos (4 grupos × 5 palavras × 3 sementes) |
| `comum/palavras.py` | lista canônica de palavras da sonda e pares mínimos (não editar sem registrar no LOG) |
| `comum/folha_contato.py` | monta folhas de contato para inspeção visual |

## Fase 2 — Métrica E1/E2 (`avaliacao_diacriticos/`)

Instrumento independente do treino: o E1 mede presença do acento na faixa, o
E2 a integridade da palavra via TrOCR, e há uma ferramenta de kappa. Estado em
`avaliacao_diacriticos/ESTADO.md`, detalhes no `README.md` da pasta. O único
conjunto de testes automatizados do repositório é
`avaliacao_diacriticos/testes_metrica.py`.

## Fase 3 — BRESSAY

| arquivo | o que é |
|---|---|
| `scripts/preparar_split.py` | monta `bressay_split/` com filtros de qualidade |
| `scripts/reduzir_split.py` | split reduzido (fração por escritor) |
| `scripts/filtrar_tinta.py` | remove palavras com tinta baixa demais (para o v2) |
| `scripts/medir_deriva.py` | o quanto os pesos se afastaram dos do IAM (registro, não critério) |
| `diagnostico/resolucao_bressay.py` | altura das palavras do BRESSAY |
| `diagnostico/comparar_preproc.py` | comparação visual v1 × v2 |
| `diagnostico/inspecionar_dados.py` | o que o modelo recebe como alvo de treino |
| `diagnostico/teste_vae_resolucao.py` | o VAE preserva o acento em 64×256? e em 32×128? |
| `docs/figuras/gerar_figuras.py` | figuras do [05b](05b_pre_processamento_bressay.md) |

## Fases 4–9 — Acentos sintéticos

**Pacote `acentos_sinteticos/`** (descrito em [06](06_acentos_sinteticos.md)):

| módulo | o que faz |
|---|---|
| `iam.py` | lista e lê as palavras do IAM |
| `geometria.py` | tinta (Otsu), corpo da palavra, espessura, ponto de contato do sinal, ajuste das fronteiras aos vales |
| `alinhamento.py` | reconhecedor CTC, alinhamento forçado (Viterbi), fatias por letra |
| `tracos.py` | formas paramétricas dos sinais |
| `desenho.py` | rasterização com a caneta da palavra, apagamento do pingo do i, visibilidade |
| `gerador.py` | escolhe letra e sinal, desenha e filtra; `acentuar_palavra` (todos os sinais), `par_na_tela` |
| `vocabulario.py` | esqueleto, grupo e conferência anti-vazamento do vocabulário |
| `zona_vogais.py` | máscara das vogais para a loss |

**Scripts:**

| script | o que faz |
|---|---|
| `scripts/treinar_alinhador.py` | treina o alinhador (cabeça conv) e o leitor (`--cabeca transformer`) |
| `scripts/avaliar_posicao_letras.py` | acurácia da localização das letras |
| `scripts/amostras_acentos.py` | folhas de amostras do gerador |
| `scripts/gerar_base_acentos.py` | base de acentos sobre palavras reais do IAM (`--teto_por_palavra`) |
| `scripts/preparar_vocabulario_pt.py` | gerou o `vocabulario_pt/` (**não rodar de novo**) |
| `scripts/gerar_base_pt.py` | base de palavras portuguesas geradas pelo DiffusionPen e acentuadas |
| `scripts/alinhar_pares.py` | põe os pares da base pt na tela da acentuada |
| `scripts/criar_pares_iam.py` | cria os pares da base sobre o IAM real |
| `scripts/mascaras_vogais.py` | pré-calcula a zona vogais de todas as amostras |
| `scripts/treinar.py` | **lançador de todo treino**, a partir de `experimentos/*.json` |
| `scripts/gerar_amostras.py` | gera amostras de um checkpoint |
| `comum/diffusionpen.py` | carrega o DiffusionPen uma vez e gera em lote (usado pela base pt e pela avaliação) |

**Avaliação** (descrita em [08](08_protocolo_de_avaliacao.md)):

| script | o que faz |
|---|---|
| `scripts/avaliar_pt.py` | protocolo fixo: gera e mede as 3.200 imagens por modelo |
| `scripts/medir_marcas.py` | detector de marcas soltas (também usado pelo `avaliar_pt.py`) |
| `scripts/comparar_avaliacoes.py` | tabela por modelo e bootstrap por palavra |
| `scripts/medir_posicao_acento.py` | o acento caiu na letra certa? |
| `scripts/comparar_paineis.py` | figuras lado a lado entre modelos |
| `diagnostico/nitidez.py` | o traço está borrado? |

**Diagnósticos e conferências:**

| script | o que faz |
|---|---|
| `diagnostico/diag_peso_acento.py` | o acento pesa pouco na loss? (fase 6a) |
| `diagnostico/conferir_mascara_acento.py` | confere a máscara do acento antes do treino |
| `diagnostico/conferir_zona.py` | confere a zona vazia |
| `diagnostico/conferir_zona_vogais.py` | confere a zona vogais |
| `diagnostico/cer_base.py` | legibilidade das bases contra a escrita real |

## Hardware (`diagnostico/`)

Validação de GPU nova e a investigação da RX 6600 XT (ver
[03](03_ambiente_e_hardware.md) e `diagnostico/README.md`):

| script | o que faz |
|---|---|
| `teste_conv_isolada.py` | **mínimo**: uma convolução CPU × GPU |
| `teste_gradiente_sintetico.py` | **mínimo**: gradientes não finitos com dados sintéticos |
| `teste_direcao_gradiente.py` | **mínimo**: cosseno do gradiente CPU × GPU |
| `teste_forward_completo.py`, `teste_lote_unet.py`, `teste_repeticao.py` | forward e repetibilidade por tamanho de lote |
| `teste_gradiente_pareado.py`, `teste_gradiente_cpu_vs_gpu.py`, `teste_acumulacao_gradiente.py` | investigação do backward na placa defeituosa |
| `treinar_cpu.py` | treino de referência na CPU |

## Resultados versionados

| pasta | o que é |
|---|---|
| `diagnostico/resultados/` | resultados curados: tabelas (`.md`/`.json`) e figuras de cada diagnóstico e avaliação |
| `saidas/diffusionpen/` | figuras da sonda, do BRESSAY e da fase 4 |
| `saidas/acentos_sinteticos/` | figuras do gerador de acentos e da avaliação das 200 amostras |

## Fora do git (só na máquina de treino)

| caminho | o que é |
|---|---|
| `DiffusionPen/` | o clone do modelo, com pesos e dados do IAM |
| `bressay/` | as imagens do BRESSAY |
| `modelos/` | o alinhador e o leitor |
| `iam_acentuado*/`, `iam_pt*/` | as bases de treino |
| `model_*/` | os modelos treinados (`models/ema_bloco_*.pt`), logs e amostras |
| `avaliacao_*/` | as avaliações completas |
| `saidas/comparacao_*/` | as figuras comparativas completas |

## Por que os scripts não foram reorganizados em subpastas

Vários scripts encontram a raiz do repositório pela própria posição
(`os.path.dirname(os.path.dirname(__file__))`) e importam uns aos outros
(`avaliar_pt.py` importa `medir_marcas.py`, `cer_base.py` importa
`avaliar_pt.py`). Mover arquivos de pasta quebraria esses caminhos e os
comandos registrados no `LOG.md`. Preferimos manter a estrutura e documentar.
