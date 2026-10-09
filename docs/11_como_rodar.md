# 11 — Como rodar

Guia prático: do zero até reproduzir um treino e uma avaliação. As armadilhas
de máquina estão em [03](03_ambiente_e_hardware.md); leia antes.

Todos os comandos rodam **na raiz do repositório**, no Linux/WSL, com o
ambiente ativado.

## 1. Código, pesos e dados

```bash
git clone https://github.com/vitorNicoletti/HTG-Research-Pt-Br
cd HTG-Research-Pt-Br
git checkout treino-split-reduzido          # a branch com todo o trabalho

# o DiffusionPen fica ao lado, fora do git
git clone https://github.com/koninik/DiffusionPen.git DiffusionPen
bash scripts/aplicar_mods.sh                # copia as nossas modificações por cima
```

Artefatos externos, que não vão para o git:

| artefato | origem | destino |
|---|---|---|
| pesos do DiffusionPen do IAM | `huggingface.co/konnik/DiffusionPen` | `DiffusionPen/diffusionpen_iam_model_path/models/{ema_ckpt,ckpt}.pt` |
| extrator de estilo do IAM | o mesmo repositório | `DiffusionPen/style_models/iam_style_diffusionpen.pth` |
| palavras do IAM (`words.tgz`) | `fki.tic.heia-fr.ch` (exige cadastro) | `DiffusionPen/iam_data/words/` |
| VAE e agendador do SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` no Hugging Face | baixado automaticamente |
| BRESSAY (só para a fase 3) | site oficial do dataset | `bressay/data/words/` |

Os modelos auxiliares treinados por nós (`modelos/alinhador_iam.pt` e
`modelos/leitor_iam_transformer.pt`) também ficam fora do git. Recrie com o
passo 4.

## 2. Ambiente

```bash
python3.12 -m venv venv-diffpen && source venv-diffpen/bin/activate
pip install -r requirements.txt
# torch: escolha o índice da sua GPU (na RX 9060 XT: repo.amd.com/rocm/whl/gfx120X-all/)
```

Na máquina atual (WSL) basta `source ~/env_htg.sh`, que ativa o venv e
exporta as variáveis do MIOpen.

## 3. Validar a GPU (obrigatório numa máquina nova)

```bash
python env/check_env.py                         # aborta se cair para CPU
python diagnostico/teste_conv_isolada.py
python diagnostico/teste_gradiente_sintetico.py
python diagnostico/teste_direcao_gradiente.py   # o cosseno tem de dar ~1,0
```

## 4. Modelos auxiliares (alinhador e leitor)

```bash
# alinhador: localiza as letras para desenhar os acentos (cabeça convolucional)
python scripts/treinar_alinhador.py --saida modelos/alinhador_iam.pt
# leitor independente: mede o CER na avaliação (cabeça Transformer)
python scripts/treinar_alinhador.py --cabeca transformer --saida modelos/leitor_iam_transformer.pt
```

## 5. Bases de treino

```bash
# base de acentos sobre palavras REAIS do IAM, com teto de 25% por palavra
python scripts/gerar_base_acentos.py --alinhador modelos/alinhador_iam.pt \
    --teto_por_palavra 0.25 --saida iam_acentuado_teto25
python scripts/criar_pares_iam.py --origem iam_acentuado_teto25 --destino iam_acentuado_teto25_pares
python scripts/mascaras_vogais.py --base iam_acentuado_teto25_pares      # só para --zona vogais (~70 min de CPU)

# base de palavras PORTUGUESAS geradas pelo DiffusionPen e acentuadas
# (o vocabulario_pt/ já está versionado e congelado: NÃO rode preparar_vocabulario_pt.py de novo)
python scripts/gerar_base_pt.py --saida iam_pt
python scripts/alinhar_pares.py --origem iam_pt --destino iam_pt_alinhado

# controle sem base: base_vazia/ já está no repositório
```

Conferências antes de treinar com pesos (não treinam nada):

```bash
python diagnostico/conferir_mascara_acento.py --base iam_pt_alinhado --peso 5 --saida <pasta>
python diagnostico/conferir_zona_vogais.py --base iam_acentuado_teto25_pares \
    --paineis <avaliacao>:<rotulo> --saida <pasta>
```

## 6. Treinar

```bash
python scripts/treinar.py experimentos/iam_acentuado_vogais.json --dry-run   # confere e mostra o comando
python scripts/treinar.py experimentos/iam_acentuado_vogais.json             # treina ou retoma
```

- Tudo vai para o `save_path` do experimento:
  - `models/ema_bloco_<N>ep.pt`: os modelos que avaliamos;
  - `treino.log`;
  - `config.jsonl`;
  - `amostras/<N>ep/`.
- **Para treinar várias horas no WSL:**
  `setsid nohup python -u scripts/treinar.py <exp> >> ~/treino.out 2>&1 < /dev/null &`.
  Mantenha uma sessão do WSL aberta.
- **Para acompanhar:**
  `tail -c 400 ~/treino.out | tr '\r' '\n' | tail -2` e `grep -a "^epoca" ~/treino.out`.

Experimentos disponíveis em `experimentos/`:

| arquivo | fase | o que é |
|---|---|---|
| `bressay_25.json`, `bressay_25_v2.json` | 3 | BRESSAY, pré-processamento v1 e v2 |
| `iam_acentuado.json`, `iam_acentuado_teto25.json` | 4 | acentos sintéticos sobre o IAM, sem e com teto |
| `iam_pt.json` | 5 | base pt gerada |
| `iam_pt_alinhado.json`, `iam_pt_peso5.json` | 6 | controle e peso 5 |
| `iam_pt_peso5_orig100.json` | 7b | peso 5 com 100% de originais |
| `iam_so_originais.json` | 7c | só originais (base vazia), 4 épocas |
| `iam_acentuado_pares.json` | 8 | IAM real + pares + peso 5 + zona vazia |
| `iam_acentuado_vogais.json` | 9 | IAM real + pares + peso 5 + zona vogais |

## 7. Gerar amostras de um checkpoint

```bash
python scripts/gerar_amostras.py --ckpt <modelo>/models/ema_bloco_12ep.pt --out am_x \
    --style DiffusionPen/style_models/iam_style_diffusionpen.pth --estilo_de iam \
    --styles 4 --seed 42 --palavras coração coracao provável provavel
```

Use sempre o **mesmo extrator de estilo do treino** (o do IAM).

## 8. Avaliar (validação)

```bash
python scripts/avaliar_pt.py --split val \
    --modelo meu_12ep=<modelo>/models/ema_bloco_12ep.pt \
    --bases <base_de_treino> --leitor modelos/leitor_iam_transformer.pt \
    --saida avaliacao_meu_val                        # ~35 min por modelo

python scripts/comparar_avaliacoes.py --modelo avaliacao_pt_val:iam \
    --modelo avaliacao_meu_val:meu_12ep --comparar meu_12ep iam
python scripts/medir_posicao_acento.py --avaliacao avaliacao_meu_val:meu_12ep --saida <pasta>
python scripts/comparar_paineis.py --modelo "IAM=avaliacao_pt_val:iam" \
    --modelo "meu=avaliacao_meu_val:meu_12ep" --saida saidas/comparacao_meu
python diagnostico/nitidez.py --modelo avaliacao_meu_val:meu_12ep
```

- **Ordem dos `--modelo`:** um processo por vez na GPU, e cada modelo leva
  ~35 min.
- **Para comparar modelos,** gere-os sempre com o mesmo protocolo. As
  palavras, os escritores e as sementes são fixos no código.
- **O teste** (`--split teste`) só no fim, uma vez.

## 9. Onde ficam as coisas (na máquina de treino)

Tudo isto fica no WSL e fora do git:

| o quê | onde |
|---|---|
| bases de treino | `iam_acentuado*/`, `iam_pt*/` |
| modelos treinados | `model_<experimento>/models/ema_bloco_<N>ep.pt` |
| amostras de bloco | `model_<experimento>/amostras/<N>ep/` |
| avaliações | `avaliacao_<nome>_val/` (`paineis/`, `paineis.tsv`, `resumo.md`) |
| figuras comparativas completas | `saidas/comparacao_*/` |

Resultados selecionados (tabelas e figuras) vão para
`diagnostico/resultados/` e são versionados.
