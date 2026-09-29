# CLAUDE.md

Research repo (PUCPR undergraduate capstone) studying **diacritic fidelity in handwritten text generation (HTG) for Brazilian Portuguese**. Models trained on English (IAM) fail on `ã ç é ê ó õ ú`, and aggregate metrics (FID/KID/CER) hide it. The work has three parts:
1. **Probe**: a minimal-pair probe of pre-trained DiffusionPen.
2. **Fine-tune**: fine-tuning DiffusionPen on **BRESSAY**, a Brazilian Portuguese handwriting dataset.
3. **Metric**: a diacritic-specific evaluation metric.

All docs, code identifiers, comments and commit messages are in **Portuguese**. Keep new code and docs in Portuguese, matching the existing style: ASCII-only comments and docstrings in `.py`/`.sh` files, no accents; commit messages are short imperative sentences.

## Layout

- `DiffusionPen/` — upstream clone (`github.com/koninik/DiffusionPen`), **gitignored**, at the repo root. Weights live inside it: `diffusionpen_iam_model_path/models/{ckpt,ema_ckpt}.pt` and `style_models/iam_style_diffusionpen.pth`.
- `diffusionpen_mods/` — **fixed replacement files** for the clone: `train.py`, `style_encoder_train.py`, `utils/bressay_dataset.py`. They are full-file overwrites, not a patch. Its README explains each fix.
- `scripts/`
  - `aplicar_mods.sh` — copies `diffusionpen_mods/` into the clone. Must run after every `git pull` or re-clone, because the clone is gitignored and a pull doesn't update it. `--conferir` only reports differences.
  - `treinar.sh` — the BRESSAY fine-tune driver (see below).
  - `gerar_amostras.py` — generates samples from a checkpoint. `--style` must be the same style encoder used in training.
  - `medir_deriva.py` — measures weight drift of a checkpoint relative to the IAM base.
  - `preparar_split.py` — builds `bressay_split/` from the raw BRESSAY word crops (quality filters, official page partition).
  - `reduzir_split.py` — builds a smaller split (for example `bressay_split_25/`). It keeps a fraction of each writer's training words, with at least 5 per writer, copies val/test unchanged, and records its parameters in `reducao.json`. Train on it with `SPLIT=./bressay_split_25 bash scripts/treinar.sh`, which passes `--dataset_folder` to `train.py`.
- `bressay_split/` — versioned splits: `splits/{train,val,test}.tsv` (`page/file.png<TAB>writer_id<TAB>transcription`) and `writers_dict.json`. Writers are disjoint across splits. The raw images (`bressay/`, 4.3 GB) are gitignored.
- `sonda/` — the Phase 1–2 probe of pre-trained DiffusionPen:
  - `smoke_test.py` and `sonda_diacriticos.py` write a JSON manifest, then run `train.py --sampling_mode sonda` with `cwd=DiffusionPen`.
  - `patches_diffusionpen.diff` adds that `sonda` mode (additive patch).
- `comum/` — `palavras.py` (canonical probe word list: 4 groups × 5 words × 3 seeds plus minimal pairs; don't edit without logging it in LOG.md) and `folha_contato.py` (labeled contact sheet).
- `avaliacao_diacriticos/` — the diacritic metric, independent of training:
  - **E1 (presence):** is there ink in the diacritic band?
  - **E2 (base integrity):** TrOCR CER with target and prediction folded to ASCII.
  - Also calibration, a human-annotation kappa tool, and `testes_metrica.py`, a synthetic known-answer test suite.
  - `ESTADO.md` holds its status. `trocr_tokenizer/` is versioned on purpose.
- `diagnostico/` — GPU numerical sanity tests (CPU vs GPU forward/gradient) and `treinar_cpu.py`.
- `docs/pre_processamento.md` — preprocessing write-up, with figures from `docs/figuras/gerar_figuras.py`.
- `env/check_env.py` — GPU/ROCm check; exits non-zero on CPU fallback.
- `flake.nix` — Nix dev shells: `nix develop .#rocm` (the default), `.#cuda` and `.#cpu`.
- `LOG.md` (dated execution diary), `ACHADOS.md` (findings), `STATUS.md`, `README.md`.

## BRESSAY fine-tune

Run from the repo root. All paths are relative (`./DiffusionPen`, `./bressay_split`, `./bressay/data/words`, which can be overridden with `BRESSAY_IMAGES`).

```bash
bash scripts/aplicar_mods.sh
SAVE_PATH=./model_bressay BLOCO=5 ALVO=40 bash scripts/treinar.sh
```

- **Blocks.** `treinar.sh` trains `BLOCO` epochs at a time, up to `ALVO` epochs. It stops early only if the Python run exits with a code other than 0 or 3.
  - The first block loads the IAM weights with `--pretrained_path`; later blocks resume with `--load_check True`. The two flags are mutually exclusive.
  - Exit code 3 comes from `--abort_after`: too many non-finite batches in a row. The script relaunches from the last checkpoint.
  - After each block it snapshots `ema_bloco_<N>ep.pt`, runs `medir_deriva.py`, and writes `./amostras_<N>ep`.
- **Hyperparameters.** lr 2e-5, batch size 32, 12 workers, `--max_samples 0` (full split), `--sample_every 0`, checkpoint every 500 steps.
- **What trains.** Only the UNet. The style encoder (`--style_path`, which defaults to the IAM one), CANINE and the VAE are frozen. With style features present the writer ID is ignored, and `style_classes` stays at 339 only so the IAM checkpoint loads.
- **Resuming.** `models/estado.pt` stores the epoch and `ema.step`. Without it, resuming would overwrite the EMA.
- **When to stop.** Judge by drift and by generated samples, not by epoch count or MSE. Drift is `‖W − W_IAM‖ / ‖W_IAM‖` over the trainable weights (CANINE excluded), printed by `medir_deriva.py`.
  - The ranges disagree between files. `medir_deriva.py` (which prints the verdict) says: <1% too early, 1–2.5% "chegando", **2.5–4.0% target** (best old run 2.5–3.2%), ~9.5% degraded. The comment in `treinar.sh` says 2.1–2.7%. Trust `medir_deriva.py`.
  - All ranges come from runs on the defective RX 6600 XT, so they are a rough guide only; generated samples are the real criterion.

  MSE improved in the same epoch that generation broke, so it is not a signal.
- **Built-in sample grid.** Keep `--sample_every 0`. The in-training grid tokenizes with `max_length=200` while training uses 40, so its samples are not valid. Use `scripts/gerar_amostras.py` instead.

## Other commands

```bash
python env/check_env.py
python sonda/smoke_test.py [--dry-run]; python sonda/sonda_diacriticos.py [--style 12] [--dry-run]
python comum/folha_contato.py saidas/diffusionpen/sonda
python scripts/gerar_amostras.py --ckpt ./model_x/models/ema_ckpt.pt --out ./am_x --style <same .pth as training>
python avaliacao_diacriticos/testes_metrica.py   # the only automated test suite
python diagnostico/teste_conv_isolada.py; python diagnostico/teste_gradiente_sintetico.py; python diagnostico/teste_direcao_gradiente.py
```

There is no linter or build. `--dry-run` on the probe scripts writes the manifest without loading the model.

## Rules and gotchas

- **Run `diagnostico/` on any new GPU before training on it.** The RX 6600 XT the fine-tune was developed on computes wrong-direction gradients for batches larger than 2: cosine similarity with the CPU gradient is 0.36. Every model trained on that card is invalid. Sampling (forward only) is fine.
- **Never accept a silent CPU fallback.** Changes to upstream code must not change generative behavior without explicit approval. The same goes for library up- or downgrades.
- **Stable Diffusion path.** `runwayml/stable-diffusion-v1-5` has been taken down from Hugging Face. `treinar.sh` and `gerar_amostras.py` now default to `stable-diffusion-v1-5/stable-diffusion-v1-5` and can be overridden with the `SD` env var. `avaliacao_diacriticos/gerar_pares.py` and `diagnostico/*` still hardcode the old ID.
- **Boolean flags.** `train.py` flags declared with `type=bool` treat any non-empty string as True: `--load_check False` means True, so omit the flag instead.
- **Current GPU machine.** Windows 11 with WSL2 Ubuntu 26.04 and an RX 9060 XT (gfx1200, 17 GB VRAM, 15 GB RAM). The old setup lives in `~/htg-tcc`, which uses the pre-reorganization layout `diffusionpen/DiffusionPen/`, with venv `venv-diffpen` (Python 3.12, torch from `repo.amd.com/rocm/whl/gfx120X-all/`).
  - Before running there, export `LD_LIBRARY_PATH=~/htg-tcc/syslibs/root/usr/lib/x86_64-linux-gnu` (libgomp).
  - Also export `CPLUS_INCLUDE_PATH` pointing to the `include/c++/16` headers under that root; MIOpen JIT-compiles kernels at runtime and needs them.
  - `treinar.sh` doesn't set either variable.
  - `HSA_OVERRIDE_GFX_VERSION` is set only on gfx1032 cards; it would be harmful on any other card.
- **Docs drift.** `STATUS.md` is outdated (dated 2026-08-19, from before BRESSAY). For the metric's status, `avaliacao_diacriticos/ESTADO.md` is the reference. For the fine-tune there is no up-to-date status doc: use `diffusionpen_mods/README.md`, `diagnostico/README.md`, the comments in `treinar.sh` and `LOG.md`.
- **Label files.** `train.py` rewrites `letter2index.json`/`index2letter.json` into the current working directory on import. They're only used by `--model_name wordstylist`; on the diffusionpen path, text goes through CANINE, which accepts diacritics as Unicode codepoints.
