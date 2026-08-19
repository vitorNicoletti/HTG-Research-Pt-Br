"""Fase 2 - sonda de diacriticos do DiffusionPen.

Gera o manifesto da sonda a partir de comum/palavras.py e invoca o
sampling_mode 'sonda' adicionado ao train.py (ver patch_sonda_train.diff).

Roda o modelo UMA vez para todos os 60 itens - carregar UNet + VAE + CANINE por
palavra seria absurdamente lento.

Uso (a partir da raiz do projeto):
    python diffusionpen/sonda_diacriticos.py
    python diffusionpen/sonda_diacriticos.py --style 12 --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "comum"))
import palavras as P  # noqa: E402

REPO = RAIZ / "diffusionpen" / "DiffusionPen"
SAIDA = RAIZ / "saidas" / "diffusionpen" / "sonda"


def construir_manifesto():
    return [
        {
            "grupo": grupo,
            "palavra": palavra,
            "seed": seed,
            "arquivo": P.nome_arquivo(grupo, palavra, seed),
        }
        for grupo, palavra, seed in P.itens()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", type=int, default=12,
                    help="indice da classe de estilo, fixo para toda a sonda")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dry-run", action="store_true",
                    help="so escreve o manifesto e imprime o comando")
    args = ap.parse_args()

    if not REPO.exists():
        sys.exit(f"FALHA: repo nao encontrado em {REPO}")

    # Pesos que o README exige. Checar antes evita descobrir que faltam depois
    # de 40 minutos de geracao.
    faltando = [d for d in ("saved_iam_data", "style_models",
                            "diffusionpen_iam_model_path")
                if not (REPO / d).exists()]
    if faltando and not args.dry_run:
        sys.exit("FALHA: artefatos ausentes em DiffusionPen/: "
                 + ", ".join(faltando)
                 + "\nBaixe de https://huggingface.co/konnik/DiffusionPen")

    SAIDA.mkdir(parents=True, exist_ok=True)
    manifesto = construir_manifesto()
    caminho_manifesto = SAIDA.parent / "manifesto_sonda.json"
    caminho_manifesto.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"manifesto: {caminho_manifesto}  ({len(manifesto)} itens)")
    print(f"nao-ASCII exercitados: {' '.join(P.caracteres_nao_ascii())}")
    print(f"estilo fixo: {args.style}   seeds: {P.SEEDS}")

    cmd = [
        sys.executable, "train.py",
        "--train_mode", "sampling",
        "--sampling_mode", "sonda",
        "--sonda_manifest", str(caminho_manifesto),
        "--sonda_out", str(SAIDA),
        "--sonda_style", str(args.style),
        "--save_path", "./diffusionpen_iam_model_path",
        "--style_path", "./style_models/iam_style_diffusionpen.pth",
        "--device", args.device,
    ]

    if args.dry_run:
        print("\ncomando (dry-run):")
        print("  cd", REPO)
        print(" ", " ".join(cmd))
        return

    print(f"\nexecutando em {REPO} ...\n")
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"FALHA: train.py saiu com codigo {r.returncode}")

    gerados = sorted(SAIDA.glob("*.png"))
    print(f"\nimagens geradas: {len(gerados)}/{len(manifesto)}")
    if len(gerados) < len(manifesto):
        print("AVISO: faltaram imagens - a folha de contato vai marcar 'ausente'")

    print("\nproximo passo:")
    print(f"  python comum/folha_contato.py {SAIDA}")


if __name__ == "__main__":
    main()
