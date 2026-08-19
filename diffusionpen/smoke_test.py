"""Fase 1 - smoke test do DiffusionPen (ingles, pesos IAM prontos).

Gera 5 palavras em ingles. Se isto funcionar, VAE + UNet + encoder CANINE +
sampler DDIM estao todos operacionais, e qualquer falha na Fase 2 pode ser
atribuida ao diacritico e nao ao setup.

Uso (a partir da raiz do projeto):
    python diffusionpen/smoke_test.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
REPO = RAIZ / "diffusionpen" / "DiffusionPen"
SAIDA = RAIZ / "saidas" / "diffusionpen" / "smoke"

PALAVRAS = ["hello", "world", "research", "handwriting", "sample"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not REPO.exists():
        sys.exit(f"FALHA: repo nao encontrado em {REPO}")

    faltando = [d for d in ("saved_iam_data", "style_models",
                            "diffusionpen_iam_model_path")
                if not (REPO / d).exists()]
    if faltando and not args.dry_run:
        sys.exit("FALHA: artefatos ausentes em DiffusionPen/: "
                 + ", ".join(faltando)
                 + "\nBaixe de https://huggingface.co/konnik/DiffusionPen")

    SAIDA.mkdir(parents=True, exist_ok=True)
    manifesto = [
        {"grupo": "smoke", "palavra": p, "seed": 0, "arquivo": f"{p}.png"}
        for p in PALAVRAS
    ]
    caminho = SAIDA.parent / "manifesto_smoke.json"
    caminho.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    cmd = [
        sys.executable, "train.py",
        "--train_mode", "sampling",
        "--sampling_mode", "sonda",
        "--sonda_manifest", str(caminho),
        "--sonda_out", str(SAIDA),
        "--sonda_style", str(args.style),
        "--save_path", "./diffusionpen_iam_model_path",
        "--style_path", "./style_models/iam_style_diffusionpen.pth",
        "--device", args.device,
    ]

    if args.dry_run:
        print("comando (dry-run):")
        print("  cd", REPO)
        print(" ", " ".join(cmd))
        return

    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit(f"FALHA: train.py saiu com codigo {r.returncode}")

    gerados = sorted(SAIDA.glob("*.png"))
    print(f"\nimagens geradas: {len(gerados)}/{len(PALAVRAS)}")
    for g in gerados:
        print("  ", g.name)

    if len(gerados) != len(PALAVRAS):
        sys.exit("FALHA: nem todas as palavras foram geradas")

    print("\nCRITERIO DE ACEITACAO: inspecione as 5 imagens.")
    print("O texto tem de corresponder ao pedido e ser legivel.")
    print("Se sair ruido -> pesos mal posicionados. NAO prosseguir para a Fase 2.")


if __name__ == "__main__":
    main()
