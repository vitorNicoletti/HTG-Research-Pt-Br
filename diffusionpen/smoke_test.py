"""Fase 1 - smoke test do DiffusionPen (ingles, pesos IAM prontos).

Gera 5 palavras em ingles. Se isto funcionar, VAE + UNet + encoder CANINE +
sampler DDIM estao todos operacionais, e qualquer falha na Fase 2 pode ser
atribuida ao diacritico e nao ao setup.

Uso (a partir da raiz do projeto):
    python diffusionpen/smoke_test.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
REPO = RAIZ / "diffusionpen" / "DiffusionPen"
SAIDA = RAIZ / "saidas" / "diffusionpen" / "smoke"

PALAVRAS = ["hello", "world", "research", "handwriting", "sample"]


def _env_rocm():
    """Ambiente necessario para o ROCm nesta maquina (WSL, instalacao minima,
    sem sudo). Ver LOG.md secao 'MIOpen JIT'.

    - LD_LIBRARY_PATH: libgomp.so.1, que o torch exige e o WSL nao traz.
    - CPLUS_INCLUDE_PATH: nao ha kernel de batch_norm pre-compilado para
      gfx1200, entao o MIOpen compila em runtime via HIPRTC e precisa dos
      headers de C++ e da glibc. O clang do HIPRTC honra esta variavel.

    Todos extraidos em ~/htg-tcc/syslibs com `apt-get download` + `dpkg -x`
    (nenhum exige root).
    """
    env = os.environ.copy()
    r = Path.home() / "htg-tcc" / "syslibs" / "root"
    if not r.exists():
        return env

    libs = r / "usr" / "lib" / "x86_64-linux-gnu"
    if libs.exists():
        ant = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{libs}:{ant}" if ant else str(libs)

    incs = [d for d in (
        r / "usr" / "include" / "c++" / "16",
        r / "usr" / "include" / "x86_64-linux-gnu" / "c++" / "16",
        r / "usr" / "include" / "x86_64-linux-gnu",
        r / "usr" / "include",
    ) if d.exists()]
    if incs:
        ant = env.get("CPLUS_INCLUDE_PATH", "")
        cam = ":".join(str(d) for d in incs)
        env["CPLUS_INCLUDE_PATH"] = f"{cam}:{ant}" if ant else cam
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", type=int, default=-1,
                    help="indice da classe de estilo (0-338). -1 = aleatorio "
                         "por palavra, como fazia o single_sampling original")
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
    # -1 replica o comportamento do single_sampling original, que sorteava um
    # estilo por palavra. Util no smoke test: expoe a variacao entre escritores.
    # A sonda da Fase 2 NAO usa isso - la o estilo precisa ser fixo.
    import random as _rnd
    if args.style < 0:
        _rnd.seed(0)  # sorteio aleatorio mas reproduzivel
        estilos = [_rnd.randint(0, 338) for _ in PALAVRAS]
    else:
        estilos = [args.style] * len(PALAVRAS)

    manifesto = [
        {"grupo": "smoke", "palavra": p, "seed": 0, "estilo": e,
         "arquivo": f"{p}__estilo{e:03d}.png"}
        for p, e in zip(PALAVRAS, estilos)
    ]
    print("estilos sorteados:", dict(zip(PALAVRAS, estilos)))
    caminho = SAIDA.parent / "manifesto_smoke.json"
    caminho.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    cmd = [
        sys.executable, "train.py",
        "--train_mode", "sampling",
        "--sampling_mode", "sonda",
        "--sonda_manifest", str(caminho),
        "--sonda_out", str(SAIDA),
        "--sonda_style", str(max(args.style, 0)),
        "--save_path", "./diffusionpen_iam_model_path",
        "--style_path", "./style_models/iam_style_diffusionpen.pth",
        "--device", args.device,
    ]

    if args.dry_run:
        print("comando (dry-run):")
        print("  cd", REPO)
        print(" ", " ".join(cmd))
        return

    r = subprocess.run(cmd, cwd=REPO, env=_env_rocm())
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
