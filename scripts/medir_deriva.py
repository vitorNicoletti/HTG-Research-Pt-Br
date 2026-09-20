"""Quanto o checkpoint se afastou do pre-treinado no IAM?

Esta e a metrica de PROGRESSO do fine-tune, nao de dano. O modelo precisa
viajar do IAM ate o BRESSAY; deriva pequena demais significa que ele saiu da
escrita nitida do IAM e ainda nao chegou na do BRESSAY -- o pior lugar.

Faixas medidas nos treinos ja feitos (ver htg-tcc-arquivo/modelos):

    0,10% - 0,16%   preso entre as duas distribuicoes: ilegivel
    2,1%  - 2,7%    BRESSAY legivel, com diacriticos  <-- o alvo
    12,9%           degradado de novo

    python scripts/medir_deriva.py CAMINHO/ema_ckpt.pt [outro.pt ...]
"""

import argparse
import os
import sys

import torch

PADRAO_BASE = "./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"


def deriva(base, alvo):
    num = den = 0.0
    n = 0
    for k, v in base.items():
        if k in alvo and alvo[k].shape == v.shape and v.is_floating_point():
            num += float((alvo[k].float() - v.float()).pow(2).sum())
            den += float(v.float().pow(2).sum())
            n += 1
    if not n:
        raise SystemExit("nenhum parametro em comum -- os checkpoints nao batem")
    return 100.0 * (num / den) ** 0.5, n


def faixa(d):
    if d < 0.5:
        return "cedo demais (preso entre IAM e BRESSAY)"
    if d < 2.0:
        return "chegando -- gere amostras"
    if d <= 3.5:
        return "FAIXA ALVO"
    if d < 8.0:
        return "passou do alvo -- gere e compare com o snapshot anterior"
    return "longe demais (a 12,9% o modelo degradou)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--base", default=os.environ.get("BASE_CKPT", PADRAO_BASE))
    a = p.parse_args()

    if not os.path.isfile(a.base):
        raise SystemExit(f"pre-treinado do IAM nao encontrado: {a.base}\n"
                         "passe --base ou defina BASE_CKPT")

    base = torch.load(a.base, map_location="cpu", weights_only=True)
    print(f"referencia: {a.base}\n")
    print(f"{'checkpoint':<44} {'deriva':>9}  situacao")
    for c in a.ckpts:
        if not os.path.isfile(c):
            print(f"{c:<44} {'--':>9}  arquivo nao existe")
            continue
        d, _ = deriva(base, torch.load(c, map_location="cpu", weights_only=True))
        print(f"{os.path.relpath(c):<44} {d:>8.4f}%  {faixa(d)}")


if __name__ == "__main__":
    main()
