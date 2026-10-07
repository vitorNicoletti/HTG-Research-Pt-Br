"""Quanto o checkpoint se afastou do pre-treinado no IAM?

Esta e a metrica de PROGRESSO do fine-tune, nao de dano. O modelo precisa
viajar do IAM ate o BRESSAY; deriva pequena demais significa que ele saiu da
escrita nitida do IAM e ainda nao chegou na do BRESSAY -- o pior lugar.

A deriva e medida sobre os pesos TREINAVEIS. O state_dict tem 169,9M
parametros, mas 132,1M deles sao o encoder de texto CANINE, que fica
congelado e nunca muda -- inclui-lo dilui a medida em ~3,6x e faz o modelo
parecer mais intacto do que esta.

Faixas medidas nos treinos ja feitos (ver htg-tcc-arquivo/modelos):

    0,5%           1 epoca: mal saiu do IAM
    2,5% - 3,2%    BRESSAY legivel, com diacriticos  <-- o alvo
    9,5%           degradado de novo

    python scripts/medir_deriva.py CAMINHO/ema_ckpt.pt [outro.pt ...]
"""

import argparse
import os
import sys

import torch

PADRAO_BASE = "./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"


def deriva(base, alvo):
    """Devolve (deriva dos treinaveis, deriva do state_dict inteiro).

    Acumula em float64: somar 170M de produtos em float32 perde precisao o
    bastante para o cosseno dar 1,04, que e impossivel.
    """
    nt = dt = ns = ds = 0.0
    n = 0
    for k, v in base.items():
        if k in alvo and alvo[k].shape == v.shape and v.is_floating_point():
            x = v.double()
            e = float((alvo[k].double() - x).pow(2).sum())
            r = float(x.pow(2).sum())
            nt += e
            dt += r
            n += 1
            if ".text_encoder." not in k:   # CANINE congelado
                ns += e
                ds += r
    if not n:
        raise SystemExit("nenhum parametro em comum -- os checkpoints nao batem")
    return 100.0 * (ns / ds) ** 0.5, 100.0 * (nt / dt) ** 0.5


def faixa(d):
    if d < 1.0:
        return "cedo demais (preso entre IAM e BRESSAY)"
    if d < 2.5:
        return "chegando -- gere amostras"
    if d <= 4.0:
        return "FAIXA ALVO"
    if d < 7.0:
        return "passou do alvo -- compare com o snapshot anterior"
    return "longe demais (a 9,5% o modelo degradou)"


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
    print(f"{'checkpoint':<40} {'treinaveis':>11} {'total':>8}  situacao")
    for c in a.ckpts:
        if not os.path.isfile(c):
            print(f"{c:<40} {'--':>11} {'--':>8}  arquivo nao existe")
            continue
        d, dt = deriva(base, torch.load(c, map_location="cpu", weights_only=True))
        print(f"{os.path.relpath(c):<40} {d:>10.3f}% {dt:>7.3f}%  {faixa(d)}")


if __name__ == "__main__":
    main()
