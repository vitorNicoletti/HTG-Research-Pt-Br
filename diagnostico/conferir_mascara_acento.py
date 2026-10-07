"""Confere a mascara do peso no acento ANTES de treinar com ela.

O train.py com --peso_acento multiplica o erro de ruido nas celulas do latente
marcadas pela mascara do IAMAcentuadoDataset (diferenca acentuada x par). Este
script passa por TODAS as amostras da base e confere:
  1. toda acentuada e todo par tem mascara nao vazia, e a do par e identica a
     da sua acentuada; sem_acento tem mascara zerada;
  2. quanto do latente a mascara cobre (por amostra e no lote de treino, com a
     fracao de originais do IAM do experimento) -- e isso que diz quanto o
     peso pesa;
  3. a formula da loss com peso 1 da exatamente a MSE do nn.MSELoss;
  4. figura: acentuada, par e mascara sobreposta, para olhar se ela pega o
     sinal e so o sinal.

    python diagnostico/conferir_mascara_acento.py --base iam_pt_alinhado --peso 5 \
        --saida diagnostico/resultados/peso_acento
"""

import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
from utils.iam_acentuado_dataset import (FORMA_LATENTE, IAMAcentuadoDataset,  # noqa: E402
                                         preprocessar_iam)


def figura(ds, amostras, destino):
    """Uma linha por acentuada: acentuada com a mascara em vermelho | par."""
    h, w = FORMA_LATENTE
    linhas = []
    for c in amostras:
        (ca, ra), (cp, rp) = ds.pares[c]
        ia = preprocessar_iam(Image.open(ca).convert("RGB"), ra)
        ip = preprocessar_iam(Image.open(cp).convert("RGB"), rp)
        m = ds.mascara(c).numpy() > 0
        sob = ia.convert("RGB").copy()
        d = ImageDraw.Draw(sob, "RGBA")
        for y, x in zip(*np.nonzero(m)):
            d.rectangle([x * 256 // w, y * 64 // h, (x + 1) * 256 // w - 1, (y + 1) * 64 // h - 1],
                        fill=(255, 0, 0, 70))
        linha = Image.new("RGB", (256 * 3 + 20, 64), "white")
        linha.paste(sob, (0, 0))
        linha.paste(ia.convert("RGB"), (266, 0))
        linha.paste(ip.convert("RGB"), (532, 0))
        linhas.append((linha, ra))
    rotulo_w = 110
    tela = Image.new("RGB", (rotulo_w + linhas[0][0].width, 18 + 66 * len(linhas)), "white")
    dr = ImageDraw.Draw(tela)
    for x, t in ((rotulo_w, "mascara (vermelho)"), (rotulo_w + 266, "acentuada"), (rotulo_w + 532, "par")):
        dr.text((x + 2, 3), t, fill="black")
    for k, (linha, r) in enumerate(linhas):
        tela.paste(linha, (rotulo_w, 18 + 66 * k))
        dr.text((4, 18 + 66 * k + 26), r, fill="black")
    tela.save(destino)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--peso", type=float, default=5.0)
    ap.add_argument("--iam_originais", type=float, default=0.3, help="a do experimento (para a cobertura no lote)")
    ap.add_argument("--n_figura", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    args = SimpleNamespace(peso_acento=a.peso, iam_originais=a.iam_originais, max_samples=0)
    ds = IAMAcentuadoDataset(a.base, transforms=None, args=args)
    man = [json.loads(l) for l in open(os.path.join(a.base, "manifesto.jsonl"), encoding="utf-8")]
    tipo = {os.path.join(a.base, x["arquivo"]): x["tipo"] for x in man}

    cobertura = {"acentuada": [], "par": [], "sem_acento": []}
    problemas = []
    mascaras = {}
    for c, _, t, sintetica in ds.data:
        if not sintetica:
            continue
        m = ds.mascara(c)
        tp = tipo[c]
        cobertura[tp].append(float(m.mean()))
        if tp == "sem_acento" and m.any():
            problemas.append(f"sem_acento com mascara: {c}")
        if tp in ("acentuada", "par"):
            if not m.any():
                problemas.append(f"{tp} com mascara vazia: {c} ({t})")
            mascaras[c] = m
    for c, ((ca, _), (cp, _)) in ds.pares.items():
        if c == ca and not torch.equal(mascaras[ca], mascaras[cp]):
            problemas.append(f"mascara do par difere da acentuada: {ca}")

    # loss com peso 1 = MSE original, bit a bit
    g = torch.Generator().manual_seed(0)
    ruido, pred = torch.randn(32, 4, *FORMA_LATENTE, generator=g), torch.randn(32, 4, *FORMA_LATENTE, generator=g)
    msk = (torch.rand(32, *FORMA_LATENTE, generator=g) < 0.1).float()
    l1 = ((1.0 + 0.0 * msk[:, None]) * (ruido - pred) ** 2).mean()
    igual_mse = bool(torch.equal(l1, torch.nn.MSELoss()(ruido, pred)))

    # cobertura no lote: fracao das celulas do latente na mascara, sobre TODAS
    # as amostras de treino (originais do IAM e sem_acento entram com zero)
    soma = sum(sum(v) for v in cobertura.values())
    cob_lote = soma / len(ds.data)
    cob_ac = np.array(cobertura["acentuada"])
    # parte da loss da amostra que vem da mascara, supondo erro igual dentro e fora
    parte = lambda c, p: p * c / (p * c + (1 - c))  # noqa: E731
    rel = {
        "base": a.base, "peso": a.peso, "iam_originais": a.iam_originais,
        "amostras": {k: len(v) for k, v in cobertura.items()} | {"originais_iam": ds.n_originais},
        "cobertura_por_acentuada": {"media": float(cob_ac.mean()), "p5": float(np.percentile(cob_ac, 5)),
                                    "mediana": float(np.median(cob_ac)), "p95": float(np.percentile(cob_ac, 95)),
                                    "max": float(cob_ac.max())},
        "cobertura_no_lote": cob_lote,
        "parte_da_loss_na_mascara_amostra_mediana": {"peso_1": parte(float(np.median(cob_ac)), 1.0),
                                                     f"peso_{a.peso:g}": parte(float(np.median(cob_ac)), a.peso)},
        "parte_da_loss_na_mascara_lote": {"peso_1": parte(cob_lote, 1.0), f"peso_{a.peso:g}": parte(cob_lote, a.peso)},
        "aumento_da_loss_media_no_lote": 1 + (a.peso - 1) * cob_lote,
        "loss_peso_1_igual_mse": igual_mse,
        "problemas": problemas,
    }
    with open(os.path.join(a.saida, "mascaras.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)
    print(json.dumps(rel, ensure_ascii=False, indent=2))

    acentuadas = sorted(c for c in mascaras if tipo[c] == "acentuada")
    figura(ds, random.Random(a.seed).sample(acentuadas, min(a.n_figura, len(acentuadas))),
           os.path.join(a.saida, "mascaras.png"))
    if problemas or not igual_mse:
        sys.exit(f"ERRO: {len(problemas)} problemas; loss peso 1 = MSE: {igual_mse}")
    print("ok")


if __name__ == "__main__":
    main()
