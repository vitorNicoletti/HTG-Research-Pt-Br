"""Comparativo visual e numerico do pre-processamento v1 vs v2 do BRESSAY.

Folha: para cada palavra, o recorte original em pixels reais (ampliado 2x sem
interpolacao so para enxergar), o que o modelo recebe hoje (v1) e o que passa
a receber (v2). Embaixo, recortes do IAM no pre-processamento do IAM -- a
aparencia que o modelo pre-treinado conhece.

Numeros, sobre --n_stats palavras sorteadas do treino: fracao com pauta
removida, altura e largura ocupadas pela tinta na imagem 256x64, e quantas
ficaram sem caixa de tinta no v2.

    python diagnostico/comparar_preproc.py --saida saidas/diffusionpen/preproc
"""

import argparse
import glob
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
from utils import bressay_dataset as bd  # noqa: E402

DIAC = set("àáâãçéêíóôõúüÀÁÂÃÇÉÊÍÓÔÕÚÜ")


def fonte(t):
    try:
        cam = subprocess.run(["fc-match", "-f", "%{file}", "sans:lang=pt"],
                             capture_output=True, text=True).stdout.strip()
        return ImageFont.truetype(cam, t)
    except Exception:
        return ImageFont.load_default()


def ocupacao(im):
    """(altura, largura) da caixa de tinta como fracao da imagem 256x64."""
    m = bd._binarizar(np.asarray(im.convert("L"), dtype=np.float32))
    cx = bd._caixa(m)
    if cx is None:
        return 0.0, 0.0
    x0, x1, y0, y1 = cx
    return (y1 - y0) / 64, (x1 - x0) / 256


def fileiras_de_pauta(caminho):
    """Quantas fileiras de pauta o v2 apagaria nesta imagem."""
    g = np.asarray(Image.open(caminho).convert("L"), dtype=np.float32)
    lo, hi = np.percentile(g, bd.P_TINTA), np.percentile(g, bd.P_FUNDO)
    if hi - lo >= 8:
        g = np.clip((g - lo) / (hi - lo), 0, 1) * 255
    import cv2
    h, w = g.shape
    g = np.clip(cv2.resize(g, (max(1, round(w * 64.0 / h)), 64),
                           interpolation=cv2.INTER_CUBIC), 0, 255)
    return bd._remover_pauta(g, bd._binarizar(g))[2]


def prep_iam(caminho):
    """Pre-processamento do IAM (utils/iam_dataset.py): altura 64, cabe em 256."""
    im = Image.open(caminho).convert("RGB")
    w, h = im.size
    im = im.resize((max(1, int(w * 64 / h)), 64))
    return ImageOps.pad(im, size=(256, 64), color="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="./bressay_split")
    ap.add_argument("--n_folha", type=int, default=24)
    ap.add_argument("--n_stats", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--saida", default="./saidas/diffusionpen/preproc")
    a = ap.parse_args()
    os.chdir(RAIZ)
    os.makedirs(a.saida, exist_ok=True)

    ds1 = bd.BRESSAY_Dataset(a.split, "train", args=SimpleNamespace(max_samples=0, preproc="v1"))
    ds2 = bd.BRESSAY_Dataset(a.split, "train", args=SimpleNamespace(max_samples=0, preproc="v2"))
    rnd = random.Random(a.seed)

    # ---- folha: metade com diacritico, metade qualquer ----
    com = [i for i, d in enumerate(ds1.data) if any(c in DIAC for c in d[1])]
    idx = rnd.sample(com, a.n_folha // 2) + rnd.sample(range(len(ds1.data)), a.n_folha - a.n_folha // 2)
    f_rot, f_tit = fonte(15), fonte(20)
    COL = [("original (pixels reais, 2x)", 0), ("v1: o que o modelo recebe hoje", 1),
           ("v2: sem pauta, recorte justo, escala do IAM", 2)]
    LX, CW, RH, TOPO = 150, 256, 64, 40
    iam = sorted(glob.glob(os.path.join(RAIZ, "DiffusionPen/iam_data/words/*/*/*.png")))
    iam = [iam[k] for k in rnd.sample(range(len(iam)), 6)] if iam else []
    linhas_iam = (len(iam) + 2) // 3
    altura = TOPO + len(idx) * (RH + 8) + (50 + linhas_iam * (RH + 8) if iam else 0) + 20
    folha = Image.new("RGB", (LX + 3 * (CW + 12) + 10, altura), "white")
    d = ImageDraw.Draw(folha)
    for titulo, c in COL:
        d.text((LX + c * (CW + 12), 12), titulo, fill=(140, 0, 0), font=f_rot)
    y = TOPO
    for i in idx:
        cam, t, _ = ds1.data[i]
        orig = Image.open(cam).convert("RGB")
        d.text((8, y + 14), t, fill="black", font=f_tit)
        d.text((8, y + 40), f"{orig.size[0]}x{orig.size[1]}", fill=(120, 120, 120), font=f_rot)
        caixa = Image.new("RGB", (CW, RH), (235, 235, 235))
        o2 = orig.resize((orig.size[0] * 2, orig.size[1] * 2), Image.NEAREST)
        o2.thumbnail((CW, RH))
        caixa.paste(o2, ((CW - o2.size[0]) // 2, (RH - o2.size[1]) // 2))
        for c, im in enumerate((caixa, ds1.load_image(cam), ds2.load_image(cam))):
            x = LX + c * (CW + 12)
            folha.paste(im, (x, y))
            d.rectangle([x - 1, y - 1, x + CW, y + RH], outline=(170, 170, 170))
        y += RH + 8
    if iam:
        y += 10
        d.text((8, y), "referencia: recortes do IAM no pre-processamento do IAM "
                       "(o que o modelo pre-treinado conhece)", fill=(140, 0, 0), font=f_rot)
        y += 30
        for k, cam in enumerate(iam):
            x = LX + (k % 3) * (CW + 12)
            folha.paste(prep_iam(cam), (x, y + (k // 3) * (RH + 8)))
            d.rectangle([x - 1, y + (k // 3) * (RH + 8) - 1, x + CW, y + (k // 3) * (RH + 8) + RH],
                        outline=(170, 170, 170))
    caminho = os.path.join(a.saida, "comparacao_v1_v2.png")
    folha.save(caminho)
    print("folha:", caminho)

    # ---- numeros ----
    amostra = rnd.sample(range(len(ds1.data)), min(a.n_stats, len(ds1.data)))
    o1, o2, pauta, vazios = [], [], 0, 0
    for i in amostra:
        cam = ds1.data[i][0]
        o1.append(ocupacao(ds1.load_image(cam)))
        oc = ocupacao(ds2.load_image(cam))
        o2.append(oc)
        vazios += oc == (0.0, 0.0)
        pauta += fileiras_de_pauta(cam) > 0
    o1, o2 = np.array(o1), np.array(o2)
    print(f"\n{len(amostra)} palavras sorteadas do treino")
    print(f"com pauta removida no v2          : {pauta} ({100 * pauta / len(amostra):.0f}%)")
    print(f"altura da tinta / 64   v1 -> v2   : mediana {np.median(o1[:, 0]):.2f} -> {np.median(o2[:, 0]):.2f}")
    print(f"largura da tinta / 256 v1 -> v2   : mediana {np.median(o1[:, 1]):.2f} -> {np.median(o2[:, 1]):.2f}")
    print(f"v2 sem caixa de tinta             : {vazios}")
    if iam:
        todos = sorted(glob.glob(os.path.join(RAIZ, "DiffusionPen/iam_data/words/*/*/*.png")))
        oi = np.array([ocupacao(prep_iam(todos[k])) for k in rnd.sample(range(len(todos)), 500)])
        print(f"referencia IAM (500 recortes)     : altura {np.median(oi[:, 0]):.2f}, "
              f"largura {np.median(oi[:, 1]):.2f}")


if __name__ == "__main__":
    main()
