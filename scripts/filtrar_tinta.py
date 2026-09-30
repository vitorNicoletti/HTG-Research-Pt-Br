"""Descarta do treino as palavras com tinta baixa demais para o pre-processamento v2.

O v2 amplia cada palavra ate a tinta ocupar os 64 px de altura. Uma palavra
com 10 px de tinta no recorte original e ampliada 6x e vira um bloco
pixelado que nem uma pessoa le -- treinar com isso ensina blocos. A altura da
tinta e medida pelo mesmo caminho do v2 (contraste, pauta removida, manchas
ignoradas), em pixels da imagem original.

Sem --altura_min: mede, imprime quanto sobra por corte e monta uma folha com
exemplos de cada faixa de altura, para escolher o corte olhando.

    python scripts/filtrar_tinta.py --origem ./bressay_split_25

Com --altura_min e --destino: grava o split filtrado. So o treino e filtrado;
validacao e teste sao copiados sem mudanca, como no reduzir_split.py.

    python scripts/filtrar_tinta.py --origem ./bressay_split_25 \\
        --altura_min 14 --destino ./bressay_split_25_v2
"""

import argparse
import collections
import json
import os
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
from utils import bressay_dataset as bd  # noqa: E402

DIAC = set("àáâãçéêíóôõúüÀÁÂÃÇÉÊÍÓÔÕÚÜ")
CORTES = (8, 10, 12, 14, 16, 18, 20)
FAIXAS = ((0, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 999))


def ler_tsv(caminho):
    return [l.split("\t")[:3] for l in open(caminho, encoding="utf-8").read().splitlines() if l.strip()]


def medir(linhas, imagens, cache):
    """Altura da tinta de cada linha do split, com cache em disco."""
    alturas = {}
    if os.path.isfile(cache):
        for l in open(cache, encoding="utf-8"):
            c, h = l.rstrip("\n").split("\t")
            alturas[c] = float(h)
    faltam = [c for c, _, _ in linhas if c not in alturas]
    if faltam:
        print(f"medindo {len(faltam)} imagens (cache: {cache})")
        with open(cache, "a", encoding="utf-8") as f:
            for k, c in enumerate(faltam, 1):
                alturas[c] = bd.altura_tinta_original(os.path.join(imagens, c))
                f.write(f"{c}\t{alturas[c]:.2f}\n")
                if k % 5000 == 0:
                    print(f"  {k}/{len(faltam)}")
    return alturas


def fonte(t):
    try:
        cam = subprocess.run(["fc-match", "-f", "%{file}", "sans:lang=pt"],
                             capture_output=True, text=True).stdout.strip()
        return ImageFont.truetype(cam, t)
    except Exception:
        return ImageFont.load_default()


def folha_faixas(linhas, alturas, imagens, destino, por_faixa=6, seed=0):
    ds = SimpleNamespace(args=SimpleNamespace(preproc="v2"))
    carregar = lambda c: bd.BRESSAY_Dataset.load_image(ds, os.path.join(imagens, c))
    rnd = random.Random(seed)
    f_rot, f_pal = fonte(17), fonte(13)
    LX, CW, RH = 150, 256, 64
    folha = Image.new("RGB", (LX + por_faixa * (CW + 8), 20 + len(FAIXAS) * (RH + 30)), "white")
    d = ImageDraw.Draw(folha)
    y = 20
    for lo, hi in FAIXAS:
        cand = [(c, t) for c, _, t in linhas if lo <= alturas[c] < hi]
        rot = f"{lo}-{hi} px" if hi < 999 else f">= {lo} px"
        d.text((8, y + 18), rot, fill=(140, 0, 0), font=f_rot)
        d.text((8, y + 42), f"{len(cand)} palavras", fill=(120, 120, 120), font=f_pal)
        for k, (c, t) in enumerate(rnd.sample(cand, min(por_faixa, len(cand)))):
            x = LX + k * (CW + 8)
            d.text((x, y), f"{t}  ({alturas[c]:.0f} px)", fill="black", font=f_pal)
            folha.paste(carregar(c), (x, y + 16))
            d.rectangle([x - 1, y + 15, x + CW, y + 16 + RH], outline=(170, 170, 170))
        y += RH + 30
    folha.save(destino)
    print("folha:", destino)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default="./bressay_split_25")
    ap.add_argument("--imagens", default=os.environ.get("BRESSAY_IMAGES", "./bressay/data/words"))
    ap.add_argument("--altura_min", type=float)
    ap.add_argument("--destino")
    ap.add_argument("--folha", default="./saidas/diffusionpen/preproc/faixas_altura_tinta.png")
    a = ap.parse_args()
    os.chdir(RAIZ)

    treino = ler_tsv(os.path.join(a.origem, "splits", "train.tsv"))
    alturas = medir(treino, a.imagens, os.path.join(a.origem, "alturas_tinta.tsv"))
    h = np.array([alturas[c] for c, _, _ in treino])
    ac = np.array([any(ch in DIAC for ch in t) for _, _, t in treino])
    print(f"\ntreino de {a.origem}: {len(treino)} palavras, {ac.sum()} com acento")
    print("altura da tinta no original (px), percentis 10/25/50/75/90: "
          + " / ".join(f"{np.percentile(h, p):.1f}" for p in (10, 25, 50, 75, 90)))
    esc = 64 / np.maximum(h, 1e-6)
    print(f"ampliacao do v2, mediana: {np.median(esc):.1f}x")
    print(f"\n{'corte':>6} | {'ficam':>6} | {'com acento':>10} | {'escritores':>10} | ampliacao max")
    for c in CORTES:
        f = h >= c
        esc_max = 64 / c
        n_esc = len({w for (_, w, _), ok in zip(treino, f) if ok})
        print(f"{c:>4}px | {f.sum():>6} | {(f & ac).sum():>10} | {n_esc:>10} | {esc_max:.1f}x")

    if a.altura_min is None:
        os.makedirs(os.path.dirname(a.folha), exist_ok=True)
        folha_faixas(treino, alturas, a.imagens, a.folha)
        return

    if not a.destino:
        sys.exit("--altura_min exige --destino")
    if os.path.abspath(a.destino) == os.path.abspath(a.origem):
        sys.exit("destino igual a origem")
    ficam = [l for l in treino if alturas[l[0]] >= a.altura_min]
    fora = [l for l in treino if alturas[l[0]] < a.altura_min]
    os.makedirs(os.path.join(a.destino, "splits"), exist_ok=True)
    with open(os.path.join(a.destino, "splits", "train.tsv"), "w", encoding="utf-8") as f:
        for l in ficam:
            f.write("\t".join(l) + "\n")
    with open(os.path.join(a.destino, "descartados_tinta.tsv"), "w", encoding="utf-8") as f:
        for c, w, t in fora:
            f.write(f"{c}\t{w}\t{t}\t{alturas[c]:.2f}\n")
    for nome in ("val.tsv", "test.tsv"):
        shutil.copyfile(os.path.join(a.origem, "splits", nome), os.path.join(a.destino, "splits", nome))
    shutil.copyfile(os.path.join(a.origem, "writers_dict.json"), os.path.join(a.destino, "writers_dict.json"))
    registro = {"origem": a.origem, "altura_min_tinta_px": a.altura_min,
                "medida": "utils/bressay_dataset.py::altura_tinta_original",
                "treino_antes": len(treino), "treino_depois": len(ficam),
                "com_acento_depois": sum(any(ch in DIAC for ch in t) for _, _, t in ficam)}
    reducao = os.path.join(a.origem, "reducao.json")
    if os.path.isfile(reducao):
        registro["reducao_da_origem"] = json.load(open(reducao, encoding="utf-8"))
    json.dump(registro, open(os.path.join(a.destino, "filtro_tinta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n{a.destino}: {len(ficam)} de {len(treino)} palavras de treino "
          f"({registro['com_acento_depois']} com acento); {len(fora)} em descartados_tinta.tsv")


if __name__ == "__main__":
    main()
