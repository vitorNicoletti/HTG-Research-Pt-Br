"""Resolucao das imagens do BRESSAY e quanto sobra filtrando por ela.

Duas medidas, que sustentam o achado 7 do ACHADOS.md:

  1. distribuicao de tamanhos de todas as imagens do zip oficial (paginas,
     paragrafos, linhas, palavras), lendo so o cabecalho de cada PNG;
  2. para o split filtrado (bressay_split/), a altura mediana das palavras de
     cada pagina, e quantas palavras (e quantas com acento) sobrariam no
     treino com um corte minimo nessa altura -- o CORTE do preparar_split.py.

    python diagnostico/resolucao_bressay.py --zip ~/Downloads/bressay.zip \
        --palavras ./bressay/data/words --split ./bressay_split
"""

import argparse
import collections
import io
import statistics
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DIAC = set("àáâãçéêíóôõúüÀÁÂÃÇÉÊÍÓÔÕÚÜ")
CORTES = (0, 25, 30, 35, 40, 45, 50)


def q(v):
    return " / ".join(str(int(x)) for x in np.percentile(v, [0, 10, 50, 90, 100]))


def varrer_zip(caminho):
    z = zipfile.ZipFile(caminho)
    tam = collections.defaultdict(list)
    for info in z.infolist():
        partes = info.filename.split("/")
        if not info.filename.endswith(".png") or len(partes) < 4 or partes[1] != "data":
            continue
        with z.open(info) as f:
            w, h = Image.open(io.BytesIO(f.read(64 * 1024))).size
        tam[partes[2]].append((w, h))

    print("== todas as imagens do zip (percentis 0 / 10 / 50 / 90 / 100)")
    for tipo in ("pages", "paragraphs", "lines", "words"):
        v = tam[tipo]
        print(f"{tipo:11} n={len(v):7}  largura {q([w for w, _ in v]):28} "
              f"altura {q([h for _, h in v])}")
    pg = tam["pages"]
    print(f"paginas com lado >= 1500 px: {sum(max(t) >= 1500 for t in pg)} de {len(pg)}")
    wd = [h for _, h in tam["words"]]
    print(f"palavras com altura >= 48 px: {sum(h >= 48 for h in wd)} de {len(wd)}\n")


def cortes(palavras, split):
    por_pagina = collections.defaultdict(list)
    for s in ("train", "val", "test"):
        for l in (split / "splits" / f"{s}.tsv").read_text(encoding="utf-8").splitlines():
            c, _, t = l.split("\t")[:3]
            with Image.open(palavras / c) as im:
                h = im.size[1]
            por_pagina[c.split("/")[0]].append((s, t, h))

    mediana = {pg: statistics.median(h for _, _, h in v) for pg, v in por_pagina.items()}
    m = sorted(mediana.values())
    print("== split filtrado: altura mediana das palavras por pagina")
    print("  " + "  ".join(f"p{p}={m[min(len(m) - 1, len(m) * p // 100)]:.0f}"
                           for p in (10, 25, 50, 75, 90, 100)))
    print(f"\n{'corte':>6} | paginas tr/va/te | palavras treino | com acento treino")
    for corte in CORTES:
        npg, cont = collections.Counter(), collections.Counter()
        for pg, med in mediana.items():
            if med < corte:
                continue
            npg[por_pagina[pg][0][0]] += 1
            for s, t, _ in por_pagina[pg]:
                cont[s] += 1
                cont[s + "_ac"] += any(ch in DIAC for ch in t)
        print(f"{corte:>4}px | {npg['train']:>4} /{npg['val']:>4} /{npg['test']:>4} | "
              f"{cont['train']:>15} | {cont['train_ac']:>17}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="bressay.zip oficial (opcional)")
    ap.add_argument("--palavras", default="./bressay/data/words")
    ap.add_argument("--split", default="./bressay_split")
    a = ap.parse_args()
    if a.zip:
        varrer_zip(a.zip)
    cortes(Path(a.palavras), Path(a.split))


if __name__ == "__main__":
    main()
