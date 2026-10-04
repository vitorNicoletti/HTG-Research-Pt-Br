"""Conta marcas soltas (acento/cedilha) em palavras geradas, painel a painel.

Para cada painel gerado por scripts/gerar_amostras.py --paineis:
  1. corpo da palavra (altura-x e linha de base) pela geometria do gerador de
     acentos (acentos_sinteticos.geometria.analisar);
  2. componentes de tinta SOLTOS: nao tocam o corpo -- terminam acima de
     topo_x + TOL (marca acima: acento) ou comecam abaixo de base - TOL (marca
     abaixo: cedilha) -- e sao pequenos demais para ser letra (altura menor
     que a altura-x). Hastes de t/h/d/l saem do corpo e nao contam.

Use palavras sem i/j: o pingo seria uma marca acima legitima. A taxa no
modelo original do IAM (que nao desenha acento) e o ruido do detector.

    python scripts/medir_marcas.py --modelo iam=pasta1 --modelo teto=pasta2 \\
        --saida saidas/medicao_marcas
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import geometria  # noqa: E402

TOL = 0.15             # em alturas-x: quanto a marca pode invadir o corpo
AREA_MIN = 12          # px (painel 64x256); menos que isso e ruido
ALTURA_MAX = 1.0       # em alturas-x; mais alto que isso e letra/haste


def marcas(g):
    """(acima, abaixo, caixas) de um painel em cinza 0..255."""
    geo = geometria.analisar(g)
    if geo is None:
        return 0, 0, [], None
    hx = geo.altura_x
    n, _, st, _ = cv2.connectedComponentsWithStats(geo.mask.astype(np.uint8), 8)
    acima = abaixo = 0
    caixas = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if area < AREA_MIN or h >= ALTURA_MAX * hx:
            continue
        if y + h <= geo.topo_x + TOL * hx:
            acima += 1
            caixas.append((x, y, w, h, "acima"))
        elif y >= geo.base - TOL * hx:
            abaixo += 1
            caixas.append((x, y, w, h, "abaixo"))
    return acima, abaixo, caixas, geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True,
                    help="rotulo=pasta (pasta de gerar_amostras.py --paineis)")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--folha", type=int, default=48, help="paineis na folha de conferencia")
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    linhas, por = [], defaultdict(lambda: [0, 0, 0, 0])   # n, acima, abaixo, alguma
    conferir = []
    modelos = [m.split("=", 1) for m in a.modelo]
    for rot, pasta in modelos:
        for palavra in sorted(os.listdir(pasta)):
            dp = os.path.join(pasta, palavra)
            if not os.path.isdir(dp):
                continue
            for arq in sorted(os.listdir(dp)):
                caminho = os.path.join(dp, arq)
                g = np.asarray(Image.open(caminho).convert("L"), dtype=np.float32)
                ac, ab, caixas, geo = marcas(g)
                c = por[(rot, palavra)]
                c[0] += 1
                c[1] += ac > 0
                c[2] += ab > 0
                c[3] += (ac + ab) > 0
                linhas.append(f"{rot}\t{palavra}\t{arq}\t{ac}\t{ab}")
                conferir.append((rot, palavra, caminho, caixas, geo))

    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("modelo\tpalavra\tpainel\tmarcas_acima\tmarcas_abaixo\n" + "\n".join(linhas) + "\n")

    palavras = sorted({p for _, p in por}, key=lambda p: (p.isascii(), p))
    rots = [r for r, _ in modelos]
    tab = ["| palavra | " + " | ".join(rots) + " |", "|---|" + "---|" * len(rots)]
    resumo = {}
    for p in palavras:
        cel = []
        for r in rots:
            n, ac, ab, alg = por[(r, p)]
            cel.append(f"{alg}/{n} ({100 * alg / max(1, n):.0f}%)")
            resumo[f"{r}/{p}"] = {"n": n, "acima": ac, "abaixo": ab, "alguma": alg}
        tab.append(f"| {p} | " + " | ".join(cel) + " |")
    texto = "\n".join(tab)
    print(texto)
    open(os.path.join(a.saida, "resumo.md"), "w", encoding="utf-8").write(
        "Paineis com ao menos uma marca solta (acima ou abaixo do corpo).\n\n" + texto + "\n")
    json.dump(resumo, open(os.path.join(a.saida, "resumo.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # folha de conferencia: corpo (azul) e marcas detectadas (vermelho/verde)
    amostra = random.Random(0).sample(conferir, min(a.folha, len(conferir)))
    E = 2
    folha = Image.new("RGB", (6 * (256 * E + 8), (len(amostra) + 5) // 6 * (64 * E + 22)), "white")
    d = ImageDraw.Draw(folha)
    for i, (rot, palavra, caminho, caixas, geo) in enumerate(amostra):
        im = Image.open(caminho).convert("RGB")
        dd = ImageDraw.Draw(im)
        if geo is not None:
            for y in (geo.topo_x, geo.base):
                dd.line([(0, y), (255, y)], fill=(0, 90, 255))
        for x, y, w, h, tipo in caixas:
            dd.rectangle([x - 1, y - 1, x + w, y + h], outline=(230, 0, 0) if tipo == "acima" else (0, 160, 0))
        im = im.resize((256 * E, 64 * E), Image.NEAREST)
        x0, y0 = (i % 6) * (256 * E + 8), (i // 6) * (64 * E + 22)
        d.text((x0, y0), f"{rot} | {palavra} | {len(caixas)}", fill="black")
        folha.paste(im, (x0, y0 + 14))
    folha.save(os.path.join(a.saida, "conferencia.png"))


if __name__ == "__main__":
    main()
