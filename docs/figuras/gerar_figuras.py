"""Refaz as figuras de docs/pre_processamento.md.

    python docs/figuras/gerar_figuras.py [caminho/do/recorte.png]

Sem argumento usa o recorte padrao, escolhido por ter fundo escuro (percentil
40 em 180, contra 245 de um recorte tipico), o que torna visivel o efeito da
normalizacao de contraste, e por conter cedilha e til.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAIDA = os.path.join(RAIZ, "docs", "figuras")
PADRAO = os.path.join(RAIZ, "bressay/data/words/7556-012/7556-012-04-02-01.png")
PALAVRA = "atribuições"

P_TINTA, P_FUNDO = 3, 40


def normaliza(bruto):
    g = np.asarray(bruto.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(g, P_TINTA), np.percentile(g, P_FUNDO)
    if hi - lo < 8:
        return bruto.copy(), g, lo, hi
    saida = np.clip((g - lo) / (hi - lo), 0, 1) * 255
    return Image.fromarray(saida.astype(np.uint8)).convert("RGB"), g, lo, hi


def fig_etapas(bruto, norm, w0, h0):
    enq = ImageOps.pad(norm, size=(256, 64), color="white")
    iam = ImageOps.pad(bruto.resize((max(1, int(w0 * 64 / h0)), 64)),
                       size=(256, 64), color="white")
    ESC = 4
    etapas = [
        ("1. recorte bruto", bruto, f"{w0}x{h0} px, como vem do dataset", (0, 0, 0)),
        ("2. contraste por percentis", norm,
         f"percentil {P_TINTA} vira preto, percentil {P_FUNDO} vira branco", (0, 0, 0)),
        ("3. enquadrado em 256x64", enq,
         f"ampliado {64/h0:.2f}x", (170, 0, 0)),
        ("caminho do IAM, para comparar", iam,
         "altura 64 preservando aspecto, depois centraliza", (0, 0, 160)),
    ]
    COL = 400
    alturas = [im.height * ESC for _, im, _, _ in etapas]
    fig = Image.new("RGB", (COL + 256 * ESC + 12, sum(a + 34 for a in alturas) + 12),
                    (255, 255, 255))
    d = ImageDraw.Draw(fig)
    y = 8
    for (titulo, im, nota, cor), alt in zip(etapas, alturas):
        d.text((10, y + alt // 2 - 12), titulo, fill=cor)
        d.text((10, y + alt // 2 + 2), nota, fill=(115, 115, 115))
        e = im.resize((im.width * ESC, alt), Image.NEAREST)
        fig.paste(e, (COL, y))
        d.rectangle([COL - 1, y - 1, COL + e.width, y + alt], outline=(195, 195, 195))
        y += alt + 34
    destino = os.path.join(SAIDA, "pipeline_etapas.png")
    fig.save(destino)
    print("escrito:", destino)


def fig_percentis(g, lo, hi, palavra):
    out = np.clip((g - lo) / (hi - lo), 0, 1) * 255
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1], hspace=.55, wspace=.22)

    for col, (dados, titulo) in enumerate(((g, f'antes, "{palavra}"'), (out, "depois"))):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(dados, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(titulo, fontsize=10)

    ax = fig.add_subplot(gs[1, :])
    ax.hist(g.ravel(), bins=64, range=(0, 255), color="#888", edgecolor="none")
    ax.axvline(lo, color="#c00", lw=2)
    ax.axvline(hi, color="#06c", lw=2)
    ax.axvspan(0, lo, color="#c00", alpha=.13)
    ax.axvspan(hi, 255, color="#06c", alpha=.13)
    topo = ax.get_ylim()[1]
    ax.text(lo - 6, topo * .93, f"lo = {lo:.0f}\n(percentil {P_TINTA})",
            ha="right", va="top", color="#c00", fontsize=9)
    ax.text(hi + 6, topo * .93, f"hi = {hi:.0f}\n(percentil {P_FUNDO})",
            ha="left", va="top", color="#06c", fontsize=9)
    frac = 100 * ((g > lo) & (g < hi)).mean()
    ax.text((lo + hi) / 2, topo * .5, f"estes {frac:.0f}% sao\nesticados p/ 0-255",
            ha="center", fontsize=9)
    ax.set_title("quantos pixels a imagem tem de cada tom de cinza", fontsize=10)
    ax.set_xlabel("tom de cinza (0 = preto, 255 = branco)")
    ax.set_ylabel("nº de pixels")
    ax.set_xlim(0, 255)

    ax = fig.add_subplot(gs[2, :])
    x = np.arange(256)
    ax.plot(x, np.clip((x - lo) / (hi - lo), 0, 1) * 255, color="k", lw=2)
    ax.axvline(lo, color="#c00", lw=1.2, ls="--")
    ax.axvline(hi, color="#06c", lw=1.2, ls="--")
    ax.set_title("a regra, tom que entra contra tom que sai", fontsize=10)
    ax.set_xlabel("entra")
    ax.set_ylabel("sai")
    ax.set_xlim(0, 255)
    ax.set_ylim(-8, 263)
    ax.text(lo / 2, 200, "tudo aqui\nvira preto", ha="center", color="#c00", fontsize=9)
    ax.text((hi + 255) / 2, 60, "tudo aqui vira branco", ha="center",
            color="#06c", fontsize=9)

    destino = os.path.join(SAIDA, "normalizacao_percentis.png")
    fig.savefig(destino, dpi=110, bbox_inches="tight", facecolor="white")
    print("escrito:", destino)


def main():
    caminho = sys.argv[1] if len(sys.argv) > 1 else PADRAO
    palavra = PALAVRA if caminho == PADRAO else os.path.basename(caminho)
    if not os.path.isfile(caminho):
        raise SystemExit(f"recorte nao encontrado: {caminho}")
    bruto = Image.open(caminho).convert("RGB")
    w0, h0 = bruto.size
    norm, g, lo, hi = normaliza(bruto)
    print(f'recorte "{palavra}" {w0}x{h0} | tinta(p{P_TINTA})={lo:.0f} '
          f'fundo(p{P_FUNDO})={hi:.0f} | ampliacao {64/h0:.2f}x')
    fig_etapas(bruto, norm, w0, h0)
    fig_percentis(g, lo, hi, palavra)


if __name__ == "__main__":
    main()
