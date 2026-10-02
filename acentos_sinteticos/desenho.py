"""Rasteriza um traco sobre a imagem, com a caneta da propria palavra.

O traco e desenhado em supersampling (SS vezes maior) e reduzido com
filtragem, o que da borda suave como a de uma digitalizacao. A espessura
afina nas pontas (como a pressao da caneta) e a tinta e misturada pelo
minimo: onde ja ha tinta mais escura, ela prevalece.
"""

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

SS = 4   # fator de supersampling


def desenhar(g, pontos, espessura, tom, afinamento=0.45):
    """Desenha a polilinha `pontos` (coordenadas da imagem) em `g` (cinza 0..255).

    espessura  -- largura no meio do traco, em px
    tom        -- cinza da tinta, 0..255
    afinamento -- 0 = largura constante; 1 = pontas com largura zero
    Devolve uma copia de g.
    """
    h, w = g.shape
    camada = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(camada)
    n = len(pontos)
    for i in range(n - 1):
        t = (i + 0.5) / (n - 1)
        larg = espessura * (1 - afinamento + afinamento * math.sin(math.pi * t))
        r = max(0.5, larg * SS / 2)
        (xa, ya), (xb, yb) = pontos[i], pontos[i + 1]
        d.line([(xa * SS, ya * SS), (xb * SS, yb * SS)], fill=255, width=max(1, int(round(2 * r))))
        d.ellipse([xb * SS - r, yb * SS - r, xb * SS + r, yb * SS + r], fill=255)
    alfa = np.asarray(camada.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    tinta = 255.0 - alfa * (255.0 - tom)
    return np.minimum(g, tinta)


def cor_do_papel(g):
    """Cinza tipico do fundo: mediana dos 40% de pixels mais claros."""
    return float(np.median(g[g >= np.percentile(g, 60)]))


def apagar(g, mascara):
    """Pinta `mascara` com a cor do papel, para tirar um pingo do i."""
    saida = g.copy()
    # dilata 1 px: a borda suavizada do pingo nao fica como sombra
    m = cv2.dilate(mascara.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    saida[m] = cor_do_papel(g)
    return saida


def ampliar_tela(g, cima, baixo, esq, dir_):
    """Acrescenta margens com a cor do papel (o acento pode passar da borda)."""
    if not (cima or baixo or esq or dir_):
        return g
    return np.pad(g, ((cima, baixo), (esq, dir_)), constant_values=cor_do_papel(g))
