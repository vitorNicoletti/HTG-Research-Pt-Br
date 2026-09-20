"""E1 minimo: tudo o que a metrica faz, em uma funcao."""
import unicodedata
import numpy as np
import cv2
from scipy.signal import fftconvolve

ACIMA = set("́̀̂̃")          # agudo, grave, circunflexo, til
ABAIXO = set("̧")            # cedilha


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def diacriticos(palavra):
    """[(indice do caractere, True se a marca fica acima), ...]"""
    saida, i = [], -1
    for c in unicodedata.normalize("NFD", palavra):
        if c in ACIMA or c in ABAIXO:
            saida.append((i, c in ACIMA))
        elif unicodedata.category(c) != "Mn":
            i += 1
    return saida


def tinta(img):
    """Imagem cinza 0..1 -> mascara booleana de tinta, sem a linha pautada."""
    if img.max() - img.min() < 0.10:
        return np.zeros(img.shape, bool)
    _, m = cv2.threshold((img * 255).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = m.astype(bool)
    x = m.any(axis=0)
    if x.any():                                   # pauta: fileira fina e cheia
        cheia = m[:, x.argmax():len(x) - x[::-1].argmax()].mean(axis=1) > 0.9
        grupos = np.diff(np.r_[0, cheia.astype(int), 0])
        for a, b in zip(*[np.where(grupos == v)[0] for v in (1, -1)]):
            if b - a <= 5:
                m[a:b] = False
    return m


def alinha(a, b, mx=20, my=8):
    """Desloca a mascara `a` para casar com `b`, por correlacao cruzada.

    O gerador nem sempre poe os dois gemeos no mesmo lugar: medido em 348
    pares, 60% saem com deslocamento zero e o p90 e de 1 px, mas o maximo
    chega a 18 px -- e sao justamente esses pares que viram falso positivo,
    porque a palavra inteira entra na subtracao. Alinhar antes de subtrair
    derruba o p95 do ruido de 0.181 para 0.103 e leva o AUC de 0.936 para
    0.949. Para o agudo o efeito e maior (p95 de 0.268 para 0.126).

    Alinhar pelo canto da caixa de tinta, que seria mais simples, PIORA
    (AUC 0.906): o canto depende de um pixel solto.
    """
    c = fftconvolve(a.astype(float), b[::-1, ::-1].astype(float), mode="same")
    H, W = a.shape
    jan = c[H // 2 - my:H // 2 + my + 1, W // 2 - mx:W // 2 + mx + 1]
    iy, ix = np.unravel_index(jan.argmax(), jan.shape)
    dy, dx = my - iy, mx - ix
    out = np.zeros_like(a)
    out[max(0, dy):min(H, H + dy), max(0, dx):min(W, W + dx)] = \
        a[max(0, -dy):min(H, H - dy), max(0, -dx):min(W, W - dx)]
    return out


def e1(img_acc, img_asc, palavra):
    """Tinta a mais na acentuada, na faixa e na coluna de cada diacritico.

    Devolve uma fracao do tamanho da letra: 0 = nenhum acento desenhado,
    ~0.17 = acento de tamanho nominal.
    """
    a, b = tinta(img_acc), tinta(img_asc)
    if not b.any() or not a.any():
        return [0.0] * len(diacriticos(palavra))
    a = alinha(a, b)
    perfil = b.sum(axis=1)
    corpo = np.where(perfil >= 0.5 * perfil.max())[0]
    topo, base = corpo[0], corpo[-1] + 1          # altura-x e linha de base
    cols = np.where(b.any(axis=0))[0]
    larg = (cols[-1] + 1 - cols[0]) / len(sem_acento(palavra))

    saida = []
    for i, acima in diacriticos(palavra):
        c = slice(max(0, int(cols[0] + (i - 0.5) * larg)),
                  int(cols[0] + (i + 1.5) * larg))
        f = slice(0, topo) if acima else slice(base, b.shape[0])
        ref = b[topo:base, c].sum()               # tamanho da letra nessa coluna
        saida.append(float(a[f, c].sum() - b[f, c].sum()) / ref if ref else 0.0)
    return saida
