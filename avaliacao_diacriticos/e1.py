"""E1 minimo: tudo o que a metrica faz, em uma funcao."""
import unicodedata
import numpy as np
import cv2

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


def e1(img_acc, img_asc, palavra):
    """Tinta a mais na acentuada, na faixa e na coluna de cada diacritico.

    Devolve uma fracao do tamanho da letra: 0 = nenhum acento desenhado,
    ~0.17 = acento de tamanho nominal.
    """
    a, b = tinta(img_acc), tinta(img_asc)
    if not b.any():
        return [0.0] * len(diacriticos(palavra))
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
