"""Geometria da tinta de uma palavra: onde o sinal deve encostar.

Metodo:
  1. mascara de tinta por Otsu e caixa da tinta (ignorando manchas);
  2. corpo da palavra pelo perfil horizontal: a faixa continua de fileiras
     em torno da mais cheia com ao menos 30% da tinta dela, com piso de 3
     espessuras de traco. O topo do
     corpo e a altura-x, o fundo e a linha de base -- ascendentes (l, t, d) e
     descendentes (g, p) ficam de fora;
  3. a largura da caixa e dividida em N fatias iguais, uma por letra;
  4. na fatia da letra escolhida, o ponto de contato SUPERIOR (para acentos)
     e a primeira tinta de cima para baixo nas colunas centrais da fatia, sem
     subir acima da altura-x (para nao pegar a ascendente da letra vizinha que
     invade a fatia); o INFERIOR (para a cedilha) e a ultima tinta de cima para
     baixo, sem descer abaixo da linha de base. O x do contato e o centroide
     da tinta do corpo dentro da fatia, que corrige um pouco a desigualdade de
     largura entre letras.
"""

from dataclasses import dataclass

import cv2
import numpy as np

AREA_MIN_MANCHA = 4        # px; componentes menores nao contam para a caixa
LIMIAR_CORPO = 0.3         # fracao da fileira mais cheia que define o corpo
MIN_ALTURA_X_ESPESSURAS = 3.0  # altura-x minima, em espessuras de traco
FRACAO_CENTRAL = 0.6       # fracao central da fatia usada no ponto de contato
TOLERANCIA_CORPO = 0.15    # quanto (em alturas-x) o contato pode sair do corpo


@dataclass
class Geometria:
    mask: np.ndarray       # bool, True = tinta
    caixa: tuple           # (x0, x1, y0, y1) da tinta
    topo_x: int            # fileira da altura-x (topo do corpo)
    base: int              # fileira da linha de base (fundo do corpo)
    espessura: float       # largura tipica do traco, em px
    tom: float             # cinza mediano da tinta, 0..255

    @property
    def altura_x(self):
        return max(1, self.base - self.topo_x)


@dataclass
class Contato:
    x: float               # coluna onde centralizar o sinal
    y: float               # fileira onde o sinal encosta na letra
    fatia: tuple           # (xa, xb) da fatia da letra


def mascara_tinta(g):
    """Cinza 0..255 -> mascara booleana de tinta (Otsu); vazia sem contraste."""
    if float(g.max()) - float(g.min()) < 25:
        return np.zeros(g.shape, dtype=bool)
    _, m = cv2.threshold(g.astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m.astype(bool)


def _sem_manchas(mask):
    n, rot, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    bons = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= AREA_MIN_MANCHA]
    return np.isin(rot, bons)


def analisar(g):
    """Geometria de uma palavra em cinza 0..255; None se nao houver tinta."""
    mask = mascara_tinta(g)
    limpa = _sem_manchas(mask)
    if not limpa.any():
        return None
    ys, xs = np.where(limpa)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1

    perfil = limpa[:, x0:x1].sum(axis=1).astype(np.float32)
    perfil = np.convolve(perfil, np.ones(3) / 3, mode="same")
    # Faixa CONTINUA em torno da fileira mais cheia, crescendo enquanto as
    # fileiras tem ao menos LIMIAR_CORPO da tinta dela. Com limiar 0,5 sobre
    # todas as fileiras o corpo encolhia quando um traco horizontal forte
    # dominava o perfil ("area": 8 px de altura-x para traco de 5,7 px); a
    # faixa da massa de tinta exagerava quando lacos de ascendentes
    # concentram a tinta ("life": 61 px). A faixa continua com limiar mais
    # baixo para nos lacos finos das hastes.
    pico = int(np.argmax(perfil))
    lim = LIMIAR_CORPO * perfil[pico]
    topo_x = pico
    while topo_x > 0 and perfil[topo_x - 1] >= lim:
        topo_x -= 1
    base = pico
    while base < len(perfil) - 1 and perfil[base + 1] >= lim:
        base += 1

    dist = cv2.distanceTransform(limpa.astype(np.uint8), cv2.DIST_L2, 3)
    espessura = max(1.0, 2.0 * float(np.percentile(dist[limpa], 90)))

    # piso: a altura-x de uma letra e sempre varias vezes a espessura do traco
    falta = MIN_ALTURA_X_ESPESSURAS * espessura - (base - topo_x)
    if falta > 0:
        topo_x = max(y0, int(topo_x - falta / 2))
        base = min(y1 - 1, int(base + falta / 2 + 0.5))
    tom = float(np.median(g[limpa]))
    return Geometria(limpa, (x0, x1, y0, y1), topo_x, base, espessura, tom)


def fatias(geo, n_letras):
    """N fatias iguais da caixa de tinta, uma por letra: [(xa, xb), ...]."""
    x0, x1 = geo.caixa[0], geo.caixa[1]
    passo = (x1 - x0) / max(1, n_letras)
    return [(x0 + k * passo, x0 + (k + 1) * passo) for k in range(n_letras)]


def _colunas_centrais(fatia, largura_img):
    xa, xb = fatia
    meio, meia = (xa + xb) / 2, (xb - xa) * FRACAO_CENTRAL / 2
    c0 = int(np.clip(np.floor(meio - meia), 0, largura_img - 1))
    c1 = int(np.clip(np.ceil(meio + meia), c0 + 1, largura_img))
    return c0, c1


def _centroide_corpo(geo, fatia):
    xa, xb = int(np.floor(fatia[0])), int(np.ceil(fatia[1]))
    bloco = geo.mask[geo.topo_x:geo.base + 1, xa:xb]
    cols = np.where(bloco.any(axis=0))[0]
    if len(cols) == 0:
        return (fatia[0] + fatia[1]) / 2
    pesos = bloco.sum(axis=0)[cols]
    return xa + float(np.average(cols, weights=pesos))


def contato_superior(geo, fatia):
    """Onde o acento encosta: topo da tinta da letra, sem subir acima da altura-x."""
    tol = TOLERANCIA_CORPO * geo.altura_x
    lim = int(max(0, geo.topo_x - tol))
    c0, c1 = _colunas_centrais(fatia, geo.mask.shape[1])
    bloco = geo.mask[lim:geo.base + 1, c0:c1]
    linhas = np.where(bloco.any(axis=1))[0]
    y = lim + int(linhas.min()) if len(linhas) else geo.topo_x
    return Contato(_centroide_corpo(geo, fatia), float(y), fatia)


def contato_inferior(geo, fatia):
    """Onde a cedilha encosta: fundo da tinta da letra, sem descer abaixo da base."""
    tol = TOLERANCIA_CORPO * geo.altura_x
    lim = int(min(geo.mask.shape[0], geo.base + tol + 1))
    c0, c1 = _colunas_centrais(fatia, geo.mask.shape[1])
    bloco = geo.mask[geo.topo_x:lim, c0:c1]
    linhas = np.where(bloco.any(axis=1))[0]
    y = geo.topo_x + int(linhas.max()) if len(linhas) else geo.base
    return Contato(_centroide_corpo(geo, fatia), float(y), fatia)


def pingos_do_i(geo, fatia):
    """Mascara dos componentes pequenos inteiramente acima da altura-x dentro
    da fatia -- o pingo do i, que sai antes de desenhar o agudo do i."""
    n, rot, st, cent = cv2.connectedComponentsWithStats(geo.mask.astype(np.uint8), 8)
    xa, xb = fatia
    area_max = 0.35 * geo.altura_x ** 2
    sai = np.zeros(geo.mask.shape, dtype=bool)
    for k in range(1, n):
        y_fundo = st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]
        if (y_fundo <= geo.topo_x and st[k, cv2.CC_STAT_AREA] <= area_max
                and xa - 2 <= cent[k][0] <= xb + 2):
            sai |= rot == k
    return sai
