"""Mascara dos "lugares de acento possivel" de uma palavra, no latente (8 x 32).

Acento em portugues so cai sobre vogal, e a cedilha so sob o c; hastes e pernas
de letras ficam nas consoantes (b d f h k l t / g j p q y). A mascara marca:
  - a faixa ACIMA de cada vogal (de TOL_CORPO a ALCANCE alturas-x acima do topo
    da altura-x), e
  - a faixa ABAIXO de cada c (de TOL_CORPO a ALCANCE alturas-x abaixo da base),
so nas colunas da letra (fatias do alinhador CTC ajustadas aos vales de tinta,
como no gerador de acentos) e so em celulas SEM tinta do alvo. Letras com o
acento pedido (indices em `pular`) ficam de fora: la o acento e o alvo.

No treino (--zona vogais) essas celulas recebem peso_zona: desenhar um acento
que o texto nao pede custa o mesmo que deixar de desenhar o pedido (ACHADOS
15: a zona vazia acima de toda a palavra encurtava as hastes).
"""

import numpy as np

from acentos_sinteticos import alinhamento, geometria

FORMA = (8, 32)          # latente de 64x256
TOL_CORPO = 0.15         # = medir_marcas.TOL / zona vazia
ALCANCE = 1.0            # altura da faixa, em alturas-x
FOLGA_TINTA = 2          # px de dilatacao da tinta: celula com traco por perto fica de fora
FRACAO_CELULA = 0.5      # fracao minima da celula dentro da faixa da letra
MIN_LOGP = -0.2          # = gerador.MIN_LOGP: alinhamento fraco -> sem mascara
VOGAIS = set("aeiouAEIOU")
CEDILHAVEIS = set("cC")
MARGEM = 3


def _recorte(g):
    m = geometria.mascara_tinta(g)
    if m.sum() < 20:
        return None, 0
    ys, xs = np.where(m)
    x0 = max(0, xs.min() - MARGEM)
    return g[max(0, ys.min() - MARGEM):ys.max() + 1 + MARGEM, x0:xs.max() + 1 + MARGEM], x0


def fatias_no_painel(g, texto, alinhador):
    """Painel 64x256 em cinza -> (fatias [(xa, xb)] em px do painel, geometria do painel, motivo)."""
    r, x0 = _recorte(g)
    if r is None:
        return None, None, "sem_tinta"
    geo_r = geometria.analisar(r)
    if geo_r is None:
        return None, None, "sem_tinta"
    al = alinhador.alinhar(r, texto)
    if al is None:
        return None, None, "sem_alinhamento"
    if al.logp_medio < MIN_LOGP:
        return None, None, "confianca"
    fats = geometria.ajustar_aos_vales(geo_r, alinhamento.fatias_do_alinhamento(al, geo_r))
    geo = geometria.analisar(g)
    return [(a + x0, b + x0) for a, b in fats], geo, "ok"


def mascara(g, texto, alinhador, pular=(), excluir_tinta=True):
    """-> (mascara bool (8, 32), motivo). Vazia (zeros) se o alinhamento falhar.
    excluir_tinta=False so na conferencia: as faixas inteiras, para ver se as
    marcas que o modelo desenha caem nelas."""
    import cv2
    vazia = np.zeros(FORMA, dtype=bool)
    fats, geo, motivo = fatias_no_painel(g, texto, alinhador)
    if fats is None or geo is None:
        return vazia, motivo
    H, W = g.shape
    hx = geo.altura_x
    linhas = np.arange(H)
    acima = (linhas < geo.topo_x - TOL_CORPO * hx) & (linhas >= geo.topo_x - ALCANCE * hx)
    abaixo = (linhas > geo.base + TOL_CORPO * hx) & (linhas <= geo.base + ALCANCE * hx)
    faixa = np.zeros((H, W), dtype=bool)
    colunas = np.arange(W)
    for k, (c, (xa, xb)) in enumerate(zip(texto, fats)):
        if k in pular:
            continue
        noletra = (colunas >= xa) & (colunas < xb)
        if c in VOGAIS:
            faixa |= acima[:, None] & noletra[None, :]
        elif c in CEDILHAVEIS:
            faixa |= abaixo[:, None] & noletra[None, :]
    tinta = cv2.dilate(geometria.mascara_tinta(g).astype(np.uint8),
                       np.ones((2 * FOLGA_TINTA + 1,) * 2, np.uint8)).astype(bool)
    h, w = FORMA
    fy, fx = H // h, W // w
    dentro = faixa.reshape(h, fy, w, fx).mean(axis=(1, 3)) >= FRACAO_CELULA
    com_tinta = tinta.reshape(h, fy, w, fx).any(axis=(1, 3))
    return (dentro & ~com_tinta) if excluir_tinta else dentro, "ok"
