"""
Presenca dos sinais pedidos numa imagem gerada, juntando as duas medidas que
passaram na anotacao humana do modelo de 40 epocas (anotacao_40ep.md):

  acento em cima  marca solta acima do corpo da palavra, pelo detector do
                  colega (scripts/medir_marcas.py), kappa 0,79
  cedilha         tinta abaixo do c numa imagem so, como em
                  experimentos/cedilha_abaixo.py (medida "tinta"), kappa 0,91

Responde as duas primeiras perguntas da metrica final:
  1. o sinal pedido esta la?            (Detector.analisar)
  2. esta na letra certa, e so nela?    (Detector.letras_marcadas, categoria)
A terceira, se e o sinal certo (til, agudo...), fica para o leitor.

As marcas em cima seguem o detector do colega com duas regras a mais, que a
anotacao humana do modelo de 40 epocas sustentou nas duas metades das palavras
(experimentos/posicao_v2.py):
  alta   aceita marca mais alta que a altura-x quando esta inteira acima do
         corpo; o detector original a descarta como haste
  vogal  a marca vai para a vogal cuja coluna mais se sobrepoe a ela, porque
         acento em portugues so cai em vogal
O scripts/medir_marcas.py e a regua das comparacoes de treino e nao e
alterado; este modulo so usa as constantes dele.

Palavras com i ou j ficam de fora da parte de cima: o pingo e uma marca
solta acima do corpo e o detector o conta como acento.

    from sinais import Detector
    d = Detector()
    d.analisar(cinza, "caçadora")   # cinza: painel 64x256, float 0..255
"""
import json
import os
import sys
import unicodedata

import cv2
import numpy as np

MODULO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(MODULO)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

from acentos_sinteticos import alinhamento, geometria  # noqa: E402
from medir_marcas import AREA_MIN, TOL  # noqa: E402
from medir_posicao_acento import recorte  # noqa: E402

ALINHADOR = os.path.join(RAIZ, "modelos", "alinhador_iam.pt")
# p95 da medida nos c sem cedilha das acentuadas do modelo de 40 epocas,
# fixado antes de olhar as cedilhas (resultados/cedilha_abaixo_40ep.json).
# Na pratica quer dizer "ha tinta abaixo do corte".
with open(os.path.join(MODULO, "resultados", "cedilha_abaixo_40ep.json"), encoding="utf-8") as _f:
    LIMIAR_CEDILHA = json.load(_f)["negativo"]["p95"]

VOGAIS = set("aeiou")
NOME_DO_SINAL = {"́": "agudo", "̀": "grave", "̂": "circunflexo",
                 "̃": "til", "̧": "cedilha"}


def sem_acento(palavra):
    return "".join(c for c in unicodedata.normalize("NFD", palavra) if unicodedata.category(c) != "Mn")


def sinais_pedidos(palavra):
    """[(posicao da letra, nome do sinal)], na palavra sem acento."""
    saida = []
    for k, letra in enumerate(unicodedata.normalize("NFC", palavra)):
        for c in unicodedata.normalize("NFD", letra)[1:]:
            if c in NOME_DO_SINAL:
                saida.append((k, NOME_DO_SINAL[c]))
    return saida


def tem_pingo(palavra):
    return bool(set("ij") & set(sem_acento(palavra)))


def marcas_acima(g):
    """[(x, y, w, h)] das manchas soltas inteiras acima do corpo da palavra, no
    painel. Igual ao scripts/medir_marcas.py, menos o limite de altura (regra
    "alta")."""
    geo = geometria.analisar(g)
    if geo is None:
        return []
    n, _, st, _ = cv2.connectedComponentsWithStats(geo.mask.astype(np.uint8), 8)
    saida = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in st[k])
        if area >= AREA_MIN and y + h <= geo.topo_x + TOL * geo.altura_x:
            saida.append((x, y, w, h))
    return saida


def categoria(palavra, marcadas, cedilha):
    """Resultado da palavra, com marcadas = letras (1..n) com marca em cima e
    cedilha = ha cedilha em algum c (ou None, sem resposta).

    Palavra pedida com acento:
      correto       todos os sinais pedidos, na letra certa, e nada a mais
      a mais        os sinais pedidos estao certos, mas ha marca em outra letra
      letra errada  um acento pedido falta na letra dele, e ha marca em outra
      ausente       falta um sinal pedido, sem marca em outro lugar
    Palavra sem nada pedido: limpa ou marca indevida.
    O tipo do sinal nao e julgado. A cedilha conta so como presenca: o
    gabarito humano da letra dela nao e confiavel."""
    pedidos = sinais_pedidos(palavra)
    em_cima = {k + 1 for k, s in pedidos if s != "cedilha"}
    pede_ced = any(s == "cedilha" for _, s in pedidos)
    if cedilha is None and pede_ced:
        return None
    outras = marcadas - em_cima
    if not pedidos:
        return "marca indevida" if outras or cedilha else "limpa"
    falta_cima = em_cima - marcadas
    falta_ced = pede_ced and not cedilha
    sobra_ced = not pede_ced and bool(cedilha)
    if not falta_cima and not falta_ced:
        return "a mais" if outras or sobra_ced else "correto"
    if falta_cima and outras:
        return "letra errada"
    return "ausente"


class Detector:
    def __init__(self, alinhador=ALINHADOR):
        self.alinhador = alinhamento.Alinhador(alinhador, "cpu")

    def fatias(self, g, letras):
        """(geometria do recorte, colunas de cada letra no recorte, x do recorte
        no painel). None se o recorte nao tem tinta ou nao alinha."""
        r, x0 = recorte(g)
        if r is None:
            return None
        geo = geometria.analisar(r)
        al = self.alinhador.alinhar(r, letras) if geo is not None else None
        if al is None:
            return None
        return geo, geometria.ajustar_aos_vales(geo, alinhamento.fatias_do_alinhamento(al, geo)), x0

    def tinta_abaixo_dos_c(self, g, letras, analise=None):
        """{posicao do c: tinta abaixo do corte / tinta do corpo do c}, como a
        medida "tinta" de experimentos/cedilha_abaixo.py. None se nao alinha."""
        analise = analise or self.fatias(g, letras)
        if analise is None:
            return None
        geo, fatias, _ = analise
        corte = int(round(geo.base + TOL * geo.altura_x)) + 1
        saida = {}
        for k, (xa, xb) in enumerate(fatias):
            if letras[k] != "c":
                continue
            a, b = int(round(xa)), max(int(round(xb)), int(round(xa)) + 1)
            abaixo = geo.mask[corte:, a:b].sum()
            corpo = geo.mask[geo.topo_x:geo.base + 1, a:b].sum()
            saida[k] = float(abaixo) / corpo if corpo else 0.0
        return saida

    def analisar(self, g, palavra):
        """Respostas para uma imagem.

        acima          ha marca solta acima do corpo (None em palavra com i ou j)
        cedilha        algum c tem tinta abaixo do corte (None se nao alinha)
        pede_acima     a palavra pede algum sinal em cima
        pede_cedilha   a palavra pede cedilha
        presente       todos os tipos de sinal pedidos estao la (None se nao
                       da para responder ou se nada foi pedido)
        indevido       ha marca em cima ou cedilha sem nada pedido daquele
                       lado (None se nao da para responder)"""
        letras = sem_acento(palavra)
        pedidos = {nome for _, nome in sinais_pedidos(palavra)}
        pede_ced = "cedilha" in pedidos
        pede_acima = bool(pedidos - {"cedilha"})
        acima = None if tem_pingo(palavra) else bool(marcas_acima(g))
        nos_c = self.tinta_abaixo_dos_c(g, letras) if "c" in letras else {}
        cedilha = None if nos_c is None else any(m > LIMIAR_CEDILHA for m in nos_c.values())
        partes = ([acima] if pede_acima else []) + ([cedilha] if pede_ced else [])
        presente = None if not partes or None in partes else all(partes)
        indevido_partes = ([acima] if not pede_acima else []) + ([cedilha] if not pede_ced and "c" in letras else [])
        indevido = None if None in indevido_partes else any(indevido_partes)
        return {"acima": acima, "cedilha": cedilha, "pede_acima": pede_acima, "pede_cedilha": pede_ced,
                "presente": presente, "indevido": indevido, "tinta_nos_c": nos_c}

    def letras_marcadas(self, g, palavra, analise=None):
        """Letras (1..n) com marca em cima, ou None se nao alinha ou se a
        palavra tem i ou j. Cada marca vai para a letra cuja coluna contem o
        centro dela; se essa letra nao e vogal, para a vogal cuja coluna mais
        se sobrepoe a marca (regra "vogal")."""
        if tem_pingo(palavra):
            return None
        letras = sem_acento(palavra)
        analise = analise or self.fatias(g, letras)
        if analise is None:
            return None
        _, fatias, x0 = analise
        saida = set()
        for x, y, w, h in marcas_acima(g):
            a, b = x - x0, x + w - x0
            cx = (a + b) / 2
            k = next((i for i, (fa, fb) in enumerate(fatias) if fa <= cx < fb),
                     0 if cx < fatias[0][0] else len(fatias) - 1)
            if letras[k] not in VOGAIS:
                sobre = [(min(b, fb) - max(a, fa), i) for i, (fa, fb) in enumerate(fatias)
                         if letras[i] in VOGAIS and min(b, fb) > max(a, fa)]
                if sobre:
                    k = max(sobre)[1]
            saida.add(k + 1)
        return saida

    def avaliar(self, g, palavra):
        """Presenca, letras e categoria de uma imagem."""
        r = self.analisar(g, palavra)
        letras = sem_acento(palavra)
        analise = self.fatias(g, letras)
        r["letras"] = self.letras_marcadas(g, palavra, analise)
        r["categoria"] = None if r["letras"] is None else categoria(palavra, r["letras"], r["cedilha"])
        return r
