"""
Etapas 1 e 2 da metrica final: a categoria de cada palavra (sinais.categoria)
contra a mesma categoria calculada do gabarito humano letra por letra, no
modelo de 40 epocas.

O gabarito por letra e o de diagnosticar_posicao.gabarito_por_letra; a cedilha
do gabarito e a resposta sim ou nao de anotacao_40ep_gabarito.csv. As duas
regras do detector (vogal e alta) nao tem parametro e nao foram ajustadas
nestas imagens; a decisao de adota-las veio de experimentos/posicao_v2.py.

Grava resultados/categorias_40ep.txt.

    python avaliacao_diacriticos/experimentos/validar_categorias.py
"""
import collections
import csv
import os
import sys

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path[:0] = [MODULO, AQUI]

from diagnosticar_posicao import gabarito_por_letra  # noqa: E402
from sinais import Detector, categoria, sinais_pedidos, tem_pingo  # noqa: E402

RES = os.path.join(MODULO, "resultados")
ORDEM = ["correto", "a mais", "letra errada", "ausente", "limpa", "marca indevida"]


def cohen(a, b):
    n = len(a)
    obs = sum(x == y for x, y in zip(a, b)) / n
    acaso = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return float("nan") if acaso == 1 else (obs - acaso) / (1 - acaso)


def main():
    chave = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_chave.csv"), encoding="utf-8"))}
    gab = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_gabarito.csv"), encoding="utf-8"))}
    letras_gab = gabarito_por_letra()
    d = Detector()
    pares = []
    for img, item in chave.items():
        palavra = item["palavra"]
        if tem_pingo(palavra):
            continue
        g = np.asarray(Image.open(os.path.join(MODULO, item["painel"])).convert("L"), dtype=np.float32)
        r = d.avaliar(g, palavra)
        h = categoria(palavra, letras_gab[img], gab[img]["cedilha"] == "sim")
        if r["categoria"] is None or h is None:
            continue
        pares.append((palavra, h, r["categoria"]))

    out = ["Categoria por palavra: detector (sinais.py) contra o gabarito humano, modelo de 40 epocas.", ""]
    for titulo, filtro in (("PALAVRAS PEDIDAS COM ACENTO", lambda p: bool(sinais_pedidos(p))),
                           ("PALAVRAS SEM NADA PEDIDO", lambda p: not sinais_pedidos(p))):
        sub = [(h, dt) for p, h, dt in pares if filtro(p)]
        hs, ds = [h for h, _ in sub], [dt for _, dt in sub]
        out.append(f"{titulo}: {len(sub)} imagens, concordam {sum(a == b for a, b in sub)} "
                   f"({sum(a == b for a, b in sub) / len(sub):.0%}), kappa {cohen(hs, ds):+.2f}")
        cats = [c for c in ORDEM if c in set(hs) | set(ds)]
        out.append("  pessoas (linhas) x detector (colunas)")
        out.append("  " + " " * 15 + "".join(f"{c:>15s}" for c in cats) + f"{'total':>8s}")
        m = collections.Counter(sub)
        for c in cats:
            out.append(f"  {c:15s}" + "".join(f"{m[(c, e)]:>15d}" for e in cats) + f"{hs.count(c):>8d}")
        out.append(f"  {'total':15s}" + "".join(f"{ds.count(e):>15d}" for e in cats))
        out.append("")
    out.append("Taxas no modelo de 40 epocas, palavras pedidas com acento:")
    sub = [(h, dt) for p, h, dt in pares if sinais_pedidos(p)]
    for c in ORDEM[:4]:
        out.append(f"  {c:13s} pessoas {sum(h == c for h, _ in sub) / len(sub):.0%}   detector {sum(dt == c for _, dt in sub) / len(sub):.0%}")
    out.append("")
    out.append("Por classe de sinal, palavras pedidas com acento (correto pelas pessoas / pela metrica, e concordancia):")
    classe = lambda p: " e ".join(sorted({s for _, s in sinais_pedidos(p)}, key=lambda s: s != "cedilha"))
    for c in ("agudo", "til", "circunflexo", "cedilha", "cedilha e til"):
        sub = [(h, dt) for p, h, dt in pares if sinais_pedidos(p) and classe(p) == c]
        if not sub:
            continue
        n = len(sub)
        out.append(f"  {c:14s} n={n:3d}  correto pessoas {sum(h == 'correto' for h, _ in sub) / n:.0%}"
                   f"  metrica {sum(d == 'correto' for _, d in sub) / n:.0%}"
                   f"  mesma categoria {sum(h == d for h, d in sub)} de {n}")
    texto = "\n".join(out) + "\n"
    open(os.path.join(RES, "categorias_40ep.txt"), "w", encoding="utf-8").write(texto)
    print(texto)


if __name__ == "__main__":
    main()
