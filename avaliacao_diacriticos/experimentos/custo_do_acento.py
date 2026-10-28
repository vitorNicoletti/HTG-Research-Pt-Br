"""
Quanto o leitor erra so porque ha um acento desenhado, com as letras iguais.

Na base iam_pt_alinhado cada acentuada tem o par: a mesma imagem sem o sinal.
As letras sao as mesmas pixel a pixel, entao a diferenca de CER entre a
acentuada e o par e o custo do acento para o leitor, sem degradacao nenhuma da
letra. E o piso do E2 pareado (CER da acentuada menos CER do esqueleto).

Mede tambem duas variantes, aplicadas nas duas imagens antes de ler:
  marcas apagadas    apaga as manchas soltas em cima (sinais.marcas_acima)
  e abaixo dos c     tambem apaga a tinta abaixo do corte nas colunas de cada c,
                     a mesma regiao em que a cedilha e medida (sinais.Detector)
Se o custo cair a perto de zero, o E2 pode ser medido assim e passa a ver so a
letra.

Mesmo leitor e mesmo recorte do scripts/avaliar_pt.py. Palavras de 3 a 10
letras, sem i e j (o pingo seria apagado junto).

Grava resultados/custo_do_acento.txt.

    python avaliacao_diacriticos/experimentos/custo_do_acento.py [--n 800]
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
RAIZ = os.path.dirname(MODULO)
sys.path[:0] = [MODULO, RAIZ, os.path.join(RAIZ, "scripts"), os.path.join(RAIZ, "DiffusionPen")]

from acentos_sinteticos import alinhamento  # noqa: E402
from avaliar_pt import levenshtein, recorte_tinta  # noqa: E402
from sinais import Detector, marcas_acima, sem_acento, sinais_pedidos  # noqa: E402
from utils.iam_acentuado_dataset import preprocessar_iam  # noqa: E402

BASE = os.path.join(RAIZ, "iam_pt_alinhado")
LEITOR = os.path.join(RAIZ, "modelos", "leitor_iam_transformer.pt")
PALAVRA = re.compile(r"^[a-hk-z]{3,10}$")


def apagar_marcas(g):
    g = g.copy()
    for x, y, w, h in marcas_acima(g):
        g[y:y + h, x:x + w] = 255.0
    return g


def cer(leitor, g, alvo):
    r = recorte_tinta(g)
    lido = leitor.ler(r) if r is not None else ""
    return levenshtein(lido, alvo) / max(1, len(alvo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    a = ap.parse_args()
    leitor = alinhamento.Alinhador(LEITOR, "cpu")
    detector = Detector()
    man = [json.loads(l) for l in open(os.path.join(BASE, "manifesto.jsonl"), encoding="utf-8")]
    pares = {}
    for x in man:
        if x["tipo"] in ("acentuada", "par"):
            pares.setdefault(x["tarefa"], {})[x["tipo"]] = x
    itens = [d for d in pares.values() if len(d) == 2 and PALAVRA.match(sem_acento(d["acentuada"]["rotulo"]))]
    itens = random.Random(0).sample(sorted(itens, key=lambda d: d["acentuada"]["arquivo"]), min(a.n, len(itens)))

    grupos = {"so acento em cima": [], "com cedilha": []}
    for d in itens:
        alvo = sem_acento(d["acentuada"]["rotulo"])
        g = {}
        for tipo in ("acentuada", "par"):
            img = Image.open(os.path.join(BASE, d[tipo]["arquivo"])).convert("RGB")
            g[tipo] = np.asarray(preprocessar_iam(img, d[tipo]["rotulo"]).convert("L"), dtype=np.float32)
        linha = {}
        variantes = (("lido como esta", lambda x: x), ("marcas apagadas", apagar_marcas),
                     ("e abaixo dos c", lambda x: detector.apagar_sinais(x, alvo)))
        for nome, f in variantes:
            linha[nome] = cer(leitor, f(g["acentuada"]), alvo) - cer(leitor, f(g["par"]), alvo)
        tem_ced = any(s == "cedilha" for _, s in sinais_pedidos(d["acentuada"]["rotulo"]))
        grupos["com cedilha" if tem_ced else "so acento em cima"].append(linha)

    out = ["Custo do acento para o leitor: CER(acentuada) - CER(par) na base iam_pt_alinhado,",
           "mesmas letras pixel a pixel. Leitor modelos/leitor_iam_transformer.pt, recorte do avaliar_pt.py.", ""]
    for grupo, linhas in grupos.items():
        out.append(f"{grupo}: {len(linhas)} pares")
        for nome in ("lido como esta", "marcas apagadas", "e abaixo dos c"):
            d = np.array([l[nome] for l in linhas])
            boot = [d[np.random.default_rng(k).integers(0, len(d), len(d))].mean() for k in range(2000)]
            out.append(f"  {nome:16s} media {d.mean():+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
                       f"  iguais {100 * (d == 0).mean():.0f}%  piora > 0,2 {100 * (d > 0.2).mean():.0f}%"
                       f"  melhora > 0,2 {100 * (d < -0.2).mean():.0f}%")
        out.append("")
    texto = "\n".join(out)
    open(os.path.join(MODULO, "resultados", "custo_do_acento.txt"), "w", encoding="utf-8").write(texto + "\n")
    print(texto)


if __name__ == "__main__":
    main()
