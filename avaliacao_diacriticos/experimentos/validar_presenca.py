"""
Etapa 1 da metrica final: a presenca unificada (sinais.py) contra o gabarito
humano do modelo de 40 epocas (resultados/anotacao_40ep_gabarito.csv).

Antes de comparar, confere que a cedilha de sinais.py reproduz, imagem por
imagem, a coluna cedilha_abaixo de resultados/anotacao_40ep_detector.csv,
que e a medida que passou na anotacao.

Grava resultados/presenca_40ep.txt.

    python avaliacao_diacriticos/experimentos/validar_presenca.py
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path.insert(0, MODULO)

from sinais import Detector, tem_pingo  # noqa: E402

RES = os.path.join(MODULO, "resultados")


def ler(nome, **k):
    with open(os.path.join(RES, nome), encoding="utf-8") as f:
        return {r["imagem"]: r for r in csv.DictReader(f, **k)}


def cohen(pares):
    n = len(pares)
    if not n:
        return float("nan")
    obs = sum(a == b for a, b in pares) / n
    pa, pb = sum(a for a, _ in pares) / n, sum(b for _, b in pares) / n
    acaso = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if acaso == 1 else (obs - acaso) / (1 - acaso)


def linha(titulo, pares):
    """pares: [(humano, detector)] booleanos."""
    n = len(pares)
    vp = sum(h and d for h, d in pares); fp = sum(d and not h for h, d in pares)
    fn = sum(h and not d for h, d in pares); vn = n - vp - fp - fn
    prec = f"{vp / (vp + fp):.2f}" if vp + fp else "-"
    rec = f"{vp / (vp + fn):.2f}" if vp + fn else "-"
    return (f"  {titulo:46s} n={n:3d}  concordam {vp + vn:3d} ({(vp + vn) / n:.0%})  "
            f"VP={vp:3d} FP={fp:2d} FN={fn:2d} VN={vn:3d}  precisao {prec}  recall {rec}  kappa {cohen(pares):+.2f}")


def main():
    chave = ler("anotacao_40ep_chave.csv")
    gab = ler("anotacao_40ep_gabarito.csv")
    det_antigo = ler("anotacao_40ep_detector.csv")
    d = Detector()
    res = {}
    for img, item in chave.items():
        g = np.asarray(Image.open(os.path.join(MODULO, item["painel"])).convert("L"), dtype=np.float32)
        res[img] = d.analisar(g, item["palavra"])

    # a cedilha daqui tem de ser a mesma da anotacao
    difere = [i for i in chave if (res[i]["cedilha"] is True) != (det_antigo[i]["cedilha_abaixo"] == "sim")]
    if difere:
        raise SystemExit(f"cedilha nao reproduz a anotacao em {len(difere)} imagens: {difere[:5]}")

    out = ["Presenca unificada contra o gabarito humano, modelo de 40 epocas.",
           "Cedilha reproduz a coluna cedilha_abaixo da anotacao nas 212 imagens.", ""]
    acima_h = lambda i: gab[i]["classe"] != "nenhum"
    ced_h = lambda i: gab[i]["cedilha"] == "sim"
    sem_ij = [i for i in chave if not tem_pingo(chave[i]["palavra"])]

    out.append("PARTES")
    out.append(linha("marca em cima, todas sem i ou j", [(acima_h(i), res[i]["acima"]) for i in sem_ij]))
    out.append(linha("cedilha, todas", [(ced_h(i), res[i]["cedilha"]) for i in chave if res[i]["cedilha"] is not None]))

    out.append("")
    out.append("SINAL PEDIDO PRESENTE (todos os tipos pedidos estao la)")
    grupos = {"so acento em cima": lambda r: r["pede_acima"] and not r["pede_cedilha"],
              "so cedilha": lambda r: r["pede_cedilha"] and not r["pede_acima"],
              "acento em cima e cedilha": lambda r: r["pede_acima"] and r["pede_cedilha"]}
    todos = []
    for nome, f in grupos.items():
        pares = []
        for i in sem_ij:
            r = res[i]
            if not f(r) or r["presente"] is None:
                continue
            h = (acima_h(i) or not r["pede_acima"]) and (ced_h(i) or not r["pede_cedilha"])
            pares.append((h, r["presente"]))
        todos += pares
        out.append(linha(nome, pares))
    out.append(linha("todas as acentuadas", todos))

    out.append("")
    out.append("SINAL ONDE NADA FOI PEDIDO (esqueletos e palavras sem acento)")
    pares = []
    for i in sem_ij:
        r = res[i]
        if r["pede_acima"] or r["pede_cedilha"] or r["indevido"] is None:
            continue
        pares.append((acima_h(i) or ced_h(i), r["indevido"]))
    out.append(linha("marca em cima ou cedilha", pares))
    out.append("")
    out.append("Imagens com i ou j ficam fora: o pingo e uma marca solta em cima.")
    texto = "\n".join(out) + "\n"
    with open(os.path.join(RES, "presenca_40ep.txt"), "w", encoding="utf-8") as f:
        f.write(texto)
    print(texto)


if __name__ == "__main__":
    main()
