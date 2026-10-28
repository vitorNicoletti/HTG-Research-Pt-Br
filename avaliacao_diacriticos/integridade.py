"""
E2, integridade da palavra, medida em pares, e o cruzamento com o E1.

O planejamento define o E2 como "a palavra continua legivel, descontando o
diacritico, pelo CER de um reconhecedor". Aqui ele e medido em pares: cada
palavra acentuada do scripts/avaliar_pt.py tem o esqueleto gerado com o mesmo
escritor e a mesma semente, e o E2 e

    dCER = CER(acentuada) - CER(esqueleto)

os dois contra a palavra sem acento. O rigor do leitor e a dificuldade da
palavra se cancelam, e sobra o que o acento fez com a palavra.

Antes de ler, as duas imagens passam por sinais.Detector.apagar_sinais, que
tira as manchas soltas em cima e a tinta abaixo dos c. Assim o leitor nao
le o proprio acento como letra. O custo que sobra, medido em pares da base
com as letras identicas, e de 0,008 nos acentos em cima e 0,029 na cedilha
(experimentos/custo_do_acento.py).

Uma palavra conta como degradada quando dCER > MARGEM. A propria variacao da
geracao tambem produz dCER negativo; a fracao de pares com dCER < -MARGEM
serve de controle desse ruido em cada modelo.

Cruzando com o E1 (presenca dos sinais pedidos, de avaliar_sinais.py), cada
palavra cai numa das quatro categorias do planejamento: acerto, omissao,
degradacao com diacritico, degradacao sem diacritico.

    python avaliacao_diacriticos/integridade.py --saida avaliacao_diacriticos/resultados/integridade \\
        --sinais avaliacao_diacriticos/resultados/sinais \\
        --modelo <rotulo>=<pasta das imagens>:<pasta com o paineis.tsv>
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image

MODULO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(MODULO)
sys.path[:0] = [MODULO, RAIZ, os.path.join(RAIZ, "scripts")]

from acentos_sinteticos import alinhamento  # noqa: E402
from avaliar_pt import levenshtein, recorte_tinta  # noqa: E402
from avaliar_sinais import ler_paineis  # noqa: E402
from sinais import Detector, sem_acento, sinais_pedidos  # noqa: E402

LEITOR = os.path.join(RAIZ, "modelos", "leitor_iam_transformer.pt")
MARGEM = 0.2
CATEGORIAS = ["acerto", "omissao", "degradacao com diacritico", "degradacao sem diacritico"]
CLASSES = ["agudo", "til", "circunflexo", "cedilha", "cedilha e til"]


def classe(palavra):
    return " e ".join(sorted({s for _, s in sinais_pedidos(palavra)}, key=lambda s: s != "cedilha"))


def categoria(presente, degradou):
    if presente:
        return "degradacao com diacritico" if degradou else "acerto"
    return "degradacao sem diacritico" if degradou else "omissao"


def resumir(rotulo, linhas):
    d = np.array([l["dcer"] for l in linhas])
    por_palavra = {}
    for l in linhas:
        por_palavra.setdefault(l["palavra"], []).append(l["dcer"])
    ws = sorted(por_palavra)
    rng = np.random.default_rng(0)
    boot = [np.mean([x for w in rng.choice(ws, len(ws)) for x in por_palavra[w]]) for _ in range(2000)]
    r = {"modelo": rotulo, "pares": len(linhas), "dcer_medio": round(float(d.mean()), 3),
         "ic95": f"{np.percentile(boot, 2.5):+.3f} a {np.percentile(boot, 97.5):+.3f}",
         "piora": round(100 * float((d > MARGEM).mean()), 1), "melhora": round(100 * float((d < -MARGEM).mean()), 1)}
    com_e1 = [l for l in linhas if l["presente"] is not None]
    for c in CATEGORIAS:
        r[c] = round(100 * sum(l["categoria"] == c for l in com_e1) / len(com_e1), 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="rotulo=pasta_das_imagens:pasta_do_paineis.tsv")
    ap.add_argument("--sinais", required=True, help="pasta com os sinais_<rotulo>.tsv de avaliar_sinais.py")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    resumo, por_classe = os.path.join(a.saida, "resumo_integridade.tsv"), os.path.join(a.saida, "por_classe.tsv")
    feitos = set()
    if os.path.isfile(resumo):
        with open(resumo, encoding="utf-8") as f:
            feitos = {r["modelo"] for r in csv.DictReader(f, delimiter="\t")}
    leitor = alinhamento.Alinhador(LEITOR, "cpu")
    d = Detector()
    for m in a.modelo:
        rotulo, resto = m.split("=", 1)
        pasta, tsv = resto.rsplit(":", 1)
        if rotulo in feitos:
            print(f"{rotulo}: ja no resumo")
            continue
        t0 = time.time()
        rs, k = ler_paineis(os.path.join(tsv, "paineis.tsv"), rotulo)
        with open(os.path.join(a.sinais, f"sinais_{rotulo}.tsv"), encoding="utf-8") as f:
            e1 = {(r["palavra"], r["tipo"], r["escritor"], r["semente"]): r for r in csv.DictReader(f, delimiter="\t")}

        def ler(palavra, tipo, escritor, semente):
            img = os.path.join(pasta, f"{k[(palavra, tipo)]:03d}_{escritor}_{semente}.png")
            g = np.asarray(Image.open(img).convert("L"), dtype=np.float32)
            g = d.apagar_sinais(g, palavra)
            r = recorte_tinta(g)
            lido = leitor.ler(r) if r is not None else ""
            alvo = sem_acento(palavra)
            return levenshtein(lido, alvo) / max(1, len(alvo))

        linhas = []
        for r in rs:
            if r["tipo"] != "acentuada":
                continue
            esq = sem_acento(r["palavra"])
            ca = ler(r["palavra"], "acentuada", r["escritor"], r["semente"])
            ce = ler(esq, "esqueleto", r["escritor"], r["semente"])
            s = e1[(r["palavra"], "acentuada", r["escritor"], r["semente"])]
            pede_c, pede_ced = s["pede_acima"] == "True", s["pede_cedilha"] == "True"
            partes = ([s["acima"]] if pede_c else []) + ([s["cedilha"]] if pede_ced else [])
            presente = None if "" in partes else all(p == "1" for p in partes)
            dcer = ca - ce
            linhas.append({"palavra": r["palavra"], "classe": classe(r["palavra"]), "escritor": r["escritor"],
                           "semente": r["semente"], "cer_acentuada": round(ca, 3), "cer_esqueleto": round(ce, 3),
                           "dcer": round(dcer, 3), "presente": presente,
                           "categoria": "" if presente is None else categoria(presente, dcer > MARGEM)})
        with open(os.path.join(a.saida, f"integridade_{rotulo}.tsv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter="\t")
            w.writeheader(); w.writerows(linhas)
        for arq, itens in ((resumo, [resumir(rotulo, linhas)]),
                           (por_classe, [dict(resumir(rotulo, [l for l in linhas if l["classe"] == c]), classe=c)
                                         for c in CLASSES if any(l["classe"] == c for l in linhas)])):
            novo = not os.path.isfile(arq)
            with open(arq, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(itens[0]), delimiter="\t")
                if novo:
                    w.writeheader()
                w.writerows(itens)
        print(f"{rotulo}: {time.time() - t0:.0f} s  " + "  ".join(f"{c} {v}" for c, v in resumir(rotulo, linhas).items() if c != "modelo"), flush=True)


if __name__ == "__main__":
    main()
