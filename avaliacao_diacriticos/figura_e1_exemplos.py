"""Duas figuras de exemplo do E1, a partir do controle IAM puro.

  figura_e1_exemplos.png  uma marca por linha: entrada, gemea, diferenca real,
                          e a mesma diferenca com um acento nominal pintado.
  figura_e1_falhas.png    os negativos que mais pontuaram. Sao falsos
                          positivos por construcao: o IAM nao desenha acento.
"""
import json
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "avaliacao_diacriticos")
import metrica as M, e1
from teste_sensibilidade_diff import pinta_acento

D = "avaliacao_diacriticos/ger_iam_n696"
itens = [json.loads(l) for l in open(f"{D}/manifest.jsonl", encoding="utf-8")]
gem = {(x["par_id"], x["escritor"], x["semente"]): x for x in itens if not x["acentuada"]}


def geometria(b, palavra):
    """(topo da altura-x, linha de base, primeira coluna, largura por caractere)"""
    perfil = b.sum(axis=1)
    corpo = np.where(perfil >= 0.5 * perfil.max())[0]
    cols = np.where(b.any(axis=0))[0]
    return (corpo[0], corpo[-1] + 1, cols[0],
            (cols[-1] + 1 - cols[0]) / len(M.sem_acento(palavra)))


def caixa(ax, b, palavra, idx, acima):
    topo, base, x0, larg = geometria(b, palavra)
    c0 = max(0, int(x0 + (idx - 0.5) * larg))
    c1 = int(x0 + (idx + 1.5) * larg)
    y0, alt = (-0.5, topo + 0.5) if acima else (base, b.shape[0] - base)
    ax.add_patch(Rectangle((c0, y0), c1 - c0, alt, ec="tab:green", fc="none", lw=1.6))


def painel_dif(ax, mask_acc, b):
    # alinhado, igual ao que a metrica faz
    d = e1.alinha(mask_acc, b).astype(int) - b.astype(int)
    img = np.ones((*b.shape, 3))
    img[b] = 0.80
    img[d > 0] = (0.85, 0.10, 0.10)
    img[d < 0] = (0.10, 0.30, 0.85)
    ax.imshow(img, interpolation="nearest")


def limpa(ax, titulo=None):
    ax.set_xticks([]); ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=8.5, loc="left", pad=3)


def casos():
    """Um par por marca, preferindo as imagens com mais contraste."""
    melhor = {}
    for r in itens:
        if not r["acentuada"] or r.get("colapsada"):
            continue
        g = gem.get((r["par_id"], r["escritor"], r["semente"]))
        if not g:
            continue
        for k, (idx, marca, onde) in enumerate(M.diacriticos(r["palavra"])):
            if marca not in melhor or r["std"] > melhor[marca][0]["std"]:
                melhor[marca] = (r, g, idx, onde, k)
    return [melhor[m] for m in ["til", "cedilha", "agudo", "circunflexo", "grave"]
            if m in melhor]


sel = casos()
fig, ax = plt.subplots(len(sel), 4, figsize=(15, 1.65 * len(sel)))
fig.suptitle("E1 em um par mínimo de cada marca — modelo do IAM puro, que nunca viu português",
             fontsize=13, y=0.995)
for i, (r, g, idx, onde, k) in enumerate(sel):
    acc = M.carregar_tinta(f"{D}/{r['arquivo']}")
    asc = M.carregar_tinta(f"{D}/{g['arquivo']}")
    pint = pinta_acento(asc, r["palavra"], idx, onde, 1.0)
    a, b, p = e1.tinta(acc), e1.tinta(asc), e1.tinta(pint)
    marca = M.diacriticos(r["palavra"])[k][1]
    e_real, e_pint = e1.e1(acc, asc, r["palavra"])[k], e1.e1(pint, asc, r["palavra"])[k]

    ax[i, 0].imshow(1 - acc, cmap="gray", vmin=0, vmax=1)
    limpa(ax[i, 0], f'{marca}: "{r["palavra"]}"')
    ax[i, 1].imshow(1 - asc, cmap="gray", vmin=0, vmax=1)
    limpa(ax[i, 1], f'gêmea: "{M.sem_acento(r["palavra"])}"')
    painel_dif(ax[i, 2], a, b); caixa(ax[i, 2], b, r["palavra"], idx, onde == M.ACIMA)
    limpa(ax[i, 2], f"diferença real     E1 = {e_real:+.3f}")
    painel_dif(ax[i, 3], p, b); caixa(ax[i, 3], b, r["palavra"], idx, onde == M.ACIMA)
    limpa(ax[i, 3], f"com acento pintado     E1 = {e_pint:+.3f}")

fig.tight_layout(rect=[0, 0.035, 1, 0.965])
fig.text(0.5, 0.008, "Coluna 3 é o que o modelo fez; coluna 4 é como seria se ele "
         "tivesse desenhado o acento. O retângulo verde é a região medida.",
         ha="center", fontsize=9.5, style="italic")
fig.savefig("avaliacao_diacriticos/figura_e1_exemplos.png", dpi=125)
print("salvo: avaliacao_diacriticos/figura_e1_exemplos.png")
for r, g, idx, onde, k in sel:
    print(f"   {M.diacriticos(r['palavra'])[k][1]:12s} {r['palavra']}")


# ---------------------------------------------------------------- falhas
# Todo escore alto aqui e falso positivo: o IAM nao desenha diacritico. Serve
# para ver COMO a metrica erra, nao so quanto.
medidos = []
for r in itens:
    if not r["acentuada"] or r.get("colapsada"):
        continue
    g = gem.get((r["par_id"], r["escritor"], r["semente"]))
    if not g:
        continue
    acc = M.carregar_tinta(f"{D}/{r['arquivo']}")
    asc = M.carregar_tinta(f"{D}/{g['arquivo']}")
    for k, (idx, marca, onde) in enumerate(M.diacriticos(r["palavra"])):
        medidos.append((e1.e1(acc, asc, r["palavra"])[k], r, g, idx, marca, onde))
medidos.sort(key=lambda t: -t[0])
piores = medidos[:4]

fig, ax = plt.subplots(4, 3, figsize=(11.5, 6.6))
fig.suptitle("Como o E1 erra — os 4 maiores escores do controle negativo\n"
             "o IAM não desenha diacrítico, então todo escore alto aqui é falso positivo",
             fontsize=12.5, y=0.995)
for i, (escore, r, g, idx, marca, onde) in enumerate(piores):
    acc = M.carregar_tinta(f"{D}/{r['arquivo']}")
    asc = M.carregar_tinta(f"{D}/{g['arquivo']}")
    a, b = e1.tinta(acc), e1.tinta(asc)
    ax[i, 0].imshow(1 - acc, cmap="gray", vmin=0, vmax=1)
    limpa(ax[i, 0], f'{marca}: "{r["palavra"]}"')
    ax[i, 1].imshow(1 - asc, cmap="gray", vmin=0, vmax=1)
    limpa(ax[i, 1], f'gêmea: "{M.sem_acento(r["palavra"])}"')
    painel_dif(ax[i, 2], a, b); caixa(ax[i, 2], b, r["palavra"], idx, onde == M.ACIMA)
    limpa(ax[i, 2], f"E1 = {escore:+.3f}   (deveria ser ~0)")
fig.tight_layout(rect=[0, 0.045, 1, 0.94])
fig.text(0.5, 0.01, "Não é acento: é o modelo redesenhando a palavra de outro jeito. "
         "Quando a mudança calha de cair dentro do retângulo, a métrica conta como acento.\n"
         "É por isso que o p95 do ruído de agudo (0,264) e cedilha (0,206) supera um "
         "acento nominal (0,166), e nessas duas marcas só a média agregada é confiável.",
         ha="center", fontsize=9, style="italic")
fig.savefig("avaliacao_diacriticos/figura_e1_falhas.png", dpi=125)
print("\nsalvo: avaliacao_diacriticos/figura_e1_falhas.png")
for escore, r, _, _, marca, _ in piores:
    print(f"   {escore:+.3f}  {marca:12s} {r['palavra']}")
n = len(medidos)
print(f"\ndistribuicao dos {n} negativos: "
      f"mediana={np.median([m[0] for m in medidos]):+.4f}  "
      f"p95={np.percentile([m[0] for m in medidos], 95):+.4f}  "
      f"max={medidos[0][0]:+.4f}")
