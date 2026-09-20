"""Gera as figuras explicativas do E1, a partir do controle IAM puro.

    python avaliacao_diacriticos/figuras.py

Saida em figuras/:
    e1_passo_a_passo.png   os 4 passos da metrica num par
    e1_exemplos.png        uma marca por linha
    e1_falhas.png          os maiores escores do negativo (falsos positivos)
    e1_alinhamento.png     por que os gemeos precisam ser alinhados
"""
import json
import os
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import metrica as M, e1                              # noqa: E402
from teste_sensibilidade_diff import pinta_acento    # noqa: E402

D = os.path.join(AQUI, "amostras", "ger_iam_n696")
FIG = os.path.join(AQUI, "figuras")
itens = [json.loads(l) for l in open(f"{D}/manifest.jsonl", encoding="utf-8")]
gem = {(x["par_id"], x["escritor"], x["semente"]): x
       for x in itens if not x["acentuada"]}


def par(r):
    g = gem.get((r["par_id"], r["escritor"], r["semente"]))
    if not g:
        return None
    return (M.carregar_tinta(f"{D}/{r['arquivo']}"),
            M.carregar_tinta(f"{D}/{g['arquivo']}"), g)


def geometria(b, palavra):
    perfil = b.sum(axis=1)
    corpo = np.where(perfil >= 0.5 * perfil.max())[0]
    cols = np.where(b.any(axis=0))[0]
    return (corpo[0], corpo[-1] + 1, cols[0],
            (cols[-1] + 1 - cols[0]) / len(M.sem_acento(palavra)))


def caixa(ax, b, palavra, idx, acima, cor="tab:green"):
    topo, base, x0, larg = geometria(b, palavra)
    c0 = max(0, int(x0 + (idx - 0.5) * larg))
    c1 = int(x0 + (idx + 1.5) * larg)
    y0, alt = (-0.5, topo + 0.5) if acima else (base, b.shape[0] - base)
    ax.add_patch(Rectangle((c0, y0), c1 - c0, alt, ec=cor, fc="none", lw=1.6))


def limpa(ax, titulo=None, fs=9):
    ax.set_xticks([]); ax.set_yticks([])
    if titulo:
        ax.set_title(titulo, fontsize=fs, loc="left", pad=3)


def cinza(ax, img, titulo=None):
    ax.imshow(1 - img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    limpa(ax, titulo)


def dif(ax, a, b, titulo=None, alinhar=True):
    """Vermelho = tinta so na acentuada, azul = so na gemea, cinza = nas duas."""
    if alinhar:
        a = e1.alinha(a, b)
    d = a.astype(int) - b.astype(int)
    img = np.ones((*b.shape, 3))
    img[b] = 0.80
    img[d > 0] = (0.85, 0.10, 0.10)
    img[d < 0] = (0.10, 0.30, 0.85)
    ax.imshow(img, interpolation="nearest")
    limpa(ax, titulo)


def sobrepoe(ax, a, b, titulo=None):
    """As duas palavras em cores diferentes, para ver se estao no mesmo lugar."""
    img = np.ones((*b.shape, 3))
    img[a] = (0.85, 0.10, 0.10)
    img[b] = (0.10, 0.30, 0.85)
    img[a & b] = (0.55, 0.55, 0.55)
    ax.imshow(img, interpolation="nearest")
    limpa(ax, titulo)


def rodape(fig, texto, y=0.01):
    fig.text(0.5, y, texto, ha="center", fontsize=9, style="italic")


# ------------------------------------------------ 1. passo a passo
def passo_a_passo():
    alvo = next((r, *par(r)) for r in itens
                if r["acentuada"] and r.get("std", 0) > 0.20
                and "til" in [n for _, n, _ in M.diacriticos(r["palavra"])]
                and par(r))
    r, acc, asc, _ = alvo
    idx, _, onde = [d for d in M.diacriticos(r["palavra"]) if d[1] == "til"][0]
    k = [d[1] for d in M.diacriticos(r["palavra"])].index("til")
    a, b = e1.tinta(acc), e1.tinta(asc)
    pint = e1.tinta(pinta_acento(asc, r["palavra"], idx, onde, 1.0))
    topo, base, _, _ = geometria(b, r["palavra"])

    fig, ax = plt.subplots(6, 1, figsize=(9, 9.2))
    fig.suptitle(f'E1 passo a passo — par mínimo "{r["palavra"]}" / '
                 f'"{M.sem_acento(r["palavra"])}"\nmesmo escritor, mesma semente: '
                 f'a única variável é o texto', fontsize=13, y=0.985)
    cinza(ax[0], acc, f'1. o que o modelo gerou para "{r["palavra"]}"')
    cinza(ax[1], asc, f'1. e para "{M.sem_acento(r["palavra"])}" — a gêmea')
    cinza(ax[2], b.astype(float),
          "2. tinta (Otsu) e corpo da letra: altura-x e linha de base")
    for y, cor, rot in ((topo, "tab:blue", "altura-x"), (base, "tab:red", "linha de base")):
        ax[2].axhline(y, color=cor, lw=1.4)
        ax[2].text(258, y, f" {rot}", color=cor, fontsize=8, va="center")
    car = M.sem_acento(r["palavra"])[idx]
    cinza(ax[3], b.astype(float), f"3. só a faixa acima da altura-x, na coluna do "
                                  f"{idx+1}º caractere (o '{car}')")
    caixa(ax[3], b, r["palavra"], idx, onde == M.ACIMA)
    ax[3].patches[-1].set_facecolor("tab:green"); ax[3].patches[-1].set_alpha(0.25)
    dif(ax[4], a, b, f"4. diferença real (vermelho = tinta a mais, azul = a menos)"
                     f"    E1 = {e1.e1(acc, asc, r['palavra'])[k]:+.3f}")
    caixa(ax[4], b, r["palavra"], idx, onde == M.ACIMA)
    dif(ax[5], pint, b, f"5. o mesmo par, com um til nominal pintado"
                        f"    E1 = {e1.e1(pinta_acento(asc, r['palavra'], idx, onde, 1.0), asc, r['palavra'])[k]:+.3f}")
    caixa(ax[5], b, r["palavra"], idx, onde == M.ACIMA)
    fig.tight_layout(rect=[0, 0.055, 1, 0.945])
    rodape(fig, "O retângulo verde é a única região que conta. Na linha 4 não sobra "
                "tinta vermelha dentro dele: o modelo do IAM não desenhou o til.\n"
                "Repare que o 'd' tem haste alta nas DUAS imagens, então ele some "
                "sozinho na subtração — por isso ascendente não vira falso positivo.",
           y=0.006)
    fig.savefig(f"{FIG}/e1_passo_a_passo.png", dpi=130)
    plt.close(fig)
    print("  figuras/e1_passo_a_passo.png")


# ------------------------------------------------ 2. uma marca por linha
def exemplos():
    melhor = {}
    for r in itens:
        if not r["acentuada"] or r.get("colapsada") or not par(r):
            continue
        for k, (idx, marca, onde) in enumerate(M.diacriticos(r["palavra"])):
            if marca not in melhor or r["std"] > melhor[marca][0]["std"]:
                melhor[marca] = (r, idx, onde, k)
    sel = [melhor[m] for m in ["til", "cedilha", "agudo", "circunflexo", "grave"]
           if m in melhor]
    fig, ax = plt.subplots(len(sel), 4, figsize=(15, 1.65 * len(sel)))
    fig.suptitle("E1 em um par mínimo de cada marca — modelo do IAM puro, "
                 "que nunca viu português", fontsize=13, y=0.995)
    for i, (r, idx, onde, k) in enumerate(sel):
        acc, asc, _ = par(r)
        pint = pinta_acento(asc, r["palavra"], idx, onde, 1.0)
        a, b, p = e1.tinta(acc), e1.tinta(asc), e1.tinta(pint)
        marca = M.diacriticos(r["palavra"])[k][1]
        cinza(ax[i, 0], acc); limpa(ax[i, 0], f'{marca}: "{r["palavra"]}"', 8.5)
        cinza(ax[i, 1], asc); limpa(ax[i, 1], f'gêmea: "{M.sem_acento(r["palavra"])}"', 8.5)
        dif(ax[i, 2], a, b, f"diferença real     E1 = {e1.e1(acc, asc, r['palavra'])[k]:+.3f}")
        ax[i, 2].title.set_fontsize(8.5)
        caixa(ax[i, 2], b, r["palavra"], idx, onde == M.ACIMA)
        dif(ax[i, 3], p, b, f"com acento pintado     E1 = {e1.e1(pint, asc, r['palavra'])[k]:+.3f}")
        ax[i, 3].title.set_fontsize(8.5)
        caixa(ax[i, 3], b, r["palavra"], idx, onde == M.ACIMA)
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    rodape(fig, "Coluna 3 é o que o modelo fez; coluna 4 é como seria se ele tivesse "
                "desenhado o acento. O retângulo verde é a região medida.", y=0.008)
    fig.savefig(f"{FIG}/e1_exemplos.png", dpi=125)
    plt.close(fig)
    print("  figuras/e1_exemplos.png")


# ------------------------------------------------ 3. falhas e 4. alinhamento
def medir_tudo():
    saida = []
    for r in itens:
        if not r["acentuada"] or r.get("colapsada"):
            continue
        pr = par(r)
        if not pr:
            continue
        acc, asc, _ = pr
        a, b = e1.tinta(acc), e1.tinta(asc)
        if not a.any() or not b.any():
            continue
        com = e1.e1(acc, asc, r["palavra"])
        for k, (idx, marca, onde) in enumerate(M.diacriticos(r["palavra"])):
            sem = _sem_alinhar(a, b, r["palavra"], k)
            saida.append(dict(r=r, idx=idx, marca=marca, onde=onde, k=k,
                              com=com[k], sem=sem, acc=acc, asc=asc))
    return saida


def _sem_alinhar(a, b, palavra, k):
    topo, base, x0, larg = geometria(b, palavra)
    idx, _, onde = M.diacriticos(palavra)[k]
    c = slice(max(0, int(x0 + (idx - 0.5) * larg)), int(x0 + (idx + 1.5) * larg))
    f = slice(0, topo) if onde == M.ACIMA else slice(base, b.shape[0])
    ref = b[topo:base, c].sum()
    return float(a[f, c].sum() - b[f, c].sum()) / ref if ref else 0.0


def falhas(med):
    piores = sorted(med, key=lambda d: -d["com"])[:4]
    fig, ax = plt.subplots(4, 3, figsize=(11.5, 6.6))
    fig.suptitle("Como o E1 erra — os 4 maiores escores do controle negativo\n"
                 "o IAM não desenha diacrítico, então todo escore alto aqui é "
                 "falso positivo", fontsize=12.5, y=0.995)
    for i, d in enumerate(piores):
        a, b = e1.tinta(d["acc"]), e1.tinta(d["asc"])
        cinza(ax[i, 0], d["acc"]); limpa(ax[i, 0], f'{d["marca"]}: "{d["r"]["palavra"]}"', 8.5)
        cinza(ax[i, 1], d["asc"]); limpa(ax[i, 1], f'gêmea: "{M.sem_acento(d["r"]["palavra"])}"', 8.5)
        dif(ax[i, 2], a, b, f'E1 = {d["com"]:+.3f}   (deveria ser ~0)')
        ax[i, 2].title.set_fontsize(8.5)
        caixa(ax[i, 2], b, d["r"]["palavra"], d["idx"], d["onde"] == M.ACIMA)
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    rodape(fig, "Não é acento: é o modelo redesenhando a palavra de outro jeito. "
                "Quando a mudança calha de cair dentro do retângulo, vira acento.\n"
                "É por isso que na cedilha o ruído ainda supera um acento nominal, "
                "e só a média agregada é confiável nessa marca.", y=0.01)
    fig.savefig(f"{FIG}/e1_falhas.png", dpi=125)
    plt.close(fig)
    print("  figuras/e1_falhas.png")


def alinhamento(med):
    """Demonstra o fantasma do desalinhamento e o efeito de corrigi-lo."""
    d = max(med, key=lambda x: x["sem"] - x["com"])
    a, b = e1.tinta(d["acc"]), e1.tinta(d["asc"])
    dx, dy = e1.deslocamento(a, b)
    pal = d["r"]["palavra"]

    # painel 1: a MESMA palavra deslocada. Como as duas sao identicas, tudo que
    # aparece e artefato do deslocamento -- e o jeito mais direto de mostrar
    # que subtrair sem alinhar duplica cada traco.
    H, W = b.shape
    desl = np.zeros_like(b)
    desl[:, 8:] = b[:, :-8]

    fig, ax = plt.subplots(3, 1, figsize=(9.5, 6.4))
    fig.suptitle("Por que os gêmeos precisam ser alinhados antes da subtração",
                 fontsize=13, y=0.985)

    dif(ax[0], desl, b, "1. a MESMA palavra, deslocada 8 px, subtraída dela mesma:\n"
                        "   cada traço vira um par vermelho/azul — puro artefato, "
                        "não há acento nenhum aqui", alinhar=False)
    dif(ax[1], a, b, f'2. caso real "{pal}" / "{M.sem_acento(pal)}", sem alinhar '
                     f'(os gêmeos saíram {abs(dx)} px fora de lugar)'
                     f'     E1 = {d["sem"]:+.3f}  ← falso positivo', alinhar=False)
    caixa(ax[1], b, pal, d["idx"], d["onde"] == M.ACIMA)
    dif(ax[2], a, b, f"3. o mesmo caso, alinhado antes de subtrair "
                     f"(dx={dx:+d} px, dy={dy:+d} px)     E1 = {d['com']:+.3f}")
    caixa(ax[2], b, pal, d["idx"], d["onde"] == M.ACIMA)
    for e in ax:
        e.title.set_fontsize(9.5)

    fig.tight_layout(rect=[0, 0.13, 1, 0.94])
    rodape(fig, "Vermelho = tinta só na acentuada, azul = só na gêmea, cinza = nas "
                "duas. No painel 1 não existe acento: todo o vermelho vem de a\n"
                "de a palavra estar fora do lugar. É esse fantasma que caía dentro do "
                "retângulo verde no painel 2.  Alinhar corrige só o deslocamento —\n"
                "o vermelho que sobra no painel 3 é o modelo tendo desenhado a "
                "palavra de outro jeito, e isso a métrica não conserta.", y=0.012)
    fig.savefig(f"{FIG}/e1_alinhamento.png", dpi=130)
    plt.close(fig)
    print(f"  figuras/e1_alinhamento.png   ({pal}: {d['sem']:+.3f} -> {d['com']:+.3f}, "
          f"dx={dx:+d} dy={dy:+d})")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    passo_a_passo()
    exemplos()
    med = medir_tudo()
    falhas(med)
    alinhamento(med)
