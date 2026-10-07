"""Gera as figuras explicativas do E1, a partir do controle IAM puro.

    python avaliacao_diacriticos/experimentos/figuras.py

Saida em figuras/:
    e1_passo_a_passo.png   os 4 passos da metrica num par
    e1_exemplos.png        uma marca por linha
    e1_falhas.png          os maiores escores do negativo
    e1_vs_ssim_psnr.png    comparacao com as metricas sugeridas pelo orientador

A figura do alinhamento foi removida: ela escolhia o par em que alinhar mais
muda o escore, e esse criterio seleciona justamente os deslocamentos grandes,
que tendem a ser tambem os casos em que o modelo desenhou a palavra de outro
jeito -- ou seja, mostrava um caso em que alinhar NAO limpa. A justificativa
do alinhamento esta medida no README (p95 do ruido de 0.181 para 0.103, AUC
de 0.936 para 0.966) e comentada em e1.deslocamento().
"""
import json
import os
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # avaliacao_diacriticos/
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
    fig.suptitle("Os 4 maiores escores do controle negativo — e nem todos são erro\n"
                 "nas cedilhas, o modelo do IAM parece estar mesmo tentando "
                 "desenhar o diacrítico", fontsize=12.5, y=0.995)
    for i, d in enumerate(piores):
        a, b = e1.tinta(d["acc"]), e1.tinta(d["asc"])
        cinza(ax[i, 0], d["acc"]); limpa(ax[i, 0], f'{d["marca"]}: "{d["r"]["palavra"]}"', 8.5)
        cinza(ax[i, 1], d["asc"]); limpa(ax[i, 1], f'gêmea: "{M.sem_acento(d["r"]["palavra"])}"', 8.5)
        dif(ax[i, 2], a, b, f'E1 = {d["com"]:+.3f}   (deveria ser ~0)')
        ax[i, 2].title.set_fontsize(8.5)
        caixa(ax[i, 2], b, d["r"]["palavra"], d["idx"], d["onde"] == M.ACIMA)
    fig.tight_layout(rect=[0, 0.045, 1, 0.94])
    rodape(fig, "Repare em \"presença\": o final virou \"ga\" — um c com cauda "
                "abaixo da linha de base, que é a forma de um ç. Na gêmea \"presenca\"\n"
                "é um \"ca\" limpo. Medido: a cedilha é a única marca em que o extra "
                "de tinta aparece na COLUNA certa e na FAIXA certa mais do que nas\n"
                "outras colunas da mesma imagem (+0,0245, IC95 [+0,0085, +0,0416]). "
                "Ou seja, o IAM não é controle negativo puro para cedilha.", y=0.01)
    fig.savefig(f"{FIG}/e1_falhas.png", dpi=125)
    plt.close(fig)
    print("  figuras/e1_falhas.png")


# ------------------------------------------------ 5. E1 contra SSIM e PSNR
def comparar_metricas(csv="resultados/ssim_psnr.csv"):
    """Le o CSV de ssim_psnr.py e desenha a comparacao."""
    import csv as _csv
    from calibrar import auc

    caminho = os.path.join(AQUI, csv)
    linhas = list(_csv.DictReader(open(caminho, encoding="utf-8")))
    cols = [("e1", "E1 (atual)"), ("dissim_local", "1 - SSIM local"),
            ("psnr_local", "-PSNR local"), ("dissim_global", "1 - SSIM global"),
            ("psnr_global", "-PSNR global")]
    negs = [("NEG-gerador", "contra o gerador"),
            ("NEG-deriva", "contra a deriva"),
            ("NEG-deslocado", "contra acento no lugar errado")]

    def vals(col, grupo):
        return [float(l[col]) for l in linhas
                if l["grupo"] == grupo and l[col] not in ("", "nan")]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("E1 contra SSIM e PSNR, nos mesmos controles\n"
                 "positivo = acento nominal pintado; "
                 f"{len(linhas)} medidas do controle IAM",
                 fontsize=13, y=0.99)

    # (a) AUC por metrica e por controle
    larg = 0.26
    x = np.arange(len(cols))
    cores = ["#2b6cb0", "#d69e2e", "#c53030"]
    for i, (g, rot) in enumerate(negs):
        alturas = [auc(vals(c, g), vals(c, "POS-pintado")) for c, _ in cols]
        ax[0].bar(x + (i - 1) * larg, alturas, larg, label=rot, color=cores[i])
        for xi, h in zip(x + (i - 1) * larg, alturas):
            ax[0].text(xi, h + 0.015, f"{h:.2f}", ha="center", fontsize=7.5)
    ax[0].axhline(0.5, color="black", ls="--", lw=1)
    ax[0].text(len(cols) - 0.4, 0.52, "acaso", fontsize=8, style="italic")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([r for _, r in cols], fontsize=8.5, rotation=12)
    ax[0].set_ylabel("AUC")
    ax[0].set_ylim(0, 1.08)
    ax[0].legend(fontsize=8.5, loc="lower left")
    ax[0].set_title("(a) quanto cada escore separa acento de não-acento",
                    fontsize=10.5, loc="left")
    ax[0].grid(axis="y", alpha=0.25)

    # (b) por que o global falha. Os dois grupos sao normalizados JUNTOS,
    # senao cada um se estica para 0..1 e some a diferenca que interessa.
    jit = np.random.default_rng(0)
    for k, (col, rot) in enumerate([("e1", "E1 (atual)"),
                                    ("dissim_local", "1 - SSIM local"),
                                    ("dissim_global", "1 - SSIM global")]):
        pos, neg = np.array(vals(col, "POS-pintado")), np.array(vals(col, "NEG-gerador"))
        lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
        base = (2 - k) * 2.4
        for j, (v, cor, nome) in enumerate([(pos, "#2b6cb0", "com acento"),
                                            (neg, "#c53030", "sem acento")]):
            y = base + (1 - j) + (jit.random(len(v)) - 0.5) * 0.5
            ax[1].scatter((v - lo) / (hi - lo + 1e-9), y, s=4, alpha=0.3,
                          color=cor, label=nome if k == 0 else None)
            ax[1].text(1.04, base + (1 - j), nome.split()[0], fontsize=7.5,
                       color=cor, va="center")
        ax[1].text(-0.04, base + 0.5, rot, ha="right", va="center", fontsize=9.5)
        ax[1].axhline(base - 0.55, color="0.85", lw=0.8)
    ax[1].set_yticks([])
    ax[1].set_xlim(-0.42, 1.16)
    ax[1].set_xlabel("escore, normalizado com os DOIS grupos juntos")
    ax[1].legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax[1].set_title("(b) onde o azul (com acento) cai em relação ao vermelho",
                    fontsize=10.5, loc="left")

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    rodape(fig, "No E1 o azul está claramente à direita do vermelho, que é o "
                "esperado: com acento pontua mais. No SSIM global acontece o\n"
                "contrário, e por isso o AUC dele fica abaixo do acaso: ele mede a "
                "variação do gerador, que é muito maior que um diacrítico.\n"
                "O PSNR local chega a ganhar do E1 contra o gerador e contra a "
                "deriva, mas perde no controle do acento no lugar errado.", y=0.012)
    fig.savefig(f"{FIG}/e1_vs_ssim_psnr.png", dpi=125)
    plt.close(fig)
    print("  figuras/e1_vs_ssim_psnr.png")


def rng_jitter(n, esc=0.32):
    return (np.random.default_rng(0).random(n) - 0.5) * esc


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    passo_a_passo()
    exemplos()
    med = medir_tudo()
    falhas(med)
    comparar_metricas()
