"""Figuras do artigo, em resultados/figuras/artigo/ (PNG 200 dpi e PDF).

    python scripts/figuras_artigo.py [--imagens arquivo/figuras_artigo]

Graficos (so precisam das tabelas versionadas da metrica):
  categorias_por_modelo   E1: correto / a mais / letra errada / ausente
  quatro_categorias       E1 x E2: acerto / omissao / degradacao com e sem diacritico
  por_classe              correto por classe de sinal, pessoas contra metrica
  evolucao_epocas         zona vogais de 6 a 40 epocas
Imagens (precisam de --imagens, saida de scripts/gerar_original_e_melhor.py):
  original_e_melhor       mesmas palavras, escritores e semente nos dois modelos
  como_a_metrica_olha     colunas das letras, marcas em cima, corte da cedilha
                          e a imagem com os sinais apagados (E2)

Paleta categorica validada pela skill dataviz (azul, laranja, verde-agua,
amarelo); o contraste das duas ultimas com o fundo e baixo, por isso todo
segmento largo leva o numero escrito e ha legenda.
A palavra de validacao "rape" nunca aparece nas figuras.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "avaliacao_diacriticos"))

SAIDA = os.path.join(RAIZ, "resultados", "figuras", "artigo")
RES = os.path.join(RAIZ, "avaliacao_diacriticos", "resultados")

CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA, TINTA_2, GRADE, FUNDO = "#1f1f1e", "#5c5b55", "#e4e3df", "#ffffff"
NAO_MOSTRAR = {"rapé", "rape"}

ROTULOS = {
    "so_prof_11ep": "só professor, 11 ép.",
    "professor_16ep": "professor, 16 ép.",
    "sem_pares_16ep": "pares sem professor, 16 ép.",
    "fora_16ep": "fora da máscara, 16 ép.",
    "zona_fora_baixo_16ep": "fora em ruído baixo, 16 ép.",
    "zona_vogais_16ep": "zona vogais, 16 ép.",
    "zona_vogais_40ep": "zona vogais, 40 ép.",
}
EPOCAS = [("zona_vogais_6ep", 6), ("zona_vogais_16ep", 16), ("zona_vogais_21ep", 21),
          ("zona_vogais_30ep", 30), ("zona_vogais_40ep", 40)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRADE,
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "text.color": TINTA, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
})


def ler_tsv(caminho):
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def salvar(fig, nome):
    os.makedirs(SAIDA, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SAIDA, f"{nome}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  ", nome)


def barras_empilhadas(linhas, colunas, nomes, titulo, nome, nota):
    """Barras horizontais de 100%, uma por modelo, com o numero escrito nos
    segmentos de pelo menos 3%."""
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(linhas) + 1.3))
    for i, (rot, valores) in enumerate(linhas):
        x = 0.0
        for k, v in enumerate(valores):
            ax.barh(i, v, left=x, height=0.66, color=CORES[k], edgecolor=FUNDO, linewidth=1.5)
            if v >= 3:
                ax.text(x + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color=TINTA)
            x += v
    ax.set_yticks(range(len(linhas)), [ROTULOS[r] for r, _ in linhas])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(titulo, loc="left", fontsize=10, color=TINTA, pad=24)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for c in CORES[:len(nomes)]], labels=nomes,
              ncol=len(nomes), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
              fontsize=8, handlelength=1, columnspacing=1.2)
    fig.text(0.01, -0.02, nota, fontsize=7, color=TINTA_2, ha="left", va="top")
    salvar(fig, nome)


def categorias_por_modelo():
    res = {r["modelo"]: r for r in ler_tsv(os.path.join(RES, "sinais", "resumo_sinais.tsv"))}
    nomes = ["correto", "a mais", "letra errada", "ausente"]
    linhas = [(m, [float(res[m][c]) for c in nomes]) for m in ROTULOS]
    barras_empilhadas(linhas, nomes, nomes, "Palavras acentuadas por resultado da métrica (E1)",
                      "categorias_por_modelo",
                      "1.200 imagens por modelo: 30 palavras de validação, 20 escritores do iam_test, sementes 42 e 43.")


def quatro_categorias():
    res = {r["modelo"]: r for r in ler_tsv(os.path.join(RES, "integridade", "resumo_integridade.tsv"))}
    cols = ["acerto", "omissao", "degradacao com diacritico", "degradacao sem diacritico"]
    nomes = ["acerto", "omissão", "degradação com diacrítico", "degradação sem diacrítico"]
    linhas = [(m, [float(res[m][c]) for c in cols]) for m in ROTULOS]
    barras_empilhadas(linhas, cols, nomes, "Sinal (E1) cruzado com a integridade da palavra (E2)",
                      "quatro_categorias",
                      "Com diacrítico: os sinais pedidos estão presentes, sem olhar a letra. Degradada: CER da acentuada "
                      "menos CER do esqueleto\n(mesmo escritor e semente) acima de 0,2, as duas lidas com os sinais apagados.")


def por_classe():
    """Correto pelas pessoas e pela metrica no modelo de 40 epocas, da tabela
    de categorias_40ep.txt."""
    linhas = []
    with open(os.path.join(RES, "categorias_40ep.txt"), encoding="utf-8") as f:
        for l in f:
            if "correto pessoas" in l:
                classe = l.split("n=")[0].strip()
                n = int(l.split("n=")[1].split()[0])
                p = int(l.split("pessoas")[1].split("%")[0])
                m = int(l.split("metrica")[1].split("%")[0])
                linhas.append((classe, n, p, m))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(linhas))
    larg = 0.36
    for k, (nome, idx) in enumerate((("pessoas", 2), ("métrica", 3))):
        vals = [l[idx] for l in linhas]
        ax.bar(x + (k - 0.5) * (larg + 0.03), vals, larg, color=CORES[k], label=nome)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * (larg + 0.03), v + 1.5, f"{v}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [f"{l[0]}\n(n = {l[1]})" for l in linhas])
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.yaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Palavras corretas, separadas pelo sinal que a palavra pede (40 épocas)", loc="left", fontsize=10, pad=22)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8, handlelength=1)
    fig.text(0.01, -0.02, "Pessoas: gabarito de dois anotadores com desempate, 135 imagens com acento pedido.",
             fontsize=7, color=TINTA_2, ha="left", va="top")
    salvar(fig, "por_classe")


def evolucao_epocas():
    sin = {r["modelo"]: r for r in ler_tsv(os.path.join(RES, "sinais", "resumo_sinais.tsv"))}
    integ = {r["modelo"]: r for r in ler_tsv(os.path.join(RES, "integridade", "resumo_integridade.tsv"))}
    series = [("correto (E1)", [float(sin[m]["correto"]) for m, _ in EPOCAS]),
              ("palavra degradada (E2)", [float(integ[m]["piora"]) for m, _ in EPOCAS]),
              ("marca em palavra sem acento", [float(sin[m]["marca_indevida"]) for m, _ in EPOCAS])]
    ep = [e for _, e in EPOCAS]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for k, (nome, vals) in enumerate(series):
        ax.plot(ep, vals, color=CORES[k], linewidth=2, marker="o", markersize=5,
                markeredgecolor=FUNDO, markeredgewidth=1.2, label=nome)
        ax.text(ep[-1] + 1.2, vals[-1], f"{vals[-1]:.0f}%", va="center", fontsize=8)
    ax.set_xticks(ep, [f"{e}" for e in ep])
    ax.set_xlim(4, 44)
    ax.set_xlabel("épocas de treino")
    ax.set_ylim(0, 80)
    ax.set_yticks(range(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
    ax.yaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Zona vogais ao longo do treino", loc="left", fontsize=10, pad=22)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8, handlelength=1.5,
              columnspacing=1.2)
    salvar(fig, "evolucao_epocas")


# --- figuras com imagens ---------------------------------------------------

def original_e_melhor(pasta, escritores=("547", "183", "590", "537")):
    from gerar_original_e_melhor import PALAVRAS, SEMENTE
    from PIL import Image
    acentuadas = [p for p in PALAVRAS if p not in NAO_MOSTRAR and not p.isascii()]
    modelos = [("original", "original"), ("zona_vogais_40ep", "zona vogais, 40 ép.")]
    # duas faixas por palavra (um modelo cada) e uma faixa vazia entre palavras
    alturas = sum(([1, 1, 0.35] for _ in acentuadas), [])[:-1]
    fig, eixos = plt.subplots(len(alturas), 1, figsize=(7.4, 0.5 * sum(alturas) + 0.3),
                              gridspec_kw={"height_ratios": alturas, "hspace": 0.04})
    esp = np.full((64, 8), 255, np.uint8)
    for ax in eixos:
        ax.axis("off")
    for i, palavra in enumerate(acentuadas):
        for j, (rot, nome) in enumerate(modelos):
            imgs = [np.asarray(Image.open(os.path.join(pasta, rot, f"{palavra}_{e}_{SEMENTE}.png"))) for e in escritores]
            ax = eixos[3 * i + j]
            ax.imshow(np.hstack(sum(([im, esp] for im in imgs), [])[:-1]), cmap="gray", vmin=0, vmax=255,
                      interpolation="lanczos")
            ax.text(-12, 32, nome, ha="right", va="center", fontsize=8, color=TINTA_2)
            if j == 0:
                ax.text(-12, -6, palavra, ha="right", va="bottom", fontsize=9.5, color=TINTA)
    fig.text(0.125, 0.06, f"Mesmos {len(escritores)} escritores do iam_test, mesmas referências de estilo e mesma "
             "semente nas duas linhas de cada palavra.\nPalavras de validação, uma por classe de sinal.",
             fontsize=7, color=TINTA_2, va="top")
    salvar(fig, "original_e_melhor")


def como_a_metrica_olha(pasta, palavra="prestação", escritor="547"):
    from gerar_original_e_melhor import SEMENTE
    from PIL import Image
    from medir_marcas import TOL
    from sinais import LIMIAR_CEDILHA, Detector, marcas_acima, sem_acento
    from acentos_sinteticos import geometria

    g = np.asarray(Image.open(os.path.join(pasta, "zona_vogais_40ep", f"{palavra}_{escritor}_{SEMENTE}.png")),
                   dtype=np.float32)
    d = Detector()
    letras = sem_acento(palavra)
    geo, fatias, x0 = d.fatias(g, letras)
    y0 = max(0, int(np.where(geometria.mascara_tinta(g))[0].min()) - 3)   # = recorte de medir_posicao_acento
    marcadas = d.letras_marcadas(g, palavra)
    nos_c = d.tinta_abaixo_dos_c(g, letras)
    apagada = d.apagar_sinais(g, palavra)

    fig, eixos = plt.subplots(1, 2, figsize=(7.4, 1.5), gridspec_kw={"wspace": 0.06})
    ax = eixos[0]
    ax.imshow(g, cmap="gray", vmin=0, vmax=255)
    topo, base = geo.topo_x + y0, geo.base + y0
    corte = int(round(geo.base + TOL * geo.altura_x)) + 1 + y0
    for y in (topo, base):
        ax.axhline(y, color=CORES[0], linewidth=0.8, linestyle="--")
    for k, (xa, xb) in enumerate(fatias):
        ax.axvline(xa + x0, color=TINTA_2, linewidth=0.4, alpha=0.5)
        achado = (k + 1) in marcadas or (letras[k] == "c" and nos_c.get(k, 0) > LIMIAR_CEDILHA)
        ax.text((xa + xb) / 2 + x0, 66, letras[k], ha="center", va="top", fontsize=7.5, clip_on=False,
                color=CORES[1] if achado else TINTA_2, fontweight="bold" if achado else "normal")
        if letras[k] == "c":
            ax.add_patch(Rectangle((xa + x0, corte), xb - xa, 63 - corte, color=CORES[1], alpha=0.18, linewidth=0))
    ax.axvline(fatias[-1][1] + x0, color=TINTA_2, linewidth=0.4, alpha=0.5)
    for x, y, w, h in marcas_acima(g):
        ax.add_patch(Rectangle((x - 1.5, y - 1.5), w + 2, h + 2, fill=False, edgecolor=CORES[1], linewidth=1.2))
    ax.set_title("o que a métrica vê (E1)", fontsize=9, loc="left")
    eixos[1].imshow(apagada, cmap="gray", vmin=0, vmax=255)
    eixos[1].set_title("sinais apagados, para o leitor (E2)", fontsize=9, loc="left")
    for ax in eixos:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(GRADE)
    fig.text(0.125, -0.1, "Tracejado azul: faixa do corpo da letra. Retângulo laranja: marca solta acima do corpo. "
             "Faixa laranja: região abaixo do c\nonde se mede a tinta da cedilha. Colunas: letras pelo alinhador; "
             "em laranja as letras em que um sinal foi achado.", fontsize=7, color=TINTA_2, va="top")
    salvar(fig, "como_a_metrica_olha")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagens", default=os.path.join(RAIZ, "arquivo", "figuras_artigo"))
    a = ap.parse_args()
    categorias_por_modelo()
    quatro_categorias()
    por_classe()
    evolucao_epocas()
    if os.path.isdir(os.path.join(a.imagens, "zona_vogais_40ep")):
        original_e_melhor(a.imagens)
        como_a_metrica_olha(a.imagens)
    else:
        print("sem --imagens: figuras com imagens puladas")


if __name__ == "__main__":
    main()
