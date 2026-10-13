"""As medidas ainda melhoram com mais epocas? Um painel por medida, uma linha
por treino, a partir dos paineis.tsv guardados em
diagnostico/resultados/acento_separado/.

Medidas (ver docs/15):
  acento a mais      marca em cima nas palavras cujo unico sinal e a cedilha
  acento pedido      marca nas palavras com agudo, til ou circunflexo
  ... sem haste      o mesmo, so nas palavras sem letra de haste alta
  onde nao se pede   marca no esqueleto e nas palavras sem acento
  CER                leitor independente, contra a palavra sem acentos

    python diagnostico/grafico_epocas.py
"""
import csv
import os
import statistics as st
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(RAIZ, "diagnostico", "resultados", "acento_separado")
SAIDA = os.path.join(R, "evolucao_por_epocas.png")

# treino -> [(epocas, pasta da avaliacao, rotulo la dentro)]
TREINOS = {
    "professor": [(6, "avaliacao_professor_val", "professor_6ep"), (11, "avaliacao_so_prof_val", "professor_11ep"),
                  (16, "avaliacao_professor_val", "professor_16ep")],
    "professor e zona vogais": [(6, "avaliacao_zona_vogais_val", "zona_vogais_6ep"), (16, "avaliacao_zona_vogais_val", "zona_vogais_16ep"),
                                (21, "avaliacao_zona_vogais_30ep_val", "zona_vogais_21ep"), (30, "avaliacao_zona_vogais_30ep_val", "zona_vogais_30ep")],
}
COR = {"professor": "#2a78d6", "professor e zona vogais": "#eb6834"}   # validadas (dataviz), modo claro
TINTA, TINTA2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HASTE = set("bdfhklt")
CEDILHA = "̧"


def sinais(p):
    return {c for c in unicodedata.normalize("NFD", p) if unicodedata.combining(c)}


def medidas(pasta, rotulo):
    with open(os.path.join(R, pasta, "paineis.tsv"), encoding="utf-8") as f:
        rs = [r for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if r["modelo"] == rotulo]
    if len(rs) != 3200:
        raise SystemExit(f"{pasta}:{rotulo}: {len(rs)} linhas, esperava 3200")
    marca = lambda v: 100 * sum(int(r["marca"]) for r in v) / len(v)
    cer = lambda v: st.mean(float(r["cer"]) for r in v)
    ac = [r for r in rs if r["tipo"] == "acentuada"]
    so_ced = [r for r in ac if sinais(r["palavra"]) == {CEDILHA}]
    vis = [r for r in ac if sinais(r["palavra"]) != {CEDILHA}]
    resto = [r for r in rs if r["tipo"] != "acentuada"]
    return {"a_mais": marca(so_ced), "pedido": marca(vis),
            "sem_haste": marca([r for r in vis if not set(r["palavra"]) & HASTE]),
            "indevido": marca(resto), "cer_ac": cer(ac), "cer_sem": cer(resto)}


# (chave, titulo, melhor para onde, unidade, (ymin, ymax), referencia ou None)
PAINEIS = [
    ("a_mais", "Acento a mais", "menor é melhor", "%", (0, 90), (6, "modelo original")),
    ("pedido", "Acento pedido", "maior é melhor", "%", (50, 100), None),
    ("sem_haste", "Acento pedido, palavras sem haste", "maior é melhor", "%", (50, 100), None),
    ("indevido", "Acento onde não se pede", "menor é melhor", "%", (0, 12), (5, "modelo original")),
    ("cer_ac", "CER das acentuadas", "menor é melhor", "", (0.2, 0.6), (0.27, "imagens da base")),
    ("cer_sem", "CER das palavras sem acento", "menor é melhor", "", (0.1, 0.3), (0.195, "modelo original")),
]


def main():
    dados = {t: [(ep, medidas(p, r)) for ep, p, r in pts] for t, pts in TREINOS.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRADE,
                         "axes.labelcolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2})
    fig, eixos = plt.subplots(2, 3, figsize=(13, 7), facecolor=FUNDO)
    for ax, (k, titulo, sentido, un, (y0, y1), ref) in zip(eixos.flat, PAINEIS):
        ax.set_facecolor(FUNDO)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.grid(axis="y", color=GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        if ref:
            ax.axhline(ref[0], color=TINTA2, linewidth=1, linestyle=(0, (4, 3)))
        fins = []
        for t, pts in dados.items():
            xs, ys = [ep for ep, _ in pts], [m[k] for _, m in pts]
            ax.plot(xs, ys, color=COR[t], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=FUNDO, markeredgewidth=1.5, label=t, clip_on=False, zorder=3)
            fins.append((xs[-1], ys[-1]))
        # valor no ultimo ponto de cada linha, em tinta de texto, acima do ponto;
        # se os dois terminam juntos, o de baixo vai para baixo
        fins.sort(key=lambda p: p[1])
        perto = len(fins) == 2 and fins[0][0] == fins[1][0] and fins[1][1] - fins[0][1] < (y1 - y0) * 0.12
        for n, (x, y) in enumerate(fins):
            baixo = perto and n == 0
            ax.annotate(f"{y:.0f}%" if un else f"{y:.2f}".replace(".", ","), (x, y),
                        xytext=(0, -11 if baixo else 9), textcoords="offset points", color=TINTA,
                        fontsize=9.5, ha="center", va="top" if baixo else "bottom")
        ax.set_xlim(4, 32)
        ax.set_ylim(y0, y1)
        ax.set_xticks([6, 11, 16, 21, 30])
        ax.tick_params(length=0)
        if un:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        else:
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}".replace(".", ","))
        ax.set_title(titulo, loc="left", color=TINTA, fontsize=11.5, pad=18)
        ax.text(0, 1.035, sentido + (f"  ·  tracejado = {ref[1]}" if ref else ""), transform=ax.transAxes,
                color=TINTA2, fontsize=9)
    for ax in eixos[1]:
        ax.set_xlabel("épocas de treino")
    h, l = eixos.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.05, 0.945), ncol=2, frameon=False, fontsize=10.5,
               labelcolor=TINTA, handlelength=2.2, columnspacing=2.5)
    fig.suptitle("Medidas da validação ao longo do treino", x=0.06, y=0.975, ha="left", color=TINTA, fontsize=14)
    fig.subplots_adjust(left=0.06, right=0.975, top=0.83, bottom=0.085, hspace=0.5, wspace=0.26)
    fig.savefig(SAIDA, dpi=140, facecolor=FUNDO)
    print(SAIDA)
    for t, pts in dados.items():
        for ep, m in pts:
            print(f"{t:26s} {ep:2d} ep  " + "  ".join(f"{k} {v:.3f}" if v < 1 else f"{k} {v:.1f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
