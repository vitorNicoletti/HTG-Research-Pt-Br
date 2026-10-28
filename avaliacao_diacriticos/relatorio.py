"""
Relatorio da metrica para a equipe, em resultados/metrica/ (na raiz):
README.md com a concordancia da metrica com a anotacao humana do modelo de
40 epocas e o resultado nos treinos, validacao.tsv com uma linha por imagem
anotada, e tres figuras (kappa, matriz das categorias, exemplos).

Recalcula a partir das planilhas da anotacao e dos resultados ja gravados por
avaliar_sinais.py e integridade.py. Nao gera imagem nova.

    python avaliacao_diacriticos/relatorio.py
"""
import collections
import csv
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODULO = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(MODULO)
sys.path[:0] = [MODULO, os.path.join(MODULO, "experimentos"), RAIZ, os.path.join(RAIZ, "scripts")]

from anotacao_dupla import ler_planilha, letras_marcadas  # noqa: E402
from diagnosticar_posicao import gabarito_por_letra  # noqa: E402
from sinais import Detector, categoria, sem_acento, sinais_pedidos, tem_pingo  # noqa: E402

RES = os.path.join(MODULO, "resultados")
SAIDA = os.path.join(RAIZ, "resultados", "metrica")
CATEGORIAS = ["correto", "a mais", "letra errada", "ausente"]
MODELOS = {
    "so_prof_11ep": "só professor, 11 ép.",
    "professor_16ep": "professor, 16 ép.",
    "sem_pares_16ep": "pares sem professor, 16 ép.",
    "fora_16ep": "fora da máscara, 16 ép.",
    "zona_fora_baixo_16ep": "fora em ruído baixo, 16 ép.",
    "zona_vogais_16ep": "zona vogais, 16 ép.",
    "zona_vogais_40ep": "zona vogais, 40 ép.",
}

AZUL, LARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, GRADE = "#1f1f1e", "#5c5b55", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRADE,
    "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "text.color": TINTA, "axes.spines.top": False, "axes.spines.right": False,
})


# --- estatisticas -----------------------------------------------------------

def kappa(a, b):
    """Kappa de Cohen entre duas listas de rotulos."""
    if not a:
        return float("nan")
    classes = set(a) | set(b)
    obs = sum(x == y for x, y in zip(a, b)) / len(a)
    acaso = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in classes)
    return float("nan") if acaso == 1 else (obs - acaso) / (1 - acaso)


def binaria(pares):
    """pares: [(referencia, metrica)] booleanos -> dicionario de medidas."""
    n = len(pares)
    vp = sum(h and d for h, d in pares)
    fp = sum(d and not h for h, d in pares)
    fn = sum(h and not d for h, d in pares)
    vn = n - vp - fp - fn
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    prec, rec = div(vp, vp + fp), div(vp, vp + fn)
    return {"n": n, "concordam": vp + vn, "acuracia": div(vp + vn, n), "precisao": prec, "recall": rec,
            "especificidade": div(vn, vn + fp), "f1": div(2 * prec * rec, prec + rec) if vp else float("nan"),
            "kappa": kappa([h for h, _ in pares], [d for _, d in pares]),
            "vp": vp, "fp": fp, "fn": fn, "vn": vn,
            "ref_sim": div(vp + fn, n), "met_sim": div(vp + fp, n)}


def f(v, pct=False):
    if v != v:          # nan
        return "-"
    return f"{v:.0%}" if pct else f"{v:.2f}".replace(".", ",")


def fk(v):
    if v != v:
        return "-"
    return "0,00" if abs(v) < 0.005 else f"{v:+.2f}".replace(".", ",")


def pc(x):
    """'71.7' -> '71,7%'; vazio -> '-'."""
    return f"{float(x):.1f}%".replace(".", ",") if x not in ("", None) else "-"


COMO_LER = """Modelo final (zona vogais, 40 épocas), 212 imagens. Bruno e Vinícius anotaram
cada um sozinho; o gabarito é a resposta dos dois onde concordam e a decisão do
Leonardo onde discordam. Palavras com i ou j ficam fora das perguntas sobre
marca em cima (o pingo parece acento), por isso essas têm 180 imagens.

Kappa: concordância descontada do que viria por acaso; 1 é perfeita, 0 é acaso.
Precisão: dos "sim" da métrica, quantos o gabarito confirma. Recall: dos "sim"
do gabarito, quantos a métrica acha.
"""

CABECALHO = """# A métrica em números

Quanto a métrica concorda com as pessoas, e o que ela mostra nos treinos.
Gerado por `python avaliacao_diacriticos/relatorio.py`; não edite à mão.
Uma linha por imagem anotada em [validacao.tsv](validacao.tsv)."""

FIGURAS = """![Kappa: pessoas e métrica](concordancia_kappa.png)

![Matriz das categorias](matriz_categorias.png)

O erro mais comum é o gabarito dizer correto e a métrica dizer ausente: o
sinal encosta na letra ou sai partido.

![Exemplos](exemplos_concordancia.png)"""

LIMITES = """## Limites

- A métrica não diz qual é o sinal (agudo, til...), só se há sinal e em que letra.
- É mais severa que as pessoas: 75% de palavras corretas contra 87%.
- O E2 (palavra degradada) não foi conferido por pessoas."""

PERGUNTAS = ["a imagem tem marca em cima?", "a imagem tem cedilha?", "esta letra tem marca em cima?"]


def mil(n):
    return f"{n:,}".replace(",", ".")


def tabela_dupla(medidas, cat_pares, a, b, com_precisao):
    """Uma tabela para a comparacao de a com b: uma linha por pergunta."""
    art = {"métrica": "a métrica", "gabarito": "o gabarito"}
    cab = (f"| pergunta | comparadas | mesma resposta | kappa | só {art.get(a, a)} disse sim | "
           f"só {art.get(b, b)} disse sim |"
           + (" precisão | recall |" if com_precisao else ""))
    out = [cab, "|---" * (cab.count("|") - 1) + "|"]
    for nome in PERGUNTAS:
        m = medidas[nome]
        un = "letras" if "letra" in nome else "imagens"
        rot = nome
        linha = (f"| {rot} | {mil(m['n'])} {un} | {mil(m['concordam'])} ({f(m['acuracia'], True)}) | "
                 f"{fk(m['kappa'])} | {m['fn']} | {m['fp']} |")
        if com_precisao:
            linha += f" {f(m['precisao'])} | {f(m['recall'])} |"
        out.append(linha)
    x, y = [p for p, _ in cat_pares], [q for _, q in cat_pares]
    iguais = sum(p == q for p, q in cat_pares)
    out.append(f"| a palavra acentuada cai na mesma categoria? | {len(cat_pares)} imagens | "
               f"{iguais} ({iguais / len(cat_pares):.0%}) | {fk(kappa(x, y))} | - | - |"
               + (" - | - |" if com_precisao else ""))
    return "\n".join(out)


def classe(palavra):
    return " e ".join(sorted({s for _, s in sinais_pedidos(palavra)}, key=lambda s: s != "cedilha"))


def ler_tsv(caminho):
    with open(caminho, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


# --- dados da anotacao ---------------------------------------------------------

def carregar():
    """Uma entrada por imagem anotada, com as respostas das pessoas, do
    gabarito e da metrica."""
    chave = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_chave.csv"), encoding="utf-8"))}
    gab = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_gabarito.csv"), encoding="utf-8"))}
    b = ler_planilha(os.path.join(RES, "planilha_anotacao_bruno.csv"))
    v = ler_planilha(os.path.join(RES, "planilha_anotacao_vini.csv"))
    letras_gab = gabarito_por_letra()
    d = Detector()
    itens = []
    for img, c in chave.items():
        palavra = c["palavra"]
        g = np.asarray(Image.open(os.path.join(MODULO, c["painel"])).convert("L"), dtype=np.float32)
        r = d.avaliar(g, palavra)
        ced_b, ced_v = letras_marcadas(b[img]["cedilha"]), letras_marcadas(v[img]["cedilha"])
        itens.append({
            "imagem": img, "palavra": palavra, "tipo": c["tipo"], "painel": c["painel"],
            "classe": classe(palavra) if sinais_pedidos(palavra) else "sem sinal pedido",
            "pingo": tem_pingo(palavra), "n_letras": len(sem_acento(palavra)),
            "gab_letras": letras_gab[img], "gab_cedilha": gab[img]["cedilha"] == "sim",
            "b_letras": letras_marcadas(b[img]["marcas_em_cima"]), "v_letras": letras_marcadas(v[img]["marcas_em_cima"]),
            "b_cedilha": None if ced_b is None else bool(ced_b), "v_cedilha": None if ced_v is None else bool(ced_v),
            "met": r,
        })
    return itens


def categoria_de(item, letras, cedilha):
    if letras is None or cedilha is None:
        return None
    return categoria(item["palavra"], letras, cedilha)


# --- tabelas ---------------------------------------------------------------------

def tabelas(itens):
    out = []
    sem_ij = [i for i in itens if not i["pingo"]]
    pedidas = [i for i in sem_ij if sinais_pedidos(i["palavra"])]

    quem = {
        "Bruno": (lambda i: i["b_letras"], lambda i: i["b_cedilha"]),
        "Vinícius": (lambda i: i["v_letras"], lambda i: i["v_cedilha"]),
        "gabarito": (lambda i: i["gab_letras"], lambda i: i["gab_cedilha"]),
        "métrica": (lambda i: i["met"]["letras"], lambda i: i["met"]["cedilha"]),
    }

    def comparar(a, b):
        """Perguntas de sim ou nao e categorias, com a como referencia."""
        (al, ac), (bl, bc) = quem[a], quem[b]
        ok_l = lambda i: not i["pingo"] and al(i) is not None and bl(i) is not None  # noqa: E731
        ok_c = lambda i: ac(i) is not None and bc(i) is not None  # noqa: E731
        m = {}
        m["a imagem tem marca em cima?"] = binaria([(bool(al(i)), bool(bl(i))) for i in itens if ok_l(i)])
        m["a imagem tem cedilha?"] = binaria([(ac(i), bc(i)) for i in itens if ok_c(i)])
        p = []
        for i in itens:
            if ok_l(i):
                p += [(k in al(i), k in bl(i)) for k in range(1, i["n_letras"] + 1)]
        m["esta letra tem marca em cima?"] = binaria(p)

        cats = [(categoria_de(i, al(i), ac(i)), categoria_de(i, bl(i), bc(i)), i) for i in pedidas if ok_l(i)]
        cats = [(x, y, i) for x, y, i in cats if x is not None and y is not None]
        return m, cats

    out.append(COMO_LER)
    secoes = [
        ("1. A métrica contra o gabarito", "gabarito", "métrica", True, None),
        ("2. Bruno contra Vinícius", "Bruno", "Vinícius", False,
         "Antes do desempate. É o teto: a métrica não tem como concordar com as pessoas mais do que elas "
         "entre si."),
    ]
    guardado = {}
    for titulo, a, b, prec, texto in secoes:
        m, cats = comparar(a, b)
        guardado[(a, b)] = (m, cats)
        out.append(f"## {titulo}\n")
        if texto:
            out.append(texto + "\n")
        out.append(tabela_dupla(m, [(x, y) for x, y, _ in cats], a, b, prec))
        out.append("")

    met, cat3 = guardado[("gabarito", "métrica")]
    pes, catp = guardado[("Bruno", "Vinícius")]
    cat = cat3
    hs, ds = [h for h, _, _ in cat], [d for _, d, _ in cat]
    resumo = [(n, pes[n]["kappa"], met[n]["kappa"]) for n in PERGUNTAS[:3]]
    resumo.append(("a palavra cai na mesma categoria?",
                   kappa([x for x, _, _ in catp], [y for _, y, _ in catp]), kappa(hs, ds)))

    out.append("## 3. Categoria da palavra, métrica contra o gabarito\n")
    out.append("Palavras acentuadas sem i nem j. Linha: gabarito. Coluna: métrica.\n")
    out.append("| gabarito \\ métrica | " + " | ".join(CATEGORIAS) + " | total |\n|---|" + "---|" * (len(CATEGORIAS) + 1))
    m = collections.Counter((h, d) for h, d, _ in cat)
    for h in CATEGORIAS:
        out.append(f"| {h} | " + " | ".join(str(m[(h, d)]) for d in CATEGORIAS) + f" | {hs.count(h)} |")
    out.append("| total | " + " | ".join(str(ds.count(d)) for d in CATEGORIAS) + f" | {len(cat)} |")
    return "\n".join(out), resumo, m, cat


def tabelas_dos_modelos():
    out = ["## 4. A métrica nos treinos\n",
           "1.200 imagens de palavras acentuadas por modelo, no protocolo de validação.\n",
           "| modelo | correto | a mais | letra errada | ausente | palavra degradada (E2) |",
           "|---|---|---|---|---|---|"]
    sin = {r["modelo"]: r for r in ler_tsv(os.path.join(RES, "sinais", "resumo_sinais.tsv"))}
    integ = {r["modelo"]: r for r in ler_tsv(os.path.join(RES, "integridade", "resumo_integridade.tsv"))}
    inteiro = lambda x: f"{float(x):.0f}%"  # noqa: E731
    for k, nome in MODELOS.items():
        r = sin[k]
        out.append(f"| {nome} | {inteiro(r['correto'])} | {inteiro(r['a mais'])} | {inteiro(r['letra errada'])} | "
                   f"{inteiro(r['ausente'])} | {inteiro(integ[k]['piora'])} |")
    return "\n".join(out)


# --- figuras --------------------------------------------------------------------

def salvar(fig, nome):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SAIDA, f"{nome}.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_concordancia(resumo):
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    y = np.arange(len(resumo))
    for k, (nome, cor, idx) in enumerate((("Bruno × Vinícius", AZUL, 1), ("métrica × gabarito", LARANJA, 2))):
        vals = [r[idx] for r in resumo]
        ax.barh(y + (k - 0.5) * 0.36, vals, 0.32, color=cor, label=nome)
        for yi, v in zip(y, vals):
            ax.text(v + 0.015, yi + (k - 0.5) * 0.36, f"{v:.2f}".replace(".", ","), va="center", fontsize=7.5)
    ax.set_yticks(y, [r[0] for r in resumo])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("kappa de Cohen")
    for x, txt in ((0.4, "moderada"), (0.6, "substancial"), (0.8, "quase perfeita")):
        ax.axvline(x, color=GRADE, linewidth=0.8, zorder=0)
        ax.text(x + 0.01, len(resumo) - 0.45, txt, fontsize=6.5, color=TINTA_2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Concordância: as pessoas entre si e a métrica contra o gabarito", loc="left", fontsize=10, pad=22)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8, handlelength=1)
    fig.text(0.01, -0.1, "Linhas cinza: faixas de Landis e Koch. Modelo de 40 épocas, 212 imagens.",
             fontsize=7, color=TINTA_2)
    salvar(fig, "concordancia_kappa")


def fig_matriz(m, total_h, total_d):
    mat = np.array([[m[(h, d)] for d in CATEGORIAS] for h in CATEGORIAS])
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.15)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, mat[a, b], ha="center", va="center", fontsize=9,
                    color="white" if mat[a, b] > mat.max() * 0.6 else TINTA)
    ax.set_xticks(range(4), CATEGORIAS, rotation=0, fontsize=8)
    ax.set_yticks(range(4), CATEGORIAS, fontsize=8)
    ax.set_xlabel("métrica")
    ax.set_ylabel("gabarito humano")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Categoria por palavra acentuada", loc="left", fontsize=10)
    salvar(fig, "matriz_categorias")


def fig_exemplos(cat):
    """Folha com casos de concordancia e de cada discordancia comum."""
    casos = [("correto", "correto", "pessoas e métrica: correto"),
             ("correto", "ausente", "pessoas: correto · métrica: ausente"),
             ("correto", "a mais", "pessoas: correto · métrica: a mais"),
             ("correto", "letra errada", "pessoas: correto · métrica: letra errada"),
             ("a mais", "a mais", "pessoas e métrica: a mais")]
    porlinha = 4
    fig, eixos = plt.subplots(len(casos), porlinha, figsize=(7.6, 0.95 * len(casos) + 0.2),
                              gridspec_kw={"hspace": 0.75, "wspace": 0.05})
    for li, (h, d, titulo) in enumerate(casos):
        sel = [i for hh, dd, i in cat if hh == h and dd == d]
        sel = sorted(sel, key=lambda i: (i["classe"], i["palavra"]))
        passo = max(1, len(sel) // porlinha)
        sel = sel[::passo][:porlinha]
        for k in range(porlinha):
            ax = eixos[li, k]
            ax.axis("off")
            if k < len(sel):
                i = sel[k]
                ax.imshow(np.asarray(Image.open(os.path.join(MODULO, i["painel"])).convert("L")), cmap="gray",
                          vmin=0, vmax=255)
                ax.set_title(i["palavra"], fontsize=7.5, color=TINTA_2, pad=2)
        n = sum(1 for hh, dd, _ in cat if hh == h and dd == d)
        eixos[li, 0].text(0, -30, f"{titulo}  ({n} {'imagem' if n == 1 else 'imagens'})",
                          fontsize=8.5, color=TINTA, transform=eixos[li, 0].transData)
    salvar(fig, "exemplos_concordancia")


def main():
    os.makedirs(SAIDA, exist_ok=True)
    itens = carregar()
    texto, resumo, matriz, cat = tabelas(itens)
    hs, ds = [h for h, _, _ in cat], [d for _, d, _ in cat]
    partes = [CABECALHO, texto, FIGURAS, tabelas_dos_modelos(), LIMITES]
    with open(os.path.join(SAIDA, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(partes) + "\n")
    with open(os.path.join(SAIDA, "validacao.tsv"), "w", encoding="utf-8") as fh:
        fh.write("imagem\tpalavra\ttipo\tclasse\tletras_gabarito\tletras_metrica\tcedilha_gabarito\t"
                 "cedilha_metrica\tcategoria_gabarito\tcategoria_metrica\n")
        for i in itens:
            j = lambda s: "" if s is None else " ".join(map(str, sorted(s)))  # noqa: E731
            cg = categoria_de(i, i["gab_letras"], i["gab_cedilha"]) if not i["pingo"] else None
            fh.write(f"{i['imagem']}\t{i['palavra']}\t{i['tipo']}\t{i['classe']}\t{j(i['gab_letras'])}\t"
                     f"{j(i['met']['letras'])}\t{int(i['gab_cedilha'])}\t"
                     f"{'' if i['met']['cedilha'] is None else int(i['met']['cedilha'])}\t{cg or ''}\t"
                     f"{i['met']['categoria'] or ''}\n")
    fig_concordancia(resumo)
    fig_matriz(matriz, hs, ds)
    fig_exemplos(cat)
    print(open(os.path.join(SAIDA, "README.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
