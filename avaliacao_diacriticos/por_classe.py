"""
A metrica final separada por classe de sinal, como pede o planejamento do TCC
("metrica estratificada por classe de caractere").

Le os resultados por imagem de avaliar_sinais.py (resultados/sinais/
sinais_<modelo>.tsv) e, para cada modelo e cada classe, da a fracao de
palavras em cada categoria e a presenca de cada sinal pedido.

Classes, pelo que a palavra pede: agudo, til, circunflexo, cedilha e
cedilha com til. No protocolo do scripts/avaliar_pt.py elas tem 10, 5, 2, 11 e
2 palavras, com 40 imagens cada; circunflexo e cedilha com til dependem de
duas palavras so.

Grava resultados/sinais/por_classe.tsv e resultados/sinais/por_classe.md.

    python avaliacao_diacriticos/por_classe.py [modelo ...]
"""
import csv
import glob
import os
import sys

MODULO = os.path.dirname(os.path.abspath(__file__))
PASTA = os.path.join(MODULO, "resultados", "sinais")
CLASSES = [("agudo", "agudo"), ("til", "til"), ("circunflexo", "circunflexo"),
           ("cedilha", "cedilha"), ("cedilha til", "cedilha e til")]
CATEGORIAS = ["correto", "a mais", "letra errada", "ausente"]


def resumir(linhas):
    saida = []
    for chave, nome in CLASSES:
        v = [l for l in linhas if l["tipo"] == "acentuada" and l["sinais"] == chave and l["categoria"]]
        if not v:
            continue
        r = {"classe": nome, "palavras": len({l["palavra"] for l in v}), "imagens": len(v)}
        for c in CATEGORIAS:
            r[c] = round(100 * sum(l["categoria"] == c for l in v) / len(v), 1)
        cima = [l for l in v if l["pede_acima"] == "True" and l["acima"] != ""]
        ced = [l for l in v if l["pede_cedilha"] == "True" and l["cedilha"] != ""]
        r["acento presente"] = round(100 * sum(l["acima"] == "1" for l in cima) / len(cima), 1) if cima else ""
        r["cedilha presente"] = round(100 * sum(l["cedilha"] == "1" for l in ced) / len(ced), 1) if ced else ""
        saida.append(r)
    return saida


def main():
    arquivos = sorted(glob.glob(os.path.join(PASTA, "sinais_*.tsv")))
    pedidos = sys.argv[1:]
    linhas_tsv, md = [], ["# Métrica final por classe de sinal", "",
                          "Palavras pedidas com acento, em porcentagem das imagens da classe. "
                          "Circunflexo e cedilha com til têm duas palavras cada.", ""]
    for arq in arquivos:
        modelo = os.path.basename(arq)[len("sinais_"):-len(".tsv")]
        if pedidos and modelo not in pedidos:
            continue
        with open(arq, encoding="utf-8") as f:
            linhas = list(csv.DictReader(f, delimiter="\t"))
        res = resumir(linhas)
        md += [f"## {modelo}", "", "| classe | palavras | correto | a mais | letra errada | ausente | acento presente | cedilha presente |",
               "|---|---|---|---|---|---|---|---|"]
        for r in res:
            linhas_tsv.append({"modelo": modelo, **r})
            fmt = lambda x: "" if x == "" else f"{x:.0f}%"
            md.append(f"| {r['classe']} | {r['palavras']} | " + " | ".join(fmt(r[c]) for c in CATEGORIAS)
                      + f" | {fmt(r['acento presente'])} | {fmt(r['cedilha presente'])} |")
        md.append("")
    with open(os.path.join(PASTA, "por_classe.tsv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_tsv[0]), delimiter="\t")
        w.writeheader(); w.writerows(linhas_tsv)
    with open(os.path.join(PASTA, "por_classe.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
