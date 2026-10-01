"""
montar_pares.py -- escolhe as palavras da sonda de pares minimos.

A lista precisa ser reprodutivel e precisa ter MUITAS PALAVRAS DISTINTAS por
marca, nao muitas repeticoes das mesmas. O intervalo de confianca do E1
reamostra palavras (ver ic_bootstrap em avaliar.py), entao o que sustenta o
numero e o numero de tipos, nao o de imagens. Gerar 18 palavras com 2 estilos e
2 sementes custa menos imagens do que 6 palavras com 4 estilos e 3 sementes, e
da um IC honesto.

Criterio de escolha, nesta ordem:
  1. uma marca so na palavra, para o escore nao misturar duas marcas;
  2. gemeo ASCII que tambem seja palavra real do corpus, quando existir, para
     nao confundir efeito de diacritico com efeito de nao-palavra;
  3. mais frequente no corpus.

Tetos reais por marca, medidos (tipos distintos, minusculas, so alfabetico):

    len>=4:  agudo 865 · circunflexo 209 · til 145 · cedilha 113 · grave 5
    len>=2:  grave sobe so para 6

Grave e quase uma forma lexical unica. No corpus inteiro sao ~800 ocorrencias,
das quais 786 sao "as" com crase. Isso basta para medir a IMAGEM da marca em
recorte real, mas nao ajuda o IC do E1, onde grave continua valendo como uma
palavra. O teto dele e do dado, nao da escolha.

    python avaliacao_diacriticos/montar_pares.py --por-marca 18 \\
        --out avaliacao_diacriticos/pares_sonda.tsv
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARCAS = ["til", "agudo", "cedilha", "circunflexo", "grave"]


def vocabulario():
    """{palavra: frequencia} somando os tres splits."""
    tok = collections.Counter()
    for sp in ("train", "val", "test"):
        caminho = os.path.join(RAIZ, "bressay_split", "splits", f"{sp}.tsv")
        for linha in open(caminho, encoding="utf-8"):
            p = linha.rstrip("\n").split("\t")
            if len(p) >= 3:
                tok[p[2]] += 1
    return tok


def candidatos(tok, min_len):
    """Palavras com exatamente uma marca, agrupadas por marca."""
    por_marca = collections.defaultdict(list)
    for w, f in tok.items():
        if not w.isalpha() or w != w.lower() or len(w) < min_len:
            continue
        marcas = {n for _, n, _ in M.diacriticos(w)}
        if len(marcas) != 1:
            continue
        asc = M.sem_acento(w)
        por_marca[next(iter(marcas))].append(
            {"acc": w, "asc": asc, "freq": f, "gemeo_real": asc in tok})
    return por_marca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-marca", type=int, default=18)
    ap.add_argument("--min-len", type=int, default=4,
                    help="grave so alcanca 6 tipos com --min-len 2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = vocabulario()
    por_marca = candidatos(tok, a.min_len)

    escolhidas = []
    print(f"{'marca':14s}{'disponiveis':>12s}{'escolhidas':>12s}{'com gemeo real':>16s}")
    for m in MARCAS:
        v = sorted(por_marca.get(m, []),
                   key=lambda c: (not c["gemeo_real"], -c["freq"]))[:a.por_marca]
        escolhidas += [(c, m) for c in v]
        print(f"{m:14s}{len(por_marca.get(m, [])):12d}{len(v):12d}"
              f"{sum(c['gemeo_real'] for c in v):16d}")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# acentuada\tascii\tmarca\tfreq_corpus\tgemeo_e_palavra_real\n")
        for c, m in escolhidas:
            f.write(f"{c['acc']}\t{c['asc']}\t{m}\t{c['freq']}\t{int(c['gemeo_real'])}\n")

    n = len(escolhidas)
    print(f"\n{n} pares -> {a.out}")
    print(f"com 2 estilos e 2 sementes: {n * 2 * 2 * 2} imagens")


if __name__ == "__main__":
    main()
