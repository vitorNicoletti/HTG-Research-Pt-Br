"""Vocabulario portugues e a particao treino/val/teste por grupo de palavra.

E o PRIMEIRO passo da base portuguesa e tem de ser congelado (commitado)
antes de qualquer imagem ser gerada: todos os scripts seguintes leem
vocabulario_pt/palavras.tsv e recusam palavra fora do split certo.

Fonte: FrequencyWords, lista pt_br_50k de 2018 (OpenSubtitles 2018),
https://github.com/hermitdave/FrequencyWords -- conteudo CC-BY-SA 4.0 (ver
vocabulario_pt/LEIAME.md).

Particao, para nao vazar palavra do teste para o treino:
  - grupo = esqueleto sem acento com flexao simples dobrada: esta/esta(acentuada),
    e/e(acentuada), nacao/nacoes, coracao/coracoes caem no MESMO grupo, e o
    grupo inteiro vai para um split so;
  - grupos FORCADOS no teste: toda lista de palavras que o repositorio ja usa
    para avaliar (sonda, pares da metrica, palavras de amostra dos
    experimentos e de gerar_amostras.py, medicao de marcas) e as palavras
    acentuadas do teste do BRESSAY; forcados na validacao: as acentuadas da
    validacao do BRESSAY;
  - o resto vai por hash estavel do grupo (sha1), nao pela ordem da lista:
    acrescentar palavras depois nunca move uma palavra de split.

    python scripts/preparar_vocabulario_pt.py
"""

import ast
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
import urllib.request
from collections import Counter

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos.vocabulario import esqueleto, grupo  # noqa: E402

URL = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/pt_br/pt_br_50k.txt"
PASTA = os.path.join(RAIZ, "vocabulario_pt")
FONTE = os.path.join(PASTA, "fonte", "pt_br_50k.txt")

LETRAS = re.compile(r"[a-zçáéíóúâêôãõà]+")
TAM_MIN, TAM_MAX = 2, 12
FRACAO_VAL, FRACAO_TESTE = 0.05, 0.10

# As 17 palavras da medicao de marcas (LOG.md 2026-10-04): usadas para avaliar
MEDICAO_MARCAS = ["the", "and", "that", "have", "nacao", "coracao", "pao", "avo", "voce",
                  "nação", "coração", "pão", "mãe", "avó", "você", "café", "até"]


def split_por_hash(g):
    h = int(hashlib.sha1(g.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < FRACAO_TESTE:
        return "teste"
    if h < FRACAO_TESTE + FRACAO_VAL:
        return "val"
    return "treino"


def palavras_de_avaliacao():
    """{palavra: motivo} de todas as listas que o repositorio usa para avaliar."""
    saida = {}

    def add(ps, motivo):
        for p in ps:
            p = unicodedata.normalize("NFC", p.strip()).lower()
            if LETRAS.fullmatch(p):   # os TSV da metrica tem colunas numericas
                saida.setdefault(p, motivo)

    from comum import palavras as cp
    add(cp.todas_palavras(), "sonda (comum/palavras.py)")
    add([w for par in cp.PARES_MINIMOS for w in par], "sonda (comum/palavras.py)")
    for arq in ("pares_gate.tsv", "pares_sonda30.tsv"):
        with open(os.path.join(RAIZ, "avaliacao_diacriticos", arq), encoding="utf-8") as f:
            for l in f:
                if l.startswith("#"):
                    continue
                add(l.rstrip("\n").split("\t"), f"metrica ({arq})")
    arv = ast.parse(open(os.path.join(RAIZ, "scripts", "gerar_amostras.py"), encoding="utf-8").read())
    for no in ast.walk(arv):
        if isinstance(no, ast.Assign) and any(getattr(t, "id", "") == "PALAVRAS_PADRAO" for t in no.targets):
            add(ast.literal_eval(no.value), "gerar_amostras.py PALAVRAS_PADRAO")
    for arq in glob.glob(os.path.join(RAIZ, "experimentos", "*.json")):
        with open(arq, encoding="utf-8") as f:
            add(json.load(f)["amostras"]["palavras"], f"amostras de {os.path.basename(arq)}")
    add(MEDICAO_MARCAS, "medicao de marcas (LOG 2026-10-04)")
    return saida


def acentuadas_bressay(split):
    ps = set()
    with open(os.path.join(RAIZ, "bressay_split", "splits", f"{split}.tsv"), encoding="utf-8") as f:
        for l in f:
            c = l.rstrip("\n").split("\t")
            if len(c) >= 3:
                p = unicodedata.normalize("NFC", c[2].strip()).lower()
                if LETRAS.fullmatch(p) and esqueleto(p) != p:
                    ps.add(p)
    return ps


def main():
    if not os.path.isfile(FONTE):
        os.makedirs(os.path.dirname(FONTE), exist_ok=True)
        print("baixando", URL)
        urllib.request.urlretrieve(URL, FONTE)
    bruto = open(FONTE, "rb").read()
    sha = hashlib.sha256(bruto).hexdigest()

    freq = {}
    for l in bruto.decode("utf-8").splitlines():
        partes = l.split(" ")
        if len(partes) != 2:
            continue
        p = unicodedata.normalize("NFC", partes[0]).lower()
        if LETRAS.fullmatch(p) and TAM_MIN <= len(p) <= TAM_MAX and "ü" not in p:
            freq[p] = freq.get(p, 0) + int(partes[1])

    avaliacao = palavras_de_avaliacao()
    teste_bressay = acentuadas_bressay("test")
    val_bressay = acentuadas_bressay("val")

    forcado = {}          # grupo -> (split, motivo)
    for p, motivo in avaliacao.items():
        forcado[grupo(p)] = ("teste", motivo)
    for p in teste_bressay:
        forcado.setdefault(grupo(p), ("teste", "acentuada do teste do BRESSAY"))
    for p in val_bressay:
        forcado.setdefault(grupo(p), ("val", "acentuada da validacao do BRESSAY"))

    # as palavras de avaliacao entram no arquivo mesmo fora da lista de
    # frequencia: elas tem de estar registradas como teste
    todas = dict(freq)
    for p in avaliacao:
        if LETRAS.fullmatch(p):
            todas.setdefault(p, 0)

    linhas = []
    for p in sorted(todas, key=lambda w: (-todas[w], w)):
        g = grupo(p)
        sp, motivo = forcado.get(g, (split_por_hash(g), ""))
        linhas.append((p, esqueleto(p), g, sp, todas[p], int(esqueleto(p) != p), motivo))

    with open(os.path.join(PASTA, "palavras.tsv"), "w", encoding="utf-8") as f:
        f.write("palavra\tesqueleto\tgrupo\tsplit\tfrequencia\tacentuada\tforcada\n")
        for l in linhas:
            f.write("\t".join(map(str, l)) + "\n")

    # conferencias: nenhum grupo em dois splits; avaliacao toda no teste
    split_do_grupo = {}
    for p, _, g, sp, *_ in linhas:
        assert split_do_grupo.setdefault(g, sp) == sp, f"grupo {g} em dois splits"
    for p in avaliacao:
        assert split_do_grupo.get(grupo(p)) == "teste", f"palavra de avaliacao fora do teste: {p}"

    cont = Counter((l[3], l[5]) for l in linhas)
    resumo = {
        "fonte": URL, "sha256_fonte": sha, "licenca_fonte": "CC-BY-SA 4.0",
        "filtro": {"letras": LETRAS.pattern, "tamanho": [TAM_MIN, TAM_MAX]},
        "fracoes_hash": {"val": FRACAO_VAL, "teste": FRACAO_TESTE},
        "palavras": len(linhas), "grupos": len(split_do_grupo),
        "por_split": {sp: {"acentuadas": cont[(sp, 1)], "sem_acento": cont[(sp, 0)]}
                      for sp in ("treino", "val", "teste")},
        "forcadas": {"avaliacao": len(avaliacao), "teste_bressay": len(teste_bressay),
                     "val_bressay": len(val_bressay), "grupos_forcados": len(forcado)},
    }
    with open(os.path.join(PASTA, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
