"""
Taxa de palavras corretas corrigida pelo erro conhecido da metrica.

A metrica final (sinais.py) subestima o "correto": no modelo de 40 epocas as
pessoas viram 87% das palavras corretas e a metrica 75%, porque ela nao ve o
acento que encosta na letra. Como sabemos, pela anotacao, com que frequencia
ela erra para cada lado, da para estimar a taxa verdadeira de um modelo a
partir da taxa medida. E a correcao de Rogan e Gladen para um teste imperfeito:

    sens = P(metrica diz correto | pessoas dizem correto)
    espec = P(metrica diz que nao | pessoas dizem que nao)
    verdadeira = (medida + espec - 1) / (sens + espec - 1)

sens e espec saem das 135 palavras acentuadas anotadas do modelo de 40 epocas.
O intervalo vem de um bootstrap que sorteia de novo as imagens anotadas e as
palavras do modelo. A suposicao forte e que a metrica erra igual em todos os
modelos; ela so foi medida num.

Resultado (resultados/correto_corrigido.txt): nao serve. A especificidade sai
de 18 palavras nao corretas e vale 0,72; em todo modelo cuja taxa medida fica
abaixo de 1 - 0,72 a correcao vai a zero, e os intervalos vao de 0 a 50 ou
mais. Alem disso a especificidade medida no modelo de 40 epocas, onde as
palavras erradas sao casos dificeis, nao vale para modelos piores, onde o erro
e mais evidente. Fica registrado como tentativa.

Le resultados/sinais/sinais_<modelo>.tsv e grava resultados/correto_corrigido.txt.

    python avaliacao_diacriticos/experimentos/corrigir_correto.py
"""
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path[:0] = [MODULO, AQUI]

from diagnosticar_posicao import gabarito_por_letra  # noqa: E402
from sinais import Detector, categoria, sinais_pedidos, tem_pingo  # noqa: E402

RES = os.path.join(MODULO, "resultados")
N_BOOT = 2000


def pares_anotados():
    """[(pessoas dizem correto, metrica diz correto)] nas acentuadas anotadas."""
    chave = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_chave.csv"), encoding="utf-8"))}
    gab = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_gabarito.csv"), encoding="utf-8"))}
    letras = gabarito_por_letra()
    d = Detector()
    pares = []
    for img, item in chave.items():
        p = item["palavra"]
        if tem_pingo(p) or not sinais_pedidos(p):
            continue
        g = np.asarray(Image.open(os.path.join(MODULO, item["painel"])).convert("L"), dtype=np.float32)
        m = d.avaliar(g, p)["categoria"]
        h = categoria(p, letras[img], gab[img]["cedilha"] == "sim")
        if m and h:
            pares.append((h == "correto", m == "correto"))
    return np.array(pares, dtype=bool)


def corrigir(medida, sens, espec):
    den = sens + espec - 1
    return np.clip((medida + espec - 1) / den, 0, 1) if den > 0 else np.nan


def main():
    pares = pares_anotados()
    h, m = pares[:, 0], pares[:, 1]
    sens, espec = m[h].mean(), (~m[~h]).mean()
    out = [f"anotacao: {len(pares)} palavras, {h.sum()} corretas pelas pessoas; sens {sens:.2f}, espec {espec:.2f} "
           f"({(~h).sum()} palavras nao corretas)", "", "modelo | correto medido | correto corrigido | ic95"]
    rng = np.random.default_rng(0)
    linhas = []
    for arq in sorted(glob.glob(os.path.join(RES, "sinais", "sinais_*.tsv"))):
        modelo = os.path.basename(arq)[len("sinais_"):-len(".tsv")]
        with open(arq, encoding="utf-8") as f:
            v = [l for l in csv.DictReader(f, delimiter="\t") if l["tipo"] == "acentuada" and l["categoria"]]
        por_palavra = {}
        for l in v:
            por_palavra.setdefault(l["palavra"], []).append(l["categoria"] == "correto")
        ws = sorted(por_palavra)
        medida = np.mean([x for w in ws for x in por_palavra[w]])
        boot = []
        for _ in range(N_BOOT):
            a = pares[rng.integers(0, len(pares), len(pares))]
            s, e = a[:, 1][a[:, 0]].mean(), (~a[:, 1][~a[:, 0]]).mean()
            med = np.mean([x for w in rng.choice(ws, len(ws)) for x in por_palavra[w]])
            boot.append(corrigir(med, s, e))
        boot = np.array([b for b in boot if not np.isnan(b)])
        linhas.append({"modelo": modelo, "correto_medido": round(100 * medida, 1),
                       "correto_corrigido": round(100 * float(corrigir(medida, sens, espec)), 1),
                       "ic95": f"{100 * np.percentile(boot, 2.5):.0f} a {100 * np.percentile(boot, 97.5):.0f}"})
        out.append(f"{modelo} | {linhas[-1]['correto_medido']} | {linhas[-1]['correto_corrigido']} | {linhas[-1]['ic95']}")
    texto = "\n".join(out) + "\n"
    open(os.path.join(RES, "correto_corrigido.txt"), "w", encoding="utf-8").write(texto)
    print(texto)


if __name__ == "__main__":
    main()
