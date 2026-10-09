"""O sinal do diacritico chega ao UNet? Medido POSICAO A POSICAO.

O UNet recebe o texto por atencao cruzada, um vetor por caractere (CANINE-C
congelado -> text_lin). O diagnostico anterior (diag_peso_acento.py) comparou
"provável" e "provavel" pela MEDIA dos 40 vetores, que mistura a letra
acentuada com as outras e com ~30 posicoes de preenchimento. Aqui a comparacao
e feita em cada posicao.

As posicoes de preenchimento (depois do fim da palavra, ~32 das 40) entram
como um grupo a parte: o UNet as consulta sem mascara.

Para ter uma regua, cada palavra acentuada e comparada com duas vizinhas:
  esqueleto -- so o acento sai            ("haverá" x "havera")
  troca     -- a letra-base vira outra    ("havera" x "havero")
Se tirar o acento mexe no vetor tanto quanto trocar a letra, o sinal chega
forte. Se mexe muito menos, chega fraco.

    python diagnostico/diag_texto_por_posicao.py [--ckpt ema_ckpt.pt]

Roda na CPU em segundos; nao usa o UNet inteiro, so a camada text_lin dele.
"""
import argparse
import os
import unicodedata

import numpy as np
import torch
from transformers import CanineModel, CanineTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALAVRAS = ["haverá", "provável", "módulo", "câmera", "fogão", "opção", "você", "café", "pão", "avó",
            "nação", "coração", "mãe", "prestação", "rústica", "pântano", "saúde", "herói", "também",
            "atrás", "português", "ônibus", "maçã", "balcão", "príncipe"]
TROCA = {"a": "o", "e": "a", "i": "u", "o": "e", "u": "i", "c": "s"}
MAX_LEN = 40


def base(c):
    return "".join(x for x in unicodedata.normalize("NFD", c) if unicodedata.category(x) != "Mn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(RAIZ, "DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"))
    a = ap.parse_args()

    tok = CanineTokenizer.from_pretrained("google/canine-c")
    enc = CanineModel.from_pretrained("google/canine-c").eval()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    lin = torch.nn.Linear(768, 320)
    lin.load_state_dict({"weight": sd["module.text_lin.weight"], "bias": sd["module.text_lin.bias"]})

    def vetores(palavras):
        t = tok(palavras, padding="max_length", truncation=True, return_tensors="pt", max_length=MAX_LEN)
        with torch.no_grad():
            h = enc(**t).last_hidden_state
            return h, lin(h)

    acc = PALAVRAS
    esq = ["".join(base(c) for c in p) for p in acc]
    # troca: no esqueleto, a letra-base do PRIMEIRO acento vira outra letra
    pos = [[i for i, c in enumerate(p) if base(c) != c] for p in acc]
    tro = [e[:ps[0]] + TROCA[e[ps[0]]] + e[ps[0] + 1:] for e, ps in zip(esq, pos)]

    resultados = {}
    for nome, k in (("canine", 0), ("text_lin", 1)):
        A, E, T = vetores(acc)[k], vetores(esq)[k], vetores(tro)[k]

        def dist(X, Y):                      # distancia relativa por posicao
            return ((X - Y).norm(dim=-1) / Y.norm(dim=-1)).numpy()

        GRUPOS = ("na letra", "vizinhas (+-1)", "outras letras", "preenchimento")

        def por_grupo(d, alvos):
            """Distancias separadas por grupo; alvos[j] = indices (na palavra) das letras mexidas."""
            g = {k: [] for k in GRUPOS}
            for j, p in enumerate(acc):
                n = len(p)
                alvo = {i + 1 for i in alvos[j]}                # +1: a posicao 0 e o [CLS]
                viz = {i for x in alvo for i in (x - 1, x + 1) if 1 <= i <= n} - alvo
                for i in range(1, n + 1):
                    g["na letra" if i in alvo else "vizinhas (+-1)" if i in viz else "outras letras"].append(d[j, i])
                g["preenchimento"] += d[j, n + 2:].tolist()
            return g

        linhas = por_grupo(dist(A, E), pos)                     # tirar o acento
        regua = por_grupo(dist(T, E), [[ps[0]] for ps in pos])  # trocar a letra
        cos = torch.nn.functional.cosine_similarity
        media = cos(A.mean(1), E.mean(1)).mean().item()
        letras = float(np.mean([cos(A[j, 1:len(p) + 1].mean(0), E[j, 1:len(p) + 1].mean(0), dim=0).item()
                                for j, p in enumerate(acc)]))
        resultados[nome] = (linhas, regua, media, letras)

    print(f"{len(acc)} palavras; ckpt {os.path.relpath(a.ckpt, RAIZ)}\n")
    for nome, (linhas, regua, media, letras) in resultados.items():
        print(f"== {nome} ==")
        print(f"  cosseno da media, acentuada x esqueleto: so as letras {letras:.3f} | os 40 vetores {media:.3f}")
        print("    (o diagnostico antigo deu 0,935 pela media; a media esconde a posicao)")
        print("  distancia relativa por posicao (mediana):")
        print(f"    {'':<16} {'tirar o acento':>15} {'trocar a letra':>15}")
        for g in linhas:
            print(f"    {g:<16} {np.median(linhas[g]):>15.3f} {np.median(regua[g]):>15.3f}   "
                  f"(n={len(linhas[g])} / {len(regua[g])})")
        for g in ("na letra", "preenchimento"):
            print(f"  => {g}: tirar o acento mexe {np.median(linhas[g]) / np.median(regua[g]):.0%} "
                  f"do que trocar a letra mexe")
        print()

    print("exemplos (acentuada / esqueleto / troca):")
    for p, e, t in list(zip(acc, esq, tro))[:6]:
        print(f"  {p} / {e} / {t}")


if __name__ == "__main__":
    main()
