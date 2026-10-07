"""
teste_sensibilidade_diff.py -- a partir de que tamanho o E1 por diferenca ve
um acento?

O eixo por diferenca e o que vale quando houver gerado com diacritico: ele usa
o par minimo e cancela tudo que os dois gemeos tem em comum. Mas ele so tem
controle NEGATIVO validado (o IAM puro, que da 0.003) -- ainda nao existe um
gerador nosso que desenhe o diacritico, entao nao ha positivo real.

Este teste constroi o positivo sem depender de gerador nenhum: pega a imagem
ASCII que o proprio modelo gerou e PINTA um acento de tamanho conhecido em
cima dela. Como a base e literalmente a mesma imagem, a unica diferenca entre
os gemeos e o acento pintado -- que e a condicao ideal que o par minimo tenta
aproximar.

Varrendo o tamanho do acento sai uma curva de deteccao: qual escore
corresponde a um acento de tamanho nominal, e qual e o menor acento que ainda
se separa do ruido. Isso da a referencia para ler os numeros do fine-tune
quando ele existir, em vez de ter que adivinhar se 0.05 e muito ou pouco.

    python avaliacao_diacriticos/teste_sensibilidade_diff.py \\
        --dir avaliacao_diacriticos/ger_iam_puro
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402

# fracoes da largura do caractere e da altura-x. 1.0 = "tamanho nominal":
# um til que ocupa 60% da largura da letra e 18% da altura-x de espessura,
# assentado logo acima da altura-x. Medidas escolhidas a priori a partir da
# geometria tipografica, nao ajustadas para o resultado dar bom.
LARG_NOMINAL = 0.60
ESP_NOMINAL = 0.18
ESCALAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def pinta_acento(tinta, palavra, indice, onde, escala):
    """Devolve copia de `tinta` com um acento retangular pintado.

    Retangulo, nao um til desenhado: o que o E1 conta e massa de tinta na
    faixa e na coluna, entao a forma nao muda a medida -- so a area. Usar uma
    forma simples deixa a area exatamente controlada pela escala.
    """
    out = tinta.copy()
    if escala <= 0:
        return out
    mask, _ = M.mascara_de_tinta(tinta)
    cx = M.caixa_tinta(mask)
    lb = M.linha_base_e_altura_x(mask)
    if cx is None or lb is None:
        return out
    x0, x1, _, _ = cx
    y_x, y_base = lb
    n = len(M.sem_acento(palavra))
    larg_char = (x1 - x0) / n
    alt_x = max(1, y_base - y_x)

    w = max(1, int(round(LARG_NOMINAL * larg_char * escala)))
    h = max(1, int(round(ESP_NOMINAL * alt_x * escala)))
    cxc = int(x0 + (indice + 0.5) * larg_char)
    a, b = max(0, cxc - w // 2), min(out.shape[1], cxc + w // 2 + 1)
    if onde == M.ACIMA:
        topo = max(0, y_x - 2 - h)
        out[topo:topo + h, a:b] = 1.0
    else:
        topo = min(out.shape[0] - h, y_base + 2)
        out[topo:topo + h, a:b] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-pares", type=int, default=60)
    a = ap.parse_args()

    itens = [json.loads(l) for l in
             open(os.path.join(a.dir, "manifest.jsonl"), encoding="utf-8")]
    gemeo = {(d["par_id"], d["escritor"], d["semente"]): d
             for d in itens if not d["acentuada"]}

    linhas = {e: [] for e in ESCALAS}
    real = []
    n = 0
    for d in itens:
        if not d["acentuada"] or d.get("colapsada"):
            continue
        g = gemeo.get((d["par_id"], d["escritor"], d["semente"]))
        if g is None:
            continue
        if n >= a.max_pares:
            break
        n += 1
        t_asc = M.carregar_tinta(os.path.join(a.dir, g["arquivo"]))
        t_acc = M.carregar_tinta(os.path.join(a.dir, d["arquivo"]))

        # o que o modelo de fato produziu
        for r in M.e1_por_diff(t_acc, t_asc, d["palavra"]):
            if r["medivel"]:
                real.append(r["delta_rel"])

        # positivo construido: acento pintado sobre a PROPRIA imagem ASCII
        for indice, _, onde in M.diacriticos(d["palavra"]):
            for esc in ESCALAS:
                pint = pinta_acento(t_asc, d["palavra"], indice, onde, esc)
                for r in M.e1_por_diff(pint, t_asc, d["palavra"]):
                    if r["medivel"] and r["indice"] == indice:
                        linhas[esc].append(r["delta_rel"])

    print(f"{n} pares do conjunto {a.dir}\n")
    print(f"acento nominal = {LARG_NOMINAL:.0%} da largura do caractere x "
          f"{ESP_NOMINAL:.0%} da altura-x\n")
    print(f"{'escala':>8s} {'n':>5s} {'delta_rel medio':>17s} {'mediana':>10s} "
          f"{'% > 0.02':>10s}")
    for e in ESCALAS:
        v = np.array(linhas[e])
        if not len(v):
            continue
        print(f"{e:>8.2f} {len(v):5d} {v.mean():17.4f} {np.median(v):10.4f} "
              f"{100*np.mean(v > 0.02):9.1f}%")
    v = np.array(real)
    print(f"\n{'IAM real':>8s} {len(v):5d} {v.mean():17.4f} {np.median(v):10.4f} "
          f"{100*np.mean(v > 0.02):9.1f}%   <- o que o modelo produziu")
    print("\nLeitura: a linha 'escala 0.00' e o piso de ruido do proprio eixo")
    print("(gemeos identicos), e a 'IAM real' e o modelo sem treino em")
    print("portugues. Se as duas coincidirem, o modelo nao desenhou nada.")


if __name__ == "__main__":
    main()
