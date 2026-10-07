"""A linha pautada do papel quebra a estimativa de geometria do E1?

Contexto: 73% dos recortes de palavra do BRESSAY tem a linha pautada do papel
atravessando a imagem de ponta a ponta (medido em 300 amostras do split de
treino por diagnostico/inspecionar_dados.py). Nos recortes de teste ja
preparados por preparar_reais.py a proporcao e 87%.

A hipotese: linha_base_e_altura_x() estima o corpo da palavra como as linhas
com pelo menos METADE da tinta da linha mais cheia. A linha pautada atravessa a
imagem inteira, entao ela E a linha mais cheia -- o limiar sobe, so as fileiras
vizinhas a ela sobrevivem, e o "corpo" encolhe para uma faixa fina em volta da
pauta. Como consequencia:

  * y_x (topo da altura-x) desce quase ate a base, e a faixa ACIMA passa a
    engolir o corpo inteiro da letra em vez de so a zona do diacritico;
  * `corpo`, que e o denominador de massa_rel, encolhe.

Os dois efeitos inflam massa_rel na mesma direcao.

Isto importa duas vezes. Primeiro porque o controle positivo (recortes reais,
E1 = 0.861) e a ancora da calibracao do limiar. Segundo, e mais grave, porque o
modelo ajustado no BRESSAY aprende a desenhar a pauta -- ela aparece nas
amostras do treino de 40 epocas --, entao o mesmo artefato vai contaminar a
medida no material gerado.

    python avaliacao_diacriticos/teste_linha_pautada.py --dir avaliacao_diacriticos/reais_test

LEITURA
    corpo estimado parecido com e sem pauta -> a geometria aguenta
    corpo encolhendo com a pauta            -> precisa remover a pauta antes
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402


def tem_pauta(mask, frac=0.85):
    """Alguma fileira preenchida em mais de `frac` da largura da tinta?"""
    cx = M.caixa_tinta(mask)
    if cx is None:
        return False
    x0, x1, _, _ = cx
    return bool(mask[:, x0:x1].mean(axis=1).max() > frac)


def sem_pauta(mask, frac=0.85):
    """Mesma mascara com as fileiras da pauta zeradas."""
    cx = M.caixa_tinta(mask)
    if cx is None:
        return mask
    x0, x1, _, _ = cx
    m = mask.copy()
    m[mask[:, x0:x1].mean(axis=1) > frac, :] = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()

    geo = {True: [], False: []}
    escores = collections.defaultdict(lambda: {True: [], False: []})
    n_img = 0

    for linha in open(os.path.join(a.dir, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(linha)
        if not r.get("acentuada") or r.get("colapsada"):
            continue
        t = M.carregar_tinta(os.path.join(a.dir, r["arquivo"]))
        mask = M.binariza(t)
        if not mask.any():
            continue
        lb = M.linha_base_e_altura_x(mask)
        if lb is None:
            continue
        n_img += 1
        pauta = tem_pauta(mask)
        y_x, y_base = lb
        _, _, y0, y1 = M.caixa_tinta(mask)
        geo[pauta].append((y_base - y_x, y1 - y0))
        for d in M.e1_por_faixa(t, r["palavra"]):
            if d["medivel"]:
                escores[d["marca"]][pauta].append(d["massa_rel"])

    print(f"{n_img} recortes acentuados | com pauta: "
          f"{100*len(geo[True])/max(n_img,1):.0f}%\n")

    print(f"{'':<20}{'com pauta':>14}{'sem pauta':>14}")
    for rotulo, i in (("corpo estimado", 0), ("tinta total", 1)):
        v = [f"{np.median([g[i] for g in geo[p]]):.0f} px" if geo[p] else "--"
             for p in (True, False)]
        print(f"{rotulo:<20}{v[0]:>14}{v[1]:>14}")

    print(f"\n{'marca':<16}{'massa_rel c/ pauta':>20}{'s/ pauta':>12}")
    for marca in sorted(escores):
        e = escores[marca]
        v = [f"{np.median(e[p]):.3f} (n={len(e[p])})" if e[p] else "--"
             for p in (True, False)]
        print(f"{marca:<16}{v[0]:>20}{v[1]:>12}")

    print("\nSe o corpo encolhe com a pauta, a faixa ACIMA engole a letra e")
    print("massa_rel infla. A funcao sem_pauta() deste arquivo mostra a")
    print("correcao minima: zerar as fileiras da pauta antes da geometria.")


if __name__ == "__main__":
    main()
