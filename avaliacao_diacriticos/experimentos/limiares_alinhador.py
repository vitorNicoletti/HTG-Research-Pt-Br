"""
Limiar por marca da e1 com alinhador, pela mesma regra da e1 atual: o p95 da
medida no controle negativo (modelo do IAM, que nao desenha acento).

Grava resultados/limiares_e1_alinhador.json. Os limiares da e1 atual ficam em
resultados/limiares_eixo_diff.json e sao recalculados aqui so para conferir.
"""
import collections
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path.insert(0, MODULO)

import e1  # noqa: E402
import e1_alinhador  # noqa: E402
import metrica  # noqa: E402

CONTROLE = os.path.join(MODULO, "amostras", "ger_iam_sonda77x4")


def main():
    with open(os.path.join(CONTROLE, "manifest.jsonl"), encoding="utf-8") as f:
        itens = [json.loads(linha) for linha in f]
    sem_acento = {(x["par_id"], x["escritor"], x["semente"]): x
                  for x in itens if not x["acentuada"]}
    alinhador = e1_alinhador.carregar_alinhador()

    ruido = {"iguais": collections.defaultdict(list),
             "alinhador": collections.defaultdict(list),
             "alinhador_folga": collections.defaultdict(list)}
    sem_alinhar = 0
    for item in itens:
        if not item["acentuada"]:
            continue
        gemea = sem_acento[(item["par_id"], item["escritor"], item["semente"])]
        acentuada = metrica.carregar_tinta(os.path.join(CONTROLE, item["arquivo"]))
        par = metrica.carregar_tinta(os.path.join(CONTROLE, gemea["arquivo"]))
        palavra = item["palavra"]
        medidas = {"iguais": e1.e1(acentuada, par, palavra),
                   "alinhador": e1_alinhador.e1_com_alinhador(acentuada, par, palavra, alinhador),
                   "alinhador_folga": e1_alinhador.e1_com_alinhador(acentuada, par, palavra, alinhador, 0.5)}
        if medidas["alinhador"] is None:
            sem_alinhar += 1
        marcas = [nome for _, nome, _ in metrica.diacriticos(palavra)]
        for versao, valores in medidas.items():
            if valores is None:
                continue
            for marca, valor in zip(marcas, valores):
                ruido[versao][marca].append(valor)

    limiares = {}
    for versao, por_marca in ruido.items():
        limiares[versao] = {marca: {"n": len(v), "p95": float(np.percentile(v, 95))}
                            for marca, v in por_marca.items()}
        print(versao, {m: round(d["p95"], 4) for m, d in limiares[versao].items()})
    print("pares que o alinhador nao alinhou:", sem_alinhar)
    limiares["controle"] = "ger_iam_sonda77x4"
    limiares["sem_alinhar"] = sem_alinhar
    saida = os.path.join(MODULO, "resultados", "limiares_e1_alinhador.json")
    with open(saida, "w", encoding="utf-8") as f:
        json.dump(limiares, f, ensure_ascii=False, indent=1)
    print("gravado", saida)


if __name__ == "__main__":
    main()
