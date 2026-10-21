"""
Cedilha pela diferenca pareada embaixo do c, com limiar tirado do proprio
modelo avaliado.

A medida e a da e1 com alinhador (e1_alinhador.py) na faixa abaixo da linha
de base, na coluna do c: a tinta que a palavra com c cedilha tem ali e o
esqueleto do mesmo escritor e da mesma semente nao tem, dividida pela tinta do
corpo do c.

O limiar antigo vinha do modelo do IAM (resultados/limiares_e1_alinhador.json)
e deixava passar 27% das cedilhas da base. Aqui o negativo sao os c SEM
cedilha das imagens do modelo avaliado, como o c de "percussao": tambem ali a
acentuada e o esqueleto diferem so pelo ruido do gerador. O limiar e o p95
desses c.

Grava resultados/cedilha_40ep.json.

    python avaliacao_diacriticos/experimentos/cedilha.py
"""
import collections
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path.insert(0, MODULO)

import e1  # noqa: E402
import e1_alinhador  # noqa: E402
from anotacao_dupla import AVALIACAO, ler_paineis  # noqa: E402

CEDILHA = "̧"


def abrir(caminho):
    """Imagem 0..1 com tinta = 1, como a e1 quer."""
    return 1.0 - np.asarray(Image.open(caminho).convert("L"), dtype=np.float32) / 255.0


def medidas_nos_c(acentuada, esqueleto, palavra, alinhador):
    """[(posicao, pedida_com_cedilha, medida)] para cada c da palavra."""
    letras = e1.sem_acento(palavra)
    colunas = e1_alinhador.colunas_pelo_alinhador(esqueleto, letras, alinhador)
    if colunas is None:
        return []
    # a cedilha pedida em todas as letras, para medir embaixo de cada c
    em_todas = e1.e1(acentuada, esqueleto, "".join(l + CEDILHA for l in letras), colunas)
    pedidas = {posicao for posicao, acima in e1.diacriticos(palavra) if not acima}
    return [(k, k in pedidas, em_todas[k]) for k, letra in enumerate(letras) if letra == "c"]


def medir_no_modelo(alinhador):
    """Medidas nos c das 1.160 pares acentuada e esqueleto da avaliacao."""
    linhas = ler_paineis(AVALIACAO)
    esqueletos = {(l["palavra"], l["escritor"], l["semente"]): l
                  for l in linhas if l["tipo"] == "esqueleto"}
    saida = []
    for linha in linhas:
        if linha["tipo"] != "acentuada" or "c" not in e1.sem_acento(linha["palavra"]):
            continue
        gemea = esqueletos[(e1.sem_acento(linha["palavra"]), linha["escritor"], linha["semente"])]
        for posicao, pedida, medida in medidas_nos_c(abrir(linha["caminho"]), abrir(gemea["caminho"]),
                                                     linha["palavra"], alinhador):
            saida.append({"palavra": linha["palavra"], "posicao": posicao,
                          "pedida": pedida, "medida": float(medida)})
    return saida


def medidas_na_base():
    """Medida da e1 com alinhador nas cedilhas da base iam_pt_alinhado, do CSV
    gravado por medir_base_alinhada.py."""
    caminho = os.path.join(MODULO, "resultados", "base_alinhada_letras.csv")
    with open(caminho, encoding="utf-8") as f:
        return [float(l["e1_alinhador_abaixo"]) for l in csv.DictReader(f)
                if l["tipo"] == "par" and l["marca"] == "cedilha" and l["alinhou"] == "1"]


def main():
    alinhador = e1_alinhador.carregar_alinhador()
    no_modelo = medir_no_modelo(alinhador)
    negativos = [m["medida"] for m in no_modelo if not m["pedida"]]
    pedidas = [m["medida"] for m in no_modelo if m["pedida"]]
    limiar = float(np.percentile(negativos, 95))

    with open(os.path.join(MODULO, "resultados", "limiares_e1_alinhador.json"), encoding="utf-8") as f:
        limiar_antigo = json.load(f)["alinhador"]["cedilha"]["p95"]
    base = medidas_na_base()

    palavras_negativas = collections.Counter(m["palavra"] for m in no_modelo if not m["pedida"])
    # em cada palavra, quantos c sem cedilha passam do limiar
    acima_por_palavra = {p: sum(m["medida"] > limiar for m in no_modelo
                                if m["palavra"] == p and not m["pedida"])
                         for p in palavras_negativas}
    resultado = {
        "negativo": {"c_sem_cedilha": len(negativos), "palavras": dict(palavras_negativas),
                     "acima_do_limiar_por_palavra": acima_por_palavra,
                     "mediana": float(np.median(negativos)), "p95": limiar},
        "limiar_antigo_iam": limiar_antigo,
        "base_iam_pt_alinhado": {
            "cedilhas": len(base), "mediana": float(np.median(base)),
            "acima_do_limiar_novo": float(np.mean(np.array(base) > limiar)),
            "acima_do_limiar_antigo": float(np.mean(np.array(base) > limiar_antigo))},
        "modelo_40ep_cedilhas_pedidas": {
            "n": len(pedidas), "mediana": float(np.median(pedidas)),
            "acima_do_limiar_novo": float(np.mean(np.array(pedidas) > limiar))},
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    with open(os.path.join(MODULO, "resultados", "cedilha_40ep.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
