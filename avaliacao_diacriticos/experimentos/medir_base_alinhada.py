"""
Mede os detectores de acento na base iam_pt_alinhado, onde a resposta e
conhecida, e grava uma linha por letra em resultados/base_alinhada_letras.csv.

Cada tarefa da base tem a imagem acentuada e o par sem acento na mesma tela.
O manifesto diz em que letra esta cada sinal. As imagens passam pelo mesmo
pre-processamento do treino (64x256), que e o formato das imagens geradas.

Por letra, saem:
  e1 com fatias iguais, e1 com alinhador sem folga e com folga de meia letra,
    cada uma medida acima e abaixo da letra;
  as marcas do detector do colega (scripts/medir_marcas.py, com a letra dada
    por scripts/medir_posicao_acento.py), na acentuada e no par;
  a tinta fora do corpo (vazamento.py), na acentuada e no par.
As imagens do tipo sem_acento entram so com as medidas de imagem unica.

A analise fica em comparar_detectores.py.
"""
import argparse
import csv
import json
import os
import sys
import unicodedata

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(AQUI))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))

import e1  # noqa: E402
import e1_alinhador  # noqa: E402
import vazamento  # noqa: E402
from medir_posicao_acento import letras_das_marcas  # noqa: E402
from utils.iam_acentuado_dataset import preprocessar_iam  # noqa: E402

NOME_DA_MARCA = {"́": "agudo", "̀": "grave", "̂": "circunflexo",
                 "̃": "til", "̧": "cedilha"}
AGUDO, CEDILHA = "́", "̧"

CAMPOS = ["tipo", "tarefa", "escritor", "palavra", "posicao", "letra", "marca",
          "alinhou",
          "e1_iguais_acima", "e1_iguais_abaixo",
          "e1_alinhador_acima", "e1_alinhador_abaixo",
          "e1_alinhador_folga_acima", "e1_alinhador_folga_abaixo",
          "colega_acentuada_acima", "colega_acentuada_abaixo",
          "colega_par_acima", "colega_par_abaixo",
          "tinta_acentuada_acima", "tinta_acentuada_abaixo",
          "tinta_par_acima", "tinta_par_abaixo"]


def carregar(base, item):
    """Imagem da base no formato do treino. Devolve o cinza 0..255."""
    imagem = Image.open(os.path.join(base, item["arquivo"])).convert("RGB")
    imagem = preprocessar_iam(imagem, item["rotulo"])
    return np.asarray(imagem.convert("L"), dtype=np.float32)


def marca_de_cada_letra(palavra):
    """Nome da marca de cada letra, ou vazio."""
    marcas = []
    for simbolo in unicodedata.normalize("NFD", palavra):
        if simbolo in NOME_DA_MARCA:
            marcas[-1] = NOME_DA_MARCA[simbolo]
        elif unicodedata.category(simbolo) != "Mn":
            marcas.append("")
    return marcas


def com_marca_em_todas(letras, marca):
    """A palavra com a mesma marca em todas as letras. Serve para pedir a
    e1() a medida em cada letra, e nao so na acentuada."""
    return "".join(letra + marca for letra in letras)


def e1_em_todas_as_letras(acentuada, par, letras, colunas):
    acima = e1.e1(acentuada, par, com_marca_em_todas(letras, AGUDO), colunas)
    abaixo = e1.e1(acentuada, par, com_marca_em_todas(letras, CEDILHA), colunas)
    return acima, abaixo


def marcas_do_colega(cinza, letras, alinhador):
    """Quantas marcas o detector do colega pos em cada letra."""
    acima = [0] * len(letras)
    abaixo = [0] * len(letras)
    achadas, _ = letras_das_marcas(cinza, letras, alinhador)
    for posicao, lado in achadas:
        if lado == "acima":
            acima[posicao] += 1
        else:
            abaixo[posicao] += 1
    return acima, abaixo


def medir(base, acentuada_item, par_item, alinhador):
    """Linhas do CSV de uma tarefa. acentuada_item e None nas sem_acento."""
    letras = par_item["rotulo"]
    cinza_par = carregar(base, par_item)
    par = 1.0 - cinza_par / 255.0
    colunas = e1_alinhador.colunas_pelo_alinhador(par, letras, alinhador)
    colunas_folga = e1_alinhador.colunas_pelo_alinhador(par, letras, alinhador, 0.5)
    alinhou = colunas is not None
    vazio = [""] * len(letras)

    medidas = {campo: vazio for campo in CAMPOS[8:]}
    medidas["colega_par_acima"], medidas["colega_par_abaixo"] = \
        marcas_do_colega(cinza_par, letras, alinhador)
    if alinhou:
        tinta_par = vazamento.tinta_fora_do_corpo(par, colunas)
        medidas["tinta_par_acima"] = [m[0] for m in tinta_par]
        medidas["tinta_par_abaixo"] = [m[1] for m in tinta_par]

    if acentuada_item is None:
        palavra = letras
    else:
        palavra = acentuada_item["rotulo"]
        cinza_acentuada = carregar(base, acentuada_item)
        acentuada = 1.0 - cinza_acentuada / 255.0
        medidas["e1_iguais_acima"], medidas["e1_iguais_abaixo"] = \
            e1_em_todas_as_letras(acentuada, par, letras, None)
        medidas["colega_acentuada_acima"], medidas["colega_acentuada_abaixo"] = \
            marcas_do_colega(cinza_acentuada, letras, alinhador)
        if alinhou:
            medidas["e1_alinhador_acima"], medidas["e1_alinhador_abaixo"] = \
                e1_em_todas_as_letras(acentuada, par, letras, colunas)
            medidas["e1_alinhador_folga_acima"], medidas["e1_alinhador_folga_abaixo"] = \
                e1_em_todas_as_letras(acentuada, par, letras, colunas_folga)
            tinta_acentuada = vazamento.tinta_fora_do_corpo(acentuada, colunas)
            medidas["tinta_acentuada_acima"] = [m[0] for m in tinta_acentuada]
            medidas["tinta_acentuada_abaixo"] = [m[1] for m in tinta_acentuada]

    linhas = []
    for posicao, (letra, marca) in enumerate(zip(letras, marca_de_cada_letra(palavra))):
        linha = [par_item["tipo"], par_item["tarefa"], par_item["escritor"],
                 palavra, posicao, letra, marca, int(alinhou)]
        linha += [medidas[campo][posicao] for campo in CAMPOS[8:]]
        linhas.append(linha)
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(RAIZ, "iam_pt_alinhado"))
    ap.add_argument("--saida", default=os.path.join(
        os.path.dirname(AQUI), "resultados", "base_alinhada_letras.csv"))
    ap.add_argument("--limite", type=int, default=0, help="so as primeiras N tarefas")
    a = ap.parse_args()

    with open(os.path.join(a.base, "manifesto.jsonl"), encoding="utf-8") as f:
        manifesto = [json.loads(linha) for linha in f]
    acentuadas = {x["tarefa"]: x for x in manifesto if x["tipo"] == "acentuada"}
    outras = [x for x in manifesto if x["tipo"] != "acentuada"]
    if a.limite:
        outras = outras[:a.limite]

    alinhador = e1_alinhador.carregar_alinhador()
    with open(a.saida, "w", newline="", encoding="utf-8") as f:
        escritor_csv = csv.writer(f)
        escritor_csv.writerow(CAMPOS)
        for n, item in enumerate(outras, 1):
            escritor_csv.writerows(medir(a.base, acentuadas.get(item["tarefa"])
                                         if item["tipo"] == "par" else None,
                                         item, alinhador))
            if n % 1000 == 0:
                print(f"{n}/{len(outras)}", flush=True)
    print("gravado", a.saida)


if __name__ == "__main__":
    main()
