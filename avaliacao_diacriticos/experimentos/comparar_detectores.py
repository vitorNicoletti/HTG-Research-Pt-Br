"""
Compara tres detectores de acento na base iam_pt_alinhado, a partir do CSV
gravado por medir_base_alinhada.py.

  e1 atual       colunas por fatias iguais (e1.py)
  e1 alinhador   colunas do alinhador, sem folga e com folga de meia letra
  colega         marcas soltas de scripts/medir_marcas.py, com a letra dada por
                 scripts/medir_posicao_acento.py

Os limiares da e1 sao o p95 do controle negativo, por marca
(resultados/limiares_e1_alinhador.json). Nenhum limiar e escolhido nesta base.

Tres perguntas:
  presenca   na letra que tem o sinal, o detector acusa?
  posicao    numa letra da mesma palavra que nao tem sinal, o detector acusa?
  vazamento  numa imagem sem acento nenhum, o detector acusa?

As linhas "acima de zero" da presenca nao sao uma regra de decisao. Dizem em
quantas letras com sinal a medida da e1 foi positiva, sem limiar nenhum.
"""
import collections
import csv
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path.insert(0, MODULO)
sys.path.insert(0, AQUI)

from calibrar import auc  # noqa: E402
from vazamento import LETRAS_SEM_HASTE, LETRAS_SEM_PERNA  # noqa: E402

MARCAS = ["agudo", "til", "circunflexo", "cedilha", "grave"]
VERSOES_E1 = {"e1 atual": "iguais", "e1 alinhador": "alinhador",
              "e1 alinhador com folga": "alinhador_folga"}


def numero(texto):
    return float(texto) if texto != "" else None


def ler(caminho):
    """Agrupa as linhas do CSV por imagem: {(tipo, tarefa): [linha por letra]}."""
    imagens = collections.defaultdict(list)
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            for campo in linha:
                if campo.startswith(("e1_", "colega_", "tinta_")):
                    linha[campo] = numero(linha[campo])
            linha["posicao"] = int(linha["posicao"])
            imagens[(linha["tipo"], linha["tarefa"])].append(linha)
    return imagens


def lado_da(marca):
    return "abaixo" if marca == "cedilha" else "acima"


def taxa(observacoes):
    """Taxa em %, com IC95 reamostrando palavras inteiras.

    observacoes: lista de (acusou, palavra)."""
    if not observacoes:
        return "sem dados"
    soma = collections.Counter()
    quantas = collections.Counter()
    for acusou, palavra in observacoes:
        soma[palavra] += acusou
        quantas[palavra] += 1
    soma = np.array(list(soma.values()), dtype=float)
    quantas = np.array(list(quantas.values()), dtype=float)
    sorteio = np.random.default_rng(0).integers(0, len(soma), (2000, len(soma)))
    medias = 100 * soma[sorteio].sum(axis=1) / quantas[sorteio].sum(axis=1)
    baixo, alto = np.percentile(medias, [2.5, 97.5])
    return (f"{100 * soma.sum() / quantas.sum():5.1f}% [{baixo:5.1f}, {alto:5.1f}]"
            f" n={len(observacoes)}")


# ---------------------------------------------------------------- presenca

def presenca(pares, limiares, so_sem_ij):
    """Por marca e detector, a fracao das letras com sinal em que ele acusa."""
    acertos = collections.defaultdict(list)
    for letras in pares:
        if letras[0]["alinhou"] != "1":
            continue
        if so_sem_ij and set("ij") & {l["letra"] for l in letras}:
            continue
        for letra in letras:
            marca = letra["marca"]
            if not marca:
                continue
            lado = lado_da(marca)
            testes = {nome: letra[f"e1_{versao}_{lado}"] > limiares[versao][marca]["p95"]
                      for nome, versao in VERSOES_E1.items()}
            for nome, versao in VERSOES_E1.items():
                testes[nome + ", acima de zero"] = letra[f"e1_{versao}_{lado}"] > 0
            testes["colega na letra"] = letra[f"colega_acentuada_{lado}"] > 0
            testes["colega em qualquer letra"] = \
                sum(l[f"colega_acentuada_{lado}"] for l in letras) > 0
            for nome, acusa in testes.items():
                acertos[(marca, nome)].append((acusa, letra["palavra"]))
    return acertos


# ----------------------------------------------------------------- posicao

def posicao(pares, limiares):
    """Nas letras SEM sinal de uma palavra acentuada, quantas o detector acusa.

    O teste em cada letra usa o lado e o limiar do sinal que a palavra tem.
    Separa a letra vizinha da acentuada das mais distantes. Tambem conta em
    quantos sinais a maior medida da e1 cai na letra certa.
    """
    falsos = collections.defaultdict(list)
    maior_na_certa = collections.defaultdict(list)
    for letras in pares:
        if letras[0]["alinhou"] != "1":
            continue
        for sinal in letras:
            marca = sinal["marca"]
            if not marca:
                continue
            lado = lado_da(marca)
            for nome, versao in VERSOES_E1.items():
                medidas = [l[f"e1_{versao}_{lado}"] for l in letras]
                maior_na_certa[(marca, nome)].append(
                    int(np.argmax(medidas)) == sinal["posicao"] and max(medidas) > 0)
            for letra in letras:
                if letra["marca"]:
                    continue
                distancia = "vizinha" if abs(letra["posicao"] - sinal["posicao"]) == 1 else "distante"
                testes = {nome: letra[f"e1_{versao}_{lado}"] > limiares[versao][marca]["p95"]
                          for nome, versao in VERSOES_E1.items()}
                testes["colega na acentuada"] = letra[f"colega_acentuada_{lado}"] > 0
                testes["colega no par"] = letra[f"colega_par_{lado}"] > 0
                for nome, acusa in testes.items():
                    falsos[(marca, distancia, nome)].append((acusa, letra["palavra"]))
    return falsos, maior_na_certa


# --------------------------------------------------------------- vazamento

def tinta_da_imagem(letras, qual):
    """Maior tinta fora do corpo entre as letras que valem (vazamento.py)."""
    maior = 0.0
    for letra in letras:
        if letra["letra"] in LETRAS_SEM_HASTE:
            maior = max(maior, letra[f"tinta_{qual}_acima"])
        if letra["letra"] in LETRAS_SEM_PERNA:
            maior = max(maior, letra[f"tinta_{qual}_abaixo"])
    return maior


def colega_na_imagem(letras, qual):
    return sum(l[f"colega_{qual}_acima"] + l[f"colega_{qual}_abaixo"] for l in letras) > 0


def marca_fora_do_pingo(letras, qual):
    """O detector do colega, descontando o pingo: sobre um i ou um j, a
    primeira marca acima e o pingo e nao conta."""
    for letra in letras:
        acima = letra[f"colega_{qual}_acima"]
        if letra["letra"] in "ij":
            acima = max(0, acima - 1)
        if acima + letra[f"colega_{qual}_abaixo"] > 0:
            return True
    return False


def vazamento(imagens, so_sem_ij):
    """Uma ficha por imagem. Negativas sao os pares e as sem_acento, positivas
    as acentuadas."""
    negativas, positivas = [], []
    for (tipo, _), letras in imagens.items():
        if letras[0]["alinhou"] != "1":
            continue
        if so_sem_ij and set("ij") & {l["letra"] for l in letras}:
            continue
        negativas.append({"tipo": tipo, "escritor": letras[0]["escritor"],
                          "colega": colega_na_imagem(letras, "par"),
                          "sem_pingo": marca_fora_do_pingo(letras, "par"),
                          "tinta": tinta_da_imagem(letras, "par")})
        if tipo == "par":
            positivas.append({"escritor": letras[0]["escritor"],
                              "marcas": [l["marca"] for l in letras if l["marca"]],
                              "colega": colega_na_imagem(letras, "acentuada"),
                              "sem_pingo": marca_fora_do_pingo(letras, "acentuada"),
                              "tinta": tinta_da_imagem(letras, "acentuada")})
    return negativas, positivas


def porcento(fichas, campo, limiar=None):
    if not fichas:
        return "  sem dados"
    if limiar is None:
        return f"{100 * np.mean([f[campo] for f in fichas]):5.1f}%"
    return f"{100 * np.mean([f[campo] > limiar for f in fichas]):5.1f}%"


def relatar_vazamento(negativas, positivas, titulo):
    """O detector do colega da sim ou nao. A tinta por letra da um numero, e o
    limiar dela sai de metade dos escritores (p95 das negativas) e e aplicado
    na outra metade. Todas as taxas abaixo sao dessa outra metade."""
    print(f"\n{titulo}")
    print(f"  tinta por letra, AUC em todas as imagens "
          f"{auc([f['tinta'] for f in negativas], [f['tinta'] for f in positivas]):.3f}")
    escritores = sorted({f["escritor"] for f in negativas + positivas})
    metade_a = set(escritores[::2])
    limiar = np.percentile([f["tinta"] for f in negativas if f["escritor"] in metade_a], 95)
    negativas = [f for f in negativas if f["escritor"] not in metade_a]
    positivas = [f for f in positivas if f["escritor"] not in metade_a]
    print(f"  limiar da tinta {limiar:.3f}; na outra metade dos escritores, "
          f"{len(negativas)} negativas e {len(positivas)} positivas")

    print(f"  {'':34s} {'colega':>7s} {'colega sem pingo':>17s} {'tinta por letra':>16s}")
    grupos = [("falso alarme no par", [f for f in negativas if f["tipo"] == "par"]),
              ("falso alarme na sem_acento", [f for f in negativas if f["tipo"] == "sem_acento"]),
              ("deteccao nas acentuadas", positivas)]
    for marca in MARCAS:
        grupos.append((f"deteccao, so {marca}", [f for f in positivas if f["marcas"] == [marca]]))
    for nome, fichas in grupos:
        print(f"  {nome:34s} {porcento(fichas, 'colega'):>7s} {porcento(fichas, 'sem_pingo'):>17s} "
              f"{porcento(fichas, 'tinta', limiar):>16s}   n={len(fichas)}")


# ------------------------------------------------------------------- saida

def main():
    resultados = os.path.join(MODULO, "resultados")
    imagens = ler(os.path.join(resultados, "base_alinhada_letras.csv"))
    with open(os.path.join(resultados, "limiares_e1_alinhador.json"), encoding="utf-8") as f:
        limiares = json.load(f)
    pares = [letras for (tipo, _), letras in imagens.items() if tipo == "par"]
    sem_alinhar = sum(1 for letras in imagens.values() if letras[0]["alinhou"] != "1")
    print(f"imagens {len(imagens)}, pares {len(pares)}, sem alinhamento {sem_alinhar}")

    detectores = list(VERSOES_E1) + ["colega na letra", "colega em qualquer letra"] + \
        [nome + ", acima de zero" for nome in VERSOES_E1]
    for so_sem_ij in (False, True):
        acertos = presenca(pares, limiares, so_sem_ij)
        print("\nPRESENCA na letra com sinal" + (", so palavras sem i nem j" if so_sem_ij else ""))
        for marca in MARCAS:
            for nome in detectores:
                if (marca, nome) in acertos:
                    print(f"  {marca:12s} {nome:38s} {taxa(acertos[(marca, nome)])}")

    falsos, maior_na_certa = posicao(pares, limiares)
    print("\nPOSICAO: letra sem sinal acusada, em palavra acentuada")
    for marca in MARCAS:
        for distancia in ("vizinha", "distante"):
            for nome in list(VERSOES_E1) + ["colega na acentuada", "colega no par"]:
                chave = (marca, distancia, nome)
                if chave in falsos:
                    print(f"  {marca:12s} {distancia:9s} {nome:26s} {taxa(falsos[chave])}")
    print("\nPOSICAO: a maior medida da e1 cai na letra do sinal")
    for marca in MARCAS:
        for nome in VERSOES_E1:
            v = maior_na_certa[(marca, nome)]
            if v:
                print(f"  {marca:12s} {nome:26s} {100 * np.mean(v):5.1f}%  n={len(v)}")

    relatar_vazamento(*vazamento(imagens, False), "VAZAMENTO, todas as palavras")
    relatar_vazamento(*vazamento(imagens, True), "VAZAMENTO, so palavras sem i nem j")


if __name__ == "__main__":
    main()
