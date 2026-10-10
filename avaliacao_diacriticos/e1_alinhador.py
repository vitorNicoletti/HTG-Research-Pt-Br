"""
E1 com a coluna de cada letra vinda do alinhador.

A e1.py divide a largura da palavra em fatias iguais, uma por letra. Isso erra
quando as letras tem larguras diferentes. Aqui um reconhecedor treinado no IAM
(acentos_sinteticos/alinhamento.py, pesos em modelos/alinhador_iam.pt) diz onde
cada letra esta, e a medida e a mesma da e1.py com essas colunas.

O alinhador le a imagem SEM acento, com a palavra sem acento. Ele nao decide se
ha acento; so localiza as letras.

    alinhador = carregar_alinhador()
    medidas = e1_com_alinhador(acentuada, sem_acento, "acao", alinhador)
"""
import os
import sys

import numpy as np

import e1

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from acentos_sinteticos import alinhamento, geometria  # noqa: E402

PESOS = os.path.join(RAIZ, "modelos", "alinhador_iam.pt")
MARGEM = 3   # pixels de papel em volta da tinta no recorte dado ao alinhador


def carregar_alinhador(pesos=PESOS):
    return alinhamento.Alinhador(pesos, "cpu")


def colunas_pelo_alinhador(imagem_sem_acento, palavra, alinhador, folga=0.0):
    """Coluna (inicio, fim) de cada letra, em pixels da imagem.

    A folga alarga cada coluna para os dois lados, em fracao da largura media
    de uma letra. Devolve None quando o alinhador nao consegue alinhar.
    """
    letras = e1.sem_acento(palavra)
    cinza = (1.0 - imagem_sem_acento) * 255.0    # o alinhador quer tinta escura

    # O alinhador foi treinado em recortes justos na palavra.
    mascara = geometria.mascara_tinta(cinza)
    if mascara.sum() < 20:
        return None
    linhas, colunas = np.where(mascara)
    esquerda = max(0, colunas.min() - MARGEM)
    recorte = cinza[max(0, linhas.min() - MARGEM):linhas.max() + 1 + MARGEM,
                    esquerda:colunas.max() + 1 + MARGEM]

    geo = geometria.analisar(recorte)
    if geo is None:
        return None
    resultado = alinhador.alinhar(recorte, letras)
    if resultado is None:
        return None

    # Fronteira entre duas letras no meio dos disparos do alinhador, movida
    # para a coluna com menos tinta ali perto.
    fatias = alinhamento.fatias_do_alinhamento(resultado, geo)
    fatias = geometria.ajustar_aos_vales(geo, fatias)

    largura_media = (fatias[-1][1] - fatias[0][0]) / len(fatias)
    sobra = folga * largura_media
    colunas_das_letras = []
    for inicio, fim in fatias:
        inicio = int(round(esquerda + inicio - sobra))
        fim = int(round(esquerda + fim + sobra))
        colunas_das_letras.append((max(0, inicio), max(fim, inicio + 1)))
    return colunas_das_letras


def e1_com_alinhador(imagem_acentuada, imagem_sem_acento, palavra, alinhador,
                     folga=0.0):
    """Igual a e1.e1, com as colunas do alinhador. None se nao alinhar."""
    colunas = colunas_pelo_alinhador(imagem_sem_acento, palavra, alinhador, folga)
    if colunas is None:
        return None
    return e1.e1(imagem_acentuada, imagem_sem_acento, palavra, colunas=colunas)
