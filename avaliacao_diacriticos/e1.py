"""
Eixo 1 da metrica de diacriticos: o acento foi desenhado?

A medida usa um par de imagens da mesma palavra, uma pedida com acento e outra
sem ("acao" com cedilha e til, e "acao"), geradas com o mesmo escritor e a
mesma semente. A unica diferenca entre as duas e o texto pedido. Entao a tinta
que a imagem acentuada tem a mais, no lugar onde o acento deveria estar, e o
acento.

Passos, na ordem em que a funcao e1() os executa:

  1. separar tinta de papel nas duas imagens (tinta);
  2. por a imagem acentuada no mesmo lugar da outra (alinha);
  3. achar onde fica o corpo das letras (corpo_das_letras);
  4. achar as colunas de cada letra (colunas_por_fatias_iguais);
  5. para cada acento, contar a tinta a mais no retangulo onde ele deveria
     estar, e dividir pelo tamanho da letra.

As imagens entram como matrizes de numeros entre 0 e 1, em que 1 e tinta.
"""
import unicodedata

import cv2
import numpy as np

# Marcas na forma "solta" do Unicode. Em NFD, a letra com til vira dois
# simbolos: a letra e o til sozinho.
MARCAS_ACIMA = set("́̀̂̃")   # agudo, grave, circunflexo, til
MARCAS_ABAIXO = set("̧")                    # cedilha

# Linha pautada do caderno: ocupa quase toda a largura da palavra e e fina.
PAUTA_PREENCHIMENTO = 0.9    # fracao da largura que a linha precisa cobrir
PAUTA_ESPESSURA_MAX = 5      # em pixels; mais grosso que isso e letra

# Ate onde procurar ao alinhar as duas imagens, em pixels.
MAX_DESLOCAMENTO_X = 20
MAX_DESLOCAMENTO_Y = 8


# ----------------------------------------------------------------- texto

def sem_acento(palavra):
    """A palavra sem as marcas. Troca a letra acentuada pela letra simples."""
    letras = []
    for simbolo in unicodedata.normalize("NFD", palavra):
        e_marca = unicodedata.category(simbolo) == "Mn"
        if not e_marca:
            letras.append(simbolo)
    return "".join(letras)


def diacriticos(palavra):
    """Onde estao os acentos da palavra.

    Devolve uma lista com um par por acento: a posicao da letra (contando a
    partir de zero) e se a marca fica acima dela. Para a palavra "acao" com
    cedilha na letra 1 e til na letra 2, devolve [(1, False), (2, True)].
    """
    acentos = []
    posicao_da_letra = -1
    for simbolo in unicodedata.normalize("NFD", palavra):
        if simbolo in MARCAS_ACIMA:
            acentos.append((posicao_da_letra, True))
        elif simbolo in MARCAS_ABAIXO:
            acentos.append((posicao_da_letra, False))
        elif unicodedata.category(simbolo) != "Mn":
            posicao_da_letra += 1
    return acentos


# ---------------------------------------------------------------- imagem

def tinta(imagem):
    """Marca onde ha tinta. Devolve uma matriz de verdadeiro e falso."""
    # Imagem de uma cor so nao tem tinta. Sem este teste, o limiar automatico
    # abaixo dividiria o ruido do papel em "tinta" e "papel".
    if imagem.max() - imagem.min() < 0.10:
        return np.zeros(imagem.shape, bool)

    # Otsu escolhe sozinho o tom de cinza que separa tinta de papel.
    em_bytes = (imagem * 255).astype(np.uint8)
    _, binaria = cv2.threshold(em_bytes, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = binaria.astype(bool)
    apagar_pauta(mascara)
    return mascara


def apagar_pauta(mascara):
    """Apaga a linha pautada do caderno, se houver. Altera a mascara."""
    colunas_com_tinta = np.where(mascara.any(axis=0))[0]
    if len(colunas_com_tinta) == 0:
        return
    esquerda = colunas_com_tinta[0]
    direita = colunas_com_tinta[-1] + 1

    # Para cada linha da imagem, a fracao pintada entre a primeira e a ultima
    # coluna com tinta. A pauta atravessa a palavra inteira; uma letra nao.
    preenchimento = mascara[:, esquerda:direita].mean(axis=1)
    linha_cheia = preenchimento > PAUTA_PREENCHIMENTO

    # Percorre as linhas procurando blocos de linhas cheias seguidas.
    altura = mascara.shape[0]
    inicio_do_bloco = None
    for y in range(altura + 1):
        cheia = y < altura and linha_cheia[y]
        if cheia and inicio_do_bloco is None:
            inicio_do_bloco = y
        elif not cheia and inicio_do_bloco is not None:
            espessura = y - inicio_do_bloco
            if espessura <= PAUTA_ESPESSURA_MAX:
                mascara[inicio_do_bloco:y] = False
            inicio_do_bloco = None


def deslocar(mascara, dx, dy):
    """Copia da mascara empurrada dx colunas para a direita e dy linhas para
    baixo. O que sai da imagem se perde e o que entra vem vazio."""
    altura, largura = mascara.shape
    deslocada = np.zeros_like(mascara)
    destino_y = slice(max(0, dy), min(altura, altura + dy))
    destino_x = slice(max(0, dx), min(largura, largura + dx))
    origem_y = slice(max(0, -dy), min(altura, altura - dy))
    origem_x = slice(max(0, -dx), min(largura, largura - dx))
    deslocada[destino_y, destino_x] = mascara[origem_y, origem_x]
    return deslocada


def deslocamento(mascara, referencia):
    """Quanto a mascara esta fora do lugar em relacao a referencia.

    Tenta todos os deslocamentos dentro do limite e fica com o que faz mais
    pixels de tinta coincidirem. Devolve (dx, dy). Se dois deslocamentos
    empatam, fica o menor.

    O gerador nem sempre desenha as duas palavras no mesmo lugar. Em 348 pares
    medidos, 60% sairam sem deslocamento, mas o maior chegou a 18 pixels.
    Subtrair sem alinhar faz a palavra inteira aparecer como diferenca.
    """
    # Candidatos do menor deslocamento para o maior. Como so troca quando a
    # contagem e estritamente maior, um empate fica com o menor deslocamento.
    candidatos = [(dx, dy)
                  for dy in range(-MAX_DESLOCAMENTO_Y, MAX_DESLOCAMENTO_Y + 1)
                  for dx in range(-MAX_DESLOCAMENTO_X, MAX_DESLOCAMENTO_X + 1)]
    candidatos.sort(key=lambda d: abs(d[0]) + abs(d[1]))

    melhor = (0, 0)
    mais_coincidencias = -1
    for dx, dy in candidatos:
        coincidencias = np.count_nonzero(deslocar(mascara, dx, dy) & referencia)
        if coincidencias > mais_coincidencias:
            mais_coincidencias = coincidencias
            melhor = (dx, dy)
    return melhor


def alinha(mascara, referencia):
    """A mascara posta no mesmo lugar da referencia."""
    dx, dy = deslocamento(mascara, referencia)
    return deslocar(mascara, dx, dy)


def corpo_das_letras(mascara):
    """As linhas da imagem onde fica o corpo das letras. Devolve (topo, base).

    Conta a tinta de cada linha. As linhas com pelo menos metade da tinta da
    linha mais cheia formam o corpo. Hastes de "d" e "l" ficam acima do topo,
    rabos de "p" e "g" ficam abaixo da base, e os acentos tambem.
    """
    tinta_por_linha = mascara.sum(axis=1)
    linhas_do_corpo = np.where(tinta_por_linha >= 0.5 * tinta_por_linha.max())[0]
    topo = linhas_do_corpo[0]
    base = linhas_do_corpo[-1] + 1
    return topo, base


def colunas_por_fatias_iguais(mascara, numero_de_letras):
    """As colunas de cada letra, supondo que todas tem a mesma largura.

    Devolve uma lista com (primeira coluna, ultima coluna + 1) por letra. Cada
    fatia leva meia letra de folga para cada lado, porque as letras nao tem
    de fato a mesma largura e o acento costuma passar um pouco dos limites.
    """
    colunas_com_tinta = np.where(mascara.any(axis=0))[0]
    inicio_da_palavra = colunas_com_tinta[0]
    fim_da_palavra = colunas_com_tinta[-1] + 1
    largura_da_letra = (fim_da_palavra - inicio_da_palavra) / numero_de_letras

    fatias = []
    for posicao in range(numero_de_letras):
        inicio = int(inicio_da_palavra + (posicao - 0.5) * largura_da_letra)
        fim = int(inicio_da_palavra + (posicao + 1.5) * largura_da_letra)
        fatias.append((max(0, inicio), fim))
    return fatias


# --------------------------------------------------------------- a medida

def e1(imagem_acentuada, imagem_sem_acento, palavra, colunas=None):
    """Quanto de acento foi desenhado. Devolve um numero por acento da palavra.

    O numero e uma fracao do tamanho da letra. Zero quer dizer que nada foi
    desenhado; um acento de tamanho normal fica entre 0.13 e 0.18.

    `colunas` permite informar onde fica cada letra, como uma lista de
    (inicio, fim) por letra. Sem isso a palavra e dividida em fatias iguais.
    """
    acentos = diacriticos(palavra)
    tinta_acentuada = tinta(imagem_acentuada)
    tinta_sem_acento = tinta(imagem_sem_acento)
    if not tinta_acentuada.any() or not tinta_sem_acento.any():
        return [0.0] * len(acentos)

    tinta_acentuada = alinha(tinta_acentuada, tinta_sem_acento)
    topo, base = corpo_das_letras(tinta_sem_acento)
    if colunas is None:
        colunas = colunas_por_fatias_iguais(tinta_sem_acento,
                                            len(sem_acento(palavra)))
    altura = tinta_sem_acento.shape[0]

    medidas = []
    for posicao_da_letra, fica_acima in acentos:
        inicio, fim = colunas[posicao_da_letra]
        colunas_da_letra = slice(inicio, fim)
        if fica_acima:
            linhas_do_acento = slice(0, topo)
        else:
            linhas_do_acento = slice(base, altura)

        tinta_no_par = tinta_sem_acento[linhas_do_acento, colunas_da_letra].sum()
        tinta_na_acentuada = tinta_acentuada[linhas_do_acento, colunas_da_letra].sum()
        tamanho_da_letra = tinta_sem_acento[topo:base, colunas_da_letra].sum()

        if tamanho_da_letra == 0:
            medidas.append(0.0)
        else:
            tinta_a_mais = float(tinta_na_acentuada) - float(tinta_no_par)
            medidas.append(tinta_a_mais / tamanho_da_letra)
    return medidas
