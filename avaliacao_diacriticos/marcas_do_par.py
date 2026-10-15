"""
Juncao da e1 com o detector do colega: marca solta que so a acentuada tem.

O detector do colega (scripts/medir_marcas.py) acha manchas de tinta soltas
acima e abaixo do corpo das letras, e o alinhador diz a que letra cada uma
pertence (scripts/medir_posicao_acento.py). Aqui isso e feito nas duas imagens
do par. A marca do acento e a que aparece na imagem acentuada e nao aparece na
sem acento, na mesma letra e do mesmo lado.

O pingo do i aparece nas duas imagens e some na comparacao.
"""
import collections
import os
import sys

import e1

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
from medir_posicao_acento import letras_das_marcas  # noqa: E402


def marcas_novas(cinza_acentuada, cinza_sem_acento, palavra, alinhador):
    """Marcas que so a imagem acentuada tem: lista de (posicao da letra, lado).

    As imagens entram em tons de cinza de 0 a 255, com a tinta escura. O lado
    e "acima" ou "abaixo".
    """
    letras = e1.sem_acento(palavra)
    na_acentuada, _ = letras_das_marcas(cinza_acentuada, letras, alinhador)
    na_sem_acento, _ = letras_das_marcas(cinza_sem_acento, letras, alinhador)
    # subtrai as contagens: duas marcas na letra 3 de uma imagem e uma na
    # outra deixam uma marca nova na letra 3
    sobra = collections.Counter(na_acentuada) - collections.Counter(na_sem_acento)
    return sorted(sobra.elements())


def acento_desenhado(cinza_acentuada, cinza_sem_acento, palavra, alinhador):
    """Para cada acento da palavra, se ha marca nova na letra dele."""
    novas = marcas_novas(cinza_acentuada, cinza_sem_acento, palavra, alinhador)
    return [(posicao, "acima" if fica_acima else "abaixo") in novas
            for posicao, fica_acima in e1.diacriticos(palavra)]
