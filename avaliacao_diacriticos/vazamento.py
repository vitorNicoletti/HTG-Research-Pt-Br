"""
Marca em palavra que nao pede marca (vazamento), medida numa imagem so.

A e1.py compara a imagem acentuada com a sem acento. Se o modelo desenha um
acento nas duas, a diferenca e zero e a e1.py nao ve. Aqui a medida olha uma
imagem sozinha: para cada letra, quanta tinta ha acima do corpo das letras e
quanta ha abaixo, dentro da coluna da letra, dividido pela tinta do corpo da
letra.

So valem as letras que nao tem haste nem perna. Um "t" tem tinta acima do corpo
sem ter acento. As vogais a, e, o, u nao tem, entao tinta acima delas e marca.
O "c" nao tem perna, entao tinta abaixo dele e cedilha. O "i" fica de fora por
causa do pingo.
"""
import e1

LETRAS_SEM_HASTE = set("aeou")   # tinta acima delas e acento
LETRAS_SEM_PERNA = set("c")      # tinta abaixo dela e cedilha


def tinta_fora_do_corpo(imagem, colunas):
    """Para cada letra, (tinta acima, tinta abaixo), em fracao do corpo dela."""
    mascara = e1.tinta(imagem)
    if not mascara.any():
        return [(0.0, 0.0)] * len(colunas)
    topo, base = e1.corpo_das_letras(mascara)
    medidas = []
    for inicio, fim in colunas:
        corpo = mascara[topo:base, inicio:fim].sum()
        if corpo == 0:
            medidas.append((0.0, 0.0))
            continue
        acima = mascara[:topo, inicio:fim].sum() / corpo
        abaixo = mascara[base:, inicio:fim].sum() / corpo
        medidas.append((float(acima), float(abaixo)))
    return medidas


def vazamento(imagem, palavra, colunas):
    """A maior medida entre as letras que valem. Zero se nenhuma vale."""
    medidas = tinta_fora_do_corpo(imagem, colunas)
    maior = 0.0
    for letra, (acima, abaixo) in zip(e1.sem_acento(palavra), medidas):
        if letra in LETRAS_SEM_HASTE:
            maior = max(maior, acima)
        if letra in LETRAS_SEM_PERNA:
            maior = max(maior, abaixo)
    return maior
