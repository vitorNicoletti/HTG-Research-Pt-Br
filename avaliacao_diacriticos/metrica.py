"""
metrica.py -- nucleo da metrica de diacriticos.

Dois eixos independentes, como no planejamento:

  E1 (presenca)  -- ha tinta na faixa esperada do diacritico?
  E2 (integridade da base) -- a palavra continua legivel descontando-se o
                              diacritico? (CER de um reconhecedor, dobrado
                              para ASCII dos dois lados)

E1 tem duas implementacoes, e elas NAO sao intercambiaveis:

  e1_por_faixa  -- usa so a imagem acentuada. Estima linha de base e altura-x
                   e olha a faixa certa RESTRITA A COLUNA do caractere
                   acentuado. E a unica que funciona em recortes reais do
                   BRESSAY, onde nao existe gemeo ASCII do mesmo punho.
  e1_por_diff   -- usa o par minimo. Mede a tinta que a acentuada tem e a
                   gemea ASCII nao tem, na mesma faixa e coluna. So vale
                   porque o Passo 1 mostrou que os gemeos saem alinhados
                   (dx=dy=0, IoU mediano 0.93, corr de colunas 0.999).

Por que a coluna importa: tinta acima da altura-x tambem vem de ascendentes
(b, d, f, h, k, l, t) e do pingo do "i". Olhar a largura toda da palavra
contaria ascendente como acento.

Por que o "i" e um caso a parte: em "pais" o "i" TEM pingo, e em "pais" com
agudo o pingo e substituido pelo acento. Na coluna do "i" os dois gemeos tem
tinta na faixa de cima, entao presenca binaria nao separa nada -- o que separa
e a MASSA (o agudo e um traco inclinado, maior que o pingo). Por isso E1
devolve um escore continuo e o limiar e calibrado nos controles do Passo 4,
em vez de ser cravado a priori.
"""

import unicodedata

import numpy as np

ACIMA, ABAIXO = "acima", "abaixo"

# combinantes Unicode -> (nome, onde fica em relacao ao corpo da letra)
MARCAS = {
    "́": ("agudo", ACIMA),
    "̀": ("grave", ACIMA),
    "̂": ("circunflexo", ACIMA),
    "̃": ("til", ACIMA),
    "̧": ("cedilha", ABAIXO),
}


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def diacriticos(palavra):
    """[(indice do caractere base, nome da marca, ACIMA/ABAIXO), ...].

    O indice e contado sobre a palavra JA sem acento, que e exatamente a
    gemea ASCII -- logo as duas imagens do par tem o mesmo numero de
    caracteres e a mesma indexacao.
    """
    saida = []
    idx = -1
    for c in unicodedata.normalize("NFD", palavra):
        if unicodedata.category(c) == "Mn":
            if c in MARCAS:
                nome, onde = MARCAS[c]
                saida.append((idx, nome, onde))
        else:
            idx += 1
    return saida


def carregar_tinta(caminho):
    """PNG -> array float 0..1 onde 1 = tinta (escuro)."""
    from PIL import Image
    g = np.asarray(Image.open(caminho).convert("L"), dtype=np.float32) / 255.0
    return 1.0 - g


def binariza(t):
    """Mascara de tinta por Otsu, com guarda de contraste.

    Limiar relativo (min/max) nao serve: numa imagem quase vazia ele promove
    o ruido do fundo a "tinta" e a metrica passaria a contar acento onde nao
    ha nada. Se a imagem nao tem dois modos, e declarada sem tinta.
    """
    import cv2
    if float(t.max()) - float(t.min()) < 0.10:
        return np.zeros(t.shape, dtype=bool)
    u8 = (np.clip(t, 0, 1) * 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m.astype(bool)


# --------------------------- linha pautada -----------------------------
#
# 73% dos alvos de treino do BRESSAY (e 88% dos recortes de teste que esta
# metrica usa como controle positivo) tem a linha pautada do papel atravessando
# a imagem. Ela nao e tinta do escritor, e estraga as duas coisas que o E1
# precisa:
#
#   1. a geometria. linha_base_e_altura_x() chama de "corpo" as fileiras com
#      pelo menos metade da tinta da fileira mais cheia. A pauta atravessa a
#      imagem inteira, entao ela E a fileira mais cheia: o limiar sobe, so as
#      vizinhas dela sobrevivem e o corpo estimado cai de 25 px para 12 px.
#      Com o corpo encolhido, a faixa ACIMA da altura-x passa a engolir a
#      letra inteira em vez de so a zona do diacritico.
#   2. a contagem. Para a cedilha a faixa e ABAIXO da linha de base, que e
#      exatamente onde a pauta costuma estar -- a pauta seria contada como
#      cedilha.
#
# Os dois efeitos inflam o escore na mesma direcao. Medido no controle
# positivo, sem remover a pauta: agudo 1.070 e til 1.035, contra 0.265 e 0.328
# nos recortes sem pauta do mesmo conjunto.
#
# ESPESSURA e o que separa pauta de traco de letra, nao cobertura. Uma palavra
# curta e cursiva tem traco horizontal cobrindo 100% da propria caixa de tinta
# -- medido em "que" e "para", 5 a 7 fileiras consecutivas a 100%. A pauta e
# fina: no conjunto real as bandas cheias se concentram em 2-4 px, com queda
# brusca depois de 5, que e o esperado para uma linha de 1-2 px ampliada 2.06x
# (os recortes do BRESSAY tem 31 px de altura e sao ampliados para 64).

ESPESSURA_MAX_PAUTA = 5
COBERTURA_PAUTA = 0.90


def bandas_cheias(mask, cobertura=COBERTURA_PAUTA):
    """Faixas de fileiras consecutivas quase totalmente preenchidas.

    A cobertura e medida sobre a caixa de tinta, nao sobre a largura da
    imagem: load_image() centraliza recortes estreitos com pad branco, entao
    uma pauta real nao chega as bordas da imagem.
    """
    cx = caixa_tinta(mask)
    if cx is None:
        return []
    x0, x1, _, _ = cx
    cheia = mask[:, x0:x1].mean(axis=1) > cobertura
    faixas, i = [], 0
    while i < len(cheia):
        if cheia[i]:
            j = i
            while j + 1 < len(cheia) and cheia[j + 1]:
                j += 1
            faixas.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return faixas


def remover_pauta(mask, espessura_max=ESPESSURA_MAX_PAUTA,
                  cobertura=COBERTURA_PAUTA):
    """Zera as fileiras da linha pautada. Devolve (mascara, n_fileiras).

    So remove bandas FINAS. Sem o teto de espessura a funcao apaga o corpo de
    palavras curtas: em "que", "para" e "das" a remocao ingenua levava 52% a
    62% da tinta e deixava a palavra irreconhecivel.

    Nao tenta reconstruir o traco da letra que cruza a pauta. Para o que o E1
    faz -- estimar geometria e contar massa -- perder 1 a 4 fileiras da letra
    e um erro pequeno e igual nos dois gemeos do par minimo.
    """
    m = mask.copy()
    n = 0
    for ini, fim in bandas_cheias(mask, cobertura):
        if fim - ini <= espessura_max:
            m[ini:fim, :] = False
            n += fim - ini
    return m, n


def tem_pauta(mask, espessura_max=ESPESSURA_MAX_PAUTA,
              cobertura=COBERTURA_PAUTA):
    return any(fim - ini <= espessura_max
               for ini, fim in bandas_cheias(mask, cobertura))


def mascara_de_tinta(tinta, remover_pauta_=True):
    """Binariza e, por padrao, tira a pauta. Devolve (mascara, n_fileiras)."""
    m = binariza(tinta)
    if not remover_pauta_:
        return m, 0
    return remover_pauta(m)


def caixa_tinta(mask):
    """(x0, x1, y0, y1) da caixa que contem toda a tinta; None se vazia."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1


def linha_base_e_altura_x(mask):
    """(y_altura_x, y_base) pelo perfil de tinta por linha.

    O corpo principal da palavra sao as linhas com pelo menos metade da tinta
    da linha mais cheia; a primeira delas e o topo da altura-x e a ultima e a
    linha de base. Ascendentes e descendentes ficam de fora justamente por
    terem pouca tinta por linha. Devolve None se nao ha tinta.
    """
    if not mask.any():
        return None
    perfil = mask.sum(axis=1).astype(np.float32)
    corpo = np.where(perfil >= 0.5 * perfil.max())[0]
    if len(corpo) == 0:
        return None
    return int(corpo[0]), int(corpo[-1]) + 1


def coluna_do_caractere(mask, indice, n_caracteres, folga=0.5):
    """Janela horizontal (c0, c1) do caractere `indice`.

    Divide a caixa de tinta em `n_caracteres` fatias iguais. E uma aproximacao
    grosseira -- as letras nao tem a mesma largura --, mas e a MESMA
    aproximacao nas duas imagens do par minimo, que tem o mesmo numero de
    caracteres e (medido no Passo 1) o mesmo alinhamento. `folga` alarga a
    janela em fracoes de largura de caractere para absorver o erro.

    Segmentar de verdade (vales da projecao vertical) foi descartado: nas
    imagens deste gerador a palavra nao e legivel e os vales nao correspondem
    a fronteiras de caractere.
    """
    cx = caixa_tinta(mask)
    if cx is None or n_caracteres <= 0:
        return None
    x0, x1, _, _ = cx
    larg = (x1 - x0) / n_caracteres
    c0 = x0 + (indice - folga) * larg
    c1 = x0 + (indice + 1 + folga) * larg
    c0 = max(0, int(np.floor(c0)))
    c1 = min(mask.shape[1], int(np.ceil(c1)))
    return (c0, c1) if c1 > c0 else None


def _regiao(mask, indice, n_caracteres, onde, folga=0.5):
    """(fatia_linhas, fatia_colunas) da faixa do diacritico, ou None."""
    col = coluna_do_caractere(mask, indice, n_caracteres, folga)
    lb = linha_base_e_altura_x(mask)
    if col is None or lb is None:
        return None
    y_x, y_base = lb
    if onde == ACIMA:
        linhas = slice(0, y_x)
    else:
        linhas = slice(y_base, mask.shape[0])
    if linhas.stop <= linhas.start:
        return None
    return linhas, slice(col[0], col[1])


def e1_por_faixa(tinta, palavra, folga=0.5, remover_pauta_=True):
    """Escore de presenca usando so a imagem acentuada.

    Devolve um dict por diacritico com:
      massa      -- pixels de tinta na faixa x coluna
      densidade  -- massa / area da regiao
      massa_rel  -- massa / tinta do corpo na MESMA coluna. Normaliza pelo
                    tamanho da letra, entao nao depende da escala do traco.
    """
    mask, n_pauta = mascara_de_tinta(tinta, remover_pauta_)
    n = len(sem_acento(palavra))
    saida = []
    for indice, nome, onde in diacriticos(palavra):
        r = _regiao(mask, indice, n, onde, folga)
        if r is None:
            saida.append({"indice": indice, "marca": nome, "onde": onde,
                          "massa": 0, "densidade": 0.0, "massa_rel": 0.0,
                          "fileiras_pauta": n_pauta, "medivel": False})
            continue
        linhas, colunas = r
        massa = int(mask[linhas, colunas].sum())
        area = (linhas.stop - linhas.start) * (colunas.stop - colunas.start)
        y_x, y_base = linha_base_e_altura_x(mask)
        corpo = int(mask[y_x:y_base, colunas].sum())
        saida.append({
            "indice": indice, "marca": nome, "onde": onde,
            "massa": massa,
            "densidade": round(massa / area, 5) if area else 0.0,
            "massa_rel": round(massa / corpo, 5) if corpo else 0.0,
            "fileiras_pauta": n_pauta,
            "medivel": True,
        })
    return saida


def e1_por_diff(tinta_acc, tinta_asc, palavra, folga=0.5, remover_pauta_=True):
    """Escore de presenca comparando o par minimo.

    A geometria (coluna, linha de base, altura-x) e calculada na imagem
    ASCII, que por construcao nao tem diacritico -- se fosse calculada na
    acentuada, o proprio acento empurraria o topo da altura-x para cima e
    encolheria a faixa onde ele deveria ser procurado.

    Devolve, por diacritico:
      massa_acc, massa_asc -- tinta de cada gemeo na regiao
      delta                -- massa_acc - massa_asc
      delta_rel            -- delta / tinta do corpo da ASCII na mesma coluna
      acima_do_topo        -- tinta da acentuada, na coluna, ALEM do extremo
                              da ASCII naquela coluna (acima dele para marcas
                              de cima, abaixo para cedilha). E o sinal mais
                              especifico: nao conta o que os dois ja tinham.
    """
    m_acc, n_pa = mascara_de_tinta(tinta_acc, remover_pauta_)
    m_asc, n_ps = mascara_de_tinta(tinta_asc, remover_pauta_)
    n = len(sem_acento(palavra))
    saida = []
    for indice, nome, onde in diacriticos(palavra):
        r = _regiao(m_asc, indice, n, onde, folga)
        if r is None:
            saida.append({"indice": indice, "marca": nome, "onde": onde,
                          "massa_acc": 0, "massa_asc": 0, "delta": 0,
                          "delta_rel": 0.0, "acima_do_topo": 0,
                          "fileiras_pauta": n_pa, "medivel": False})
            continue
        linhas, colunas = r
        m1 = int(m_acc[linhas, colunas].sum())
        m0 = int(m_asc[linhas, colunas].sum())
        y_x, y_base = linha_base_e_altura_x(m_asc)
        corpo = int(m_asc[y_x:y_base, colunas].sum())

        # extremo da ASCII dentro da coluna
        col_asc = m_asc[:, colunas]
        col_acc = m_acc[:, colunas]
        if col_asc.any():
            ys = np.where(col_asc.any(axis=1))[0]
            if onde == ACIMA:
                alem = int(col_acc[:ys.min(), :].sum())
            else:
                alem = int(col_acc[ys.max() + 1:, :].sum())
        else:
            alem = int(col_acc.sum())

        saida.append({
            "indice": indice, "marca": nome, "onde": onde,
            "massa_acc": m1, "massa_asc": m0, "delta": m1 - m0,
            "delta_rel": round((m1 - m0) / corpo, 5) if corpo else 0.0,
            "acima_do_topo": alem,
            "fileiras_pauta": n_pa,
            "medivel": True,
        })
    return saida


# ----------------------------- E2: CER ---------------------------------

def cer(referencia, hipotese):
    """Erro por caractere = Levenshtein / len(referencia).

    Implementado aqui para nao depender de editdistance/jiwer, que nao estao
    no ambiente. Nao ha limite superior: inserir muito texto pode passar de 1.
    """
    r, h = list(referencia), list(hipotese)
    if not r:
        return 0.0 if not h else 1.0
    ant = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        atual = [i]
        for j, hc in enumerate(h, 1):
            atual.append(min(ant[j] + 1, atual[j - 1] + 1,
                             ant[j - 1] + (rc != hc)))
        ant = atual
    return ant[-1] / len(r)


def dobra_ascii(s):
    """Minusculas, sem acento, so alfanumerico.

    O TrOCR e treinado em ingles e nao le diacriticos. Se o CER fosse
    calculado com eles, a incapacidade do RECONHECEDOR viraria erro do
    GERADOR, e o eixo de integridade da base mediria a coisa errada. Dobrar
    os dois lados para ASCII isola E2 de E1.
    """
    s = sem_acento(s).lower()
    return "".join(c for c in s if c.isalnum())
