"""Acentua uma palavra do IAM: escolhe a letra e o sinal, posiciona e desenha."""

import math
from dataclasses import dataclass, field

import numpy as np

from . import alinhamento, desenho, geometria, tracos

# letra-base -> letras acentuadas do portugues que se desenham sobre ela
VARIANTES = {
    "a": ("ã", "á", "â", "à"),
    "o": ("õ", "ó", "ô"),
    "e": ("é", "ê"),
    "i": ("í",),
    "u": ("ú",),
    "c": ("ç",),
}

TIPO = {"ã": "til", "õ": "til", "á": "agudo", "é": "agudo", "í": "agudo",
        "ó": "agudo", "ú": "agudo", "à": "grave", "â": "circunflexo",
        "ê": "circunflexo", "ô": "circunflexo", "ç": "cedilha"}

# peso de sorteio de cada letra acentuada. O til pesa mais: e o diacritico
# mais frequente do portugues e o unico que nenhum dataset real com boa
# resolucao cobre (nem o RIMES).
PESOS_PADRAO = {"ã": 3.0, "õ": 1.5, "ç": 2.0, "é": 1.5, "ê": 1.0, "á": 1.0,
                "ó": 1.0, "í": 1.0, "ú": 0.8, "â": 0.7, "ô": 0.7, "à": 0.5}

# faixas sorteadas no posicionamento (em alturas-x da palavra)
FOLGA_ACENTO = (0.10, 0.35)     # distancia vertical entre a letra e o acento
DESVIO_X = (-0.10, 0.10)        # deslocamento horizontal do acento
ESPESSURA_REL = (0.60, 1.00)    # espessura do sinal / espessura da palavra
# extensao minima do sinal, em espessuras do proprio traco: com caneta grossa
# um agudo curto vira uma gota que parece pingo ("make" com traco de 7 px)
MIN_EXTENSAO_ESPESSURAS = 3.5
TOM_REL = (0.90, 1.10)          # tom do sinal / tom da tinta da palavra

# filtros de qualidade de gerar() (calibrados em
# saidas/acentos_sinteticos/avaliacao_200, ver LOG.md 2026-10-02)
MIN_LOGP = -0.2                 # confianca minima do alinhamento CTC
# fracao minima do sinal visivel como tinta nova: abaixo de 0,6 ficaram os 4
# sinais escondidos em tinta existente (for, has, strewn, side) e 1 bom
MIN_VISIBILIDADE = 0.6
TENTATIVAS = 3                  # sorteios por palavra ate o sinal ficar visivel


@dataclass
class Amostra:
    imagem: np.ndarray          # cinza uint8, ja com as margens acrescentadas
    original: str
    rotulo: str
    indice: int                 # posicao da letra acentuada
    letra: str                  # letra acentuada (ex.: "ã")
    tipo: str                   # til, agudo, grave, circunflexo, cedilha
    params: dict                # forma sorteada + posicionamento
    # para a folha de depuracao, em coordenadas da imagem final
    fatias: list = field(default_factory=list)
    contato: tuple = (0.0, 0.0)
    corpo: tuple = (0, 0)       # (topo_x, base)

    def manifesto(self):
        return {"original": self.original, "rotulo": self.rotulo,
                "indice": self.indice, "letra": self.letra, "tipo": self.tipo,
                "params": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in self.params.items()}}


def elegivel(texto):
    """Palavra que entra na base: minuscula, so letras ASCII, 3-10 letras,
    com ao menos uma letra acentuavel."""
    return (texto.isascii() and texto.isalpha() and texto.islower()
            and 3 <= len(texto) <= 10 and bool(candidatos(texto)))


def candidatos(palavra, pesos=PESOS_PADRAO):
    """[(indice, letra_acentuada, peso)] possiveis numa palavra."""
    return [(i, v, pesos.get(v, 0.0)) for i, c in enumerate(palavra)
            for v in VARIANTES.get(c, ()) if pesos.get(v, 0.0) > 0]


def _desenhar_sinal(g, tinta, geo, fatia, tipo, rnd, ox, oy):
    """Desenha um sinal na letra da `fatia`.

    geo e fatia estao nas coordenadas da imagem ORIGINAL; (ox, oy) e quanto a
    tela atual `g` ja foi ampliada a esquerda e em cima por sinais anteriores.
    `tinta` e a mascara de tinta da tela atual (para medir a visibilidade).
    A ordem dos sorteios (folga, desvio, forma, espessura, tom) e a do
    acentuar() original: a base inglesa continua reprodutivel.
    """
    ref = geo.altura_x
    if tipo == "cedilha":
        c = geometria.contato_inferior(geo, fatia)
        dy = -0.3 * geo.espessura          # nasce encostada na letra
    else:
        c = geometria.contato_superior(geo, fatia)
        dy = -(ref * rnd.uniform(*FOLGA_ACENTO) + geo.espessura / 2)
    dx = ref * rnd.uniform(*DESVIO_X)

    forma, params = tracos.FORMAS[tipo](rnd, ref)
    esp = geo.espessura * rnd.uniform(*ESPESSURA_REL)
    tom = float(np.clip(geo.tom * rnd.uniform(*TOM_REL), 0, 200))

    # garante o sinal comprido o bastante para a espessura; amplia sem
    # distorcer, a partir do ponto de encaixe (a origem)
    ext = max(max(x for x, _ in forma) - min(x for x, _ in forma),
              max(y for _, y in forma) - min(y for _, y in forma))
    esc = max(1.0, MIN_EXTENSAO_ESPESSURAS * esp / max(ext, 1e-6))
    forma = [(x * esc, y * esc) for x, y in forma]
    params["escala_minima"] = round(esc, 3)
    pts = [(c.x + ox + dx + x, c.y + oy + dy + y) for x, y in forma]

    # abre espaco se o sinal passar da borda (o IAM corta justo no topo)
    m = esp + 2
    h, w = g.shape
    cima = max(0, math.ceil(m - min(p[1] for p in pts)))
    baixo = max(0, math.ceil(max(p[1] for p in pts) + m - (h - 1)))
    esq = max(0, math.ceil(m - min(p[0] for p in pts)))
    dir_ = max(0, math.ceil(max(p[0] for p in pts) + m - (w - 1)))
    g = desenho.ampliar_tela(g, cima, baixo, esq, dir_)
    tinta = np.pad(tinta, ((cima, baixo), (esq, dir_)), constant_values=False)
    pts = [(x + esq, y + cima) for x, y in pts]

    antes = g
    g = desenho.desenhar(g, pts, esp, tom)
    vis = round(desenho.visibilidade(antes, g, tinta, pts, esp), 3)
    tinta = tinta | (antes - g > 30)         # o sinal vira tinta para os proximos
    return {"g": g, "tinta": tinta, "contato": c, "dy": dy, "dx": dx, "esp": esp,
            "tom": tom, "params": params, "margens": (cima, baixo, esq, dir_),
            "visibilidade": vis}


def acentuar(g, palavra, rnd, pesos=PESOS_PADRAO, escolha=None, alinhador=None, al=None):
    """Desenha um sinal numa palavra do IAM.

    g         -- imagem em cinza 0..255 (float ou uint8)
    palavra   -- transcricao; cada caractere conta como uma letra nas fatias
    rnd       -- random.Random; toda a variacao sai dele (reprodutivel)
    escolha   -- (indice, letra_acentuada) para forcar; None = sorteia
    alinhador -- alinhamento.Alinhador para localizar as letras; None (ou
                 falha no alinhamento) = fatias iguais. Com ele, as fronteiras
                 do CTC sao ajustadas aos vales de tinta (melhor estimador em
                 scripts/avaliar_posicao_letras.py) e params["logp_alinhamento"]
                 registra a confianca do alinhamento, para filtro
    al        -- alinhamento ja calculado (evita refazer a cada tentativa)
    params["visibilidade"] mede quanto do sinal virou tinta nova (ver
    desenho.visibilidade).
    Devolve Amostra, ou None se a palavra nao tiver tinta ou candidatos.
    """
    g = np.asarray(g, dtype=np.float32)
    geo = geometria.analisar(g)
    cands = candidatos(palavra, pesos)
    if geo is None or not cands:
        return None
    if escolha is None:
        i, letra, _ = rnd.choices(cands, weights=[c[2] for c in cands])[0]
    else:
        i, letra = escolha
    tipo = TIPO[letra]
    ref = geo.altura_x
    if al is None and alinhador is not None:
        al = alinhador.alinhar(g, palavra)
    if al is not None:
        fats = geometria.ajustar_aos_vales(geo, alinhamento.fatias_do_alinhamento(al, geo))
        segmentacao = "ctc_vale"
    else:
        fats = geometria.fatias(geo, len(palavra))
        segmentacao = "igual"
    fatia = fats[i]

    if letra == "í":
        pingo = geometria.pingos_do_i(geo, fatia)
        if pingo.any():
            g = desenho.apagar(g, pingo)
            geo.mask &= ~pingo

    s = _desenhar_sinal(g, geo.mask, geo, fatia, tipo, rnd, 0, 0)
    g, tinta, c, dy, dx, esp, tom, params = (s["g"], s["tinta"], s["contato"], s["dy"],
                                             s["dx"], s["esp"], s["tom"], s["params"])
    cima, baixo, esq, dir_ = s["margens"]
    params["visibilidade"] = s["visibilidade"]

    params.update({"folga_y": round(-dy, 3), "desvio_x": round(dx, 3),
                   "espessura": round(esp, 3), "tom": round(tom, 1),
                   "altura_x": ref, "margens": [cima, baixo, esq, dir_],
                   "segmentacao": segmentacao})
    if al is not None:
        params.update({"logp_alinhamento": round(al.logp_medio, 4),
                       "leitura_alinhador": al.leitura})
    return Amostra(
        imagem=np.clip(g, 0, 255).astype(np.uint8),
        original=palavra,
        rotulo=palavra[:i] + letra + palavra[i + 1:],
        indice=i, letra=letra, tipo=tipo, params=params,
        fatias=[(a + esq, b + esq) for a, b in fats],
        contato=(c.x + esq, c.y + cima),
        corpo=(geo.topo_x + cima, geo.base + cima),
    )


def gerar(g, palavra, rnd, alinhador, pesos=PESOS_PADRAO, min_logp=MIN_LOGP,
          min_visibilidade=MIN_VISIBILIDADE, tentativas=TENTATIVAS):
    """acentuar() com os filtros de qualidade, para montar a base.

    Descarta a palavra se o alinhamento falhar ou tiver confianca abaixo de
    min_logp (recorte ambiguo); sorteia de novo (letra, forma, posicao) ate
    `tentativas` vezes enquanto o sinal sair com visibilidade abaixo de
    min_visibilidade. Tudo sai do mesmo rnd, entao continua reprodutivel.
    Devolve (Amostra ou None, motivo): "ok", "sem_alinhamento",
    "confianca", "sem_candidato" ou "invisivel".
    """
    g = np.asarray(g, dtype=np.float32)
    al = alinhador.alinhar(g, palavra)
    if al is None:
        return None, "sem_alinhamento"
    if al.logp_medio < min_logp:
        return None, "confianca"
    for t in range(tentativas):
        am = acentuar(g, palavra, rnd, pesos, alinhador=alinhador, al=al)
        if am is None:
            return None, "sem_candidato"
        if min_visibilidade is None or am.params["visibilidade"] >= min_visibilidade:
            am.params["tentativa"] = t + 1
            return am, "ok"
    return None, "invisivel"


def acentuar_palavra(g, base, alvo, rnd, alinhador, min_logp=MIN_LOGP,
                     min_visibilidade=MIN_VISIBILIDADE, tentativas=TENTATIVAS):
    """Desenha TODOS os sinais de `alvo` sobre a imagem da palavra `base`.

    base  -- transcricao sem acento da imagem (ex.: "coracao"); e o que o
             alinhador CTC le
    alvo  -- a palavra acentuada (ex.: "coração"), mesmo comprimento; onde
             difere de base, alvo[i] tem de ser variante de base[i]
    Cada sinal tem ate `tentativas` sorteios para passar na visibilidade; se
    um deles nao passar, a amostra inteira e descartada -- rotulo com acento
    que nao foi desenhado e justamente o erro que o trabalho estuda.
    Devolve (imagem uint8 ou None, motivo, info).
    """
    if len(base) != len(alvo):
        raise ValueError(f"base e alvo com tamanhos diferentes: {base!r} {alvo!r}")
    posicoes = []
    for i, (b, a) in enumerate(zip(base, alvo)):
        if a != b:
            if a not in VARIANTES.get(b, ()):
                raise ValueError(f"{alvo!r}: {a!r} nao e variante de {b!r}")
            posicoes.append(i)
    g = np.asarray(g, dtype=np.float32)
    geo = geometria.analisar(g)
    if geo is None:
        return None, "sem_tinta", {}
    al = alinhador.alinhar(g, base)
    if al is None:
        return None, "sem_alinhamento", {}
    info = {"logp_alinhamento": round(al.logp_medio, 4), "leitura_alinhador": al.leitura,
            "altura_x": geo.altura_x, "sinais": []}
    if al.logp_medio < min_logp:
        return None, "confianca", info
    fats = geometria.ajustar_aos_vales(geo, alinhamento.fatias_do_alinhamento(al, geo))

    # pingos do i saem antes de qualquer desenho, nas coordenadas originais
    for i in posicoes:
        if alvo[i] == "í":
            pingo = geometria.pingos_do_i(geo, fats[i])
            if pingo.any():
                g = desenho.apagar(g, pingo)
                geo.mask &= ~pingo

    tinta, ox, oy = geo.mask.copy(), 0, 0
    for i in posicoes:
        for t in range(tentativas):
            s = _desenhar_sinal(g, tinta, geo, fats[i], TIPO[alvo[i]], rnd, ox, oy)
            if min_visibilidade is None or s["visibilidade"] >= min_visibilidade:
                break
        else:
            return None, "invisivel", info
        g, tinta = s["g"], s["tinta"]
        cima, _, esq, _ = s["margens"]
        ox, oy = ox + esq, oy + cima
        s["params"].update({"indice": i, "letra": alvo[i], "tentativa": t + 1,
                            "visibilidade": s["visibilidade"]})
        info["sinais"].append(s["params"])
    info["margens_esq_cima"] = [ox, oy]
    return np.clip(g, 0, 255).astype(np.uint8), "ok", info


def par_na_tela(g, forma_acentuada, margens_esq_cima):
    """Poe a imagem SEM acento na mesma tela da acentuada.

    Quando um sinal passa da borda, acentuar_palavra amplia a tela; sem este
    passo o par e a acentuada chegam ao treino em escalas diferentes (o
    pre-processamento leva as duas a 64 px de altura) e o contraste "so o
    acento muda" se perde (ACHADOS.md, secao 12). Mesma cor de papel e mesma
    funcao de margem do gerador.
    """
    ox, oy = margens_esq_cima
    H, W = forma_acentuada
    h, w = g.shape
    baixo, dir_ = H - h - oy, W - w - ox
    if min(ox, oy, baixo, dir_) < 0:
        raise ValueError(f"par {g.shape} nao cabe na tela {forma_acentuada} com margens {margens_esq_cima}")
    return desenho.ampliar_tela(np.asarray(g, dtype=np.float32), oy, baixo, ox, dir_)

