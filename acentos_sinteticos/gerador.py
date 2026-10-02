"""Acentua uma palavra do IAM: escolhe a letra e o sinal, posiciona e desenha."""

import math
from dataclasses import dataclass, field

import numpy as np

from . import desenho, geometria, tracos

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


def candidatos(palavra, pesos=PESOS_PADRAO):
    """[(indice, letra_acentuada, peso)] possiveis numa palavra."""
    return [(i, v, pesos.get(v, 0.0)) for i, c in enumerate(palavra)
            for v in VARIANTES.get(c, ()) if pesos.get(v, 0.0) > 0]


def acentuar(g, palavra, rnd, pesos=PESOS_PADRAO, escolha=None):
    """Desenha um sinal numa palavra do IAM.

    g        -- imagem em cinza 0..255 (float ou uint8)
    palavra  -- transcricao; cada caractere conta como uma letra nas fatias
    rnd      -- random.Random; toda a variacao sai dele (reprodutivel)
    escolha  -- (indice, letra_acentuada) para forcar; None = sorteia
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
    fats = geometria.fatias(geo, len(palavra))
    fatia = fats[i]

    if letra == "í":
        pingo = geometria.pingos_do_i(geo, fatia)
        if pingo.any():
            g = desenho.apagar(g, pingo)
            geo.mask &= ~pingo

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
    pts = [(c.x + dx + x, c.y + dy + y) for x, y in forma]

    # abre espaco se o sinal passar da borda (o IAM corta justo no topo)
    m = esp + 2
    h, w = g.shape
    cima = max(0, math.ceil(m - min(p[1] for p in pts)))
    baixo = max(0, math.ceil(max(p[1] for p in pts) + m - (h - 1)))
    esq = max(0, math.ceil(m - min(p[0] for p in pts)))
    dir_ = max(0, math.ceil(max(p[0] for p in pts) + m - (w - 1)))
    g = desenho.ampliar_tela(g, cima, baixo, esq, dir_)
    pts = [(x + esq, y + cima) for x, y in pts]

    g = desenho.desenhar(g, pts, esp, tom)

    params.update({"folga_y": round(-dy, 3), "desvio_x": round(dx, 3),
                   "espessura": round(esp, 3), "tom": round(tom, 1),
                   "altura_x": ref, "margens": [cima, baixo, esq, dir_]})
    return Amostra(
        imagem=np.clip(g, 0, 255).astype(np.uint8),
        original=palavra,
        rotulo=palavra[:i] + letra + palavra[i + 1:],
        indice=i, letra=letra, tipo=tipo, params=params,
        fatias=[(a + esq, b + esq) for a, b in fats],
        contato=(c.x + esq, c.y + cima),
        corpo=(geo.topo_x + cima, geo.base + cima),
    )
