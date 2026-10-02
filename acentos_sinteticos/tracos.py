"""Formas parametricas dos sinais, com variacao aleatoria.

Cada funcao recebe um random.Random e o tamanho de referencia (altura-x da
palavra, em px) e devolve (pontos, params):

    pontos -- lista de (x, y) em px, y crescendo para BAIXO, com o ponto de
              encaixe na origem: para acentos, (0, 0) e o meio da borda de
              baixo do sinal; para a cedilha, (0, 0) e onde ela sai da letra;
    params -- os valores sorteados, para o manifesto.

Todos os parametros sao continuos, entao dois sinais do mesmo tipo nunca saem
iguais: largura, altura, curvatura, inclinacao, assimetria e encurvamento dos
bracos variam a cada chamada. As faixas sao proporcoes da altura-x, para o
sinal acompanhar o tamanho da letra.
"""

import math

N_PONTOS = 40


def _bezier(p, n=N_PONTOS):
    """Curva de Bezier (quadratica ou cubica) pelos pontos de controle p."""
    saida = []
    for i in range(n):
        t = i / (n - 1)
        if len(p) == 3:
            a, b, c = p
            x = (1 - t) ** 2 * a[0] + 2 * (1 - t) * t * b[0] + t ** 2 * c[0]
            y = (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * b[1] + t ** 2 * c[1]
        else:
            a, b, c, d = p
            x = ((1 - t) ** 3 * a[0] + 3 * (1 - t) ** 2 * t * b[0]
                 + 3 * (1 - t) * t ** 2 * c[0] + t ** 3 * d[0])
            y = ((1 - t) ** 3 * a[1] + 3 * (1 - t) ** 2 * t * b[1]
                 + 3 * (1 - t) * t ** 2 * c[1] + t ** 3 * d[1])
        saida.append((x, y))
    return saida


def _girar(pontos, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [(x * c - y * s, x * s + y * c) for x, y in pontos]


def _encaixar_embaixo(pontos):
    """Desloca para o meio da borda de baixo ficar em (0, 0)."""
    xs, ys = [p[0] for p in pontos], [p[1] for p in pontos]
    dx, dy = (min(xs) + max(xs)) / 2, max(ys)
    return [(x - dx, y - dy) for x, y in pontos]


def til(rnd, ref):
    """Onda: sobe, desce e sobe de novo. Curvatura = amplitude/largura e ciclos."""
    p = {
        "largura": ref * rnd.uniform(0.75, 1.25),
        "amplitude": ref * rnd.uniform(0.08, 0.20),
        "ciclos": rnd.uniform(0.8, 1.2),
        "fase": rnd.uniform(-0.35, 0.35),
        "assimetria": rnd.uniform(-0.35, 0.35),   # lobo esquerdo vs direito
        "inclinacao": rnd.uniform(-0.30, 0.20),   # rad; negativo sobe a direita
    }
    pts = []
    for i in range(N_PONTOS):
        t = i / (N_PONTOS - 1)
        amp = p["amplitude"] * (1 + p["assimetria"] * (1 if t < 0.5 else -1))
        y = -amp * math.sin(2 * math.pi * p["ciclos"] * t + p["fase"])
        pts.append(((t - 0.5) * p["largura"], y))
    return _encaixar_embaixo(_girar(pts, p["inclinacao"])), p


def _traco_inclinado(rnd, ref, para_direita):
    p = {
        "comprimento": ref * rnd.uniform(0.40, 0.75),
        "angulo_graus": rnd.uniform(40, 75),       # em relacao a horizontal
        "curvatura": rnd.uniform(-0.18, 0.18),     # flecha / comprimento
    }
    a = math.radians(p["angulo_graus"])
    L = p["comprimento"]
    fim = (L * math.cos(a) * (1 if para_direita else -1), -L * math.sin(a))
    meio = (fim[0] / 2, fim[1] / 2)
    nx, ny = -fim[1] / L, fim[0] / L               # normal unitaria
    ctrl = (meio[0] + nx * p["curvatura"] * L, meio[1] + ny * p["curvatura"] * L)
    return _encaixar_embaixo(_bezier([(0, 0), ctrl, fim])), p


def agudo(rnd, ref):
    """Traco curto subindo para a direita, levemente curvo."""
    return _traco_inclinado(rnd, ref, para_direita=True)


def grave(rnd, ref):
    """Traco curto subindo para a esquerda, levemente curvo."""
    return _traco_inclinado(rnd, ref, para_direita=False)


def circunflexo(rnd, ref):
    """V invertido: abertura, bracos desiguais, apice deslocado e arredondado."""
    p = {
        "largura": ref * rnd.uniform(0.55, 0.95),
        "altura": ref * rnd.uniform(0.25, 0.45),
        "desvio_apice": rnd.uniform(-0.15, 0.15),  # fracao da largura
        "razao_bracos": rnd.uniform(0.8, 1.2),     # altura do pe direito / esquerdo
        "arredondamento": rnd.uniform(0.0, 0.35),  # 0 = apice em bico
        "curvatura_bracos": rnd.uniform(-0.12, 0.12),
    }
    w, h = p["largura"], p["altura"]
    ape = (p["desvio_apice"] * w, -h)
    esq = (-w / 2, 0.0)
    dir_ = (w / 2, -h + h * p["razao_bracos"])
    r = p["arredondamento"]
    a1 = (ape[0] + (esq[0] - ape[0]) * r, ape[1] + (esq[1] - ape[1]) * r)
    a2 = (ape[0] + (dir_[0] - ape[0]) * r, ape[1] + (dir_[1] - ape[1]) * r)
    cb = p["curvatura_bracos"] * w

    def braco(a, b):
        meio = ((a[0] + b[0]) / 2 + cb, (a[1] + b[1]) / 2 - cb)
        return _bezier([a, meio, b], N_PONTOS // 2)

    pts = braco(esq, a1) + _bezier([a1, ape, a2], 8) + braco(a2, dir_)
    return _encaixar_embaixo(pts), p


def cedilha(rnd, ref):
    """Gancho que desce da base do c e volta para a esquerda.

    Pontos de controle de uma Bezier cubica: desce quase reto, abre para a
    direita (barriga) e termina voltando para a esquerda. O tamanho da
    barriga e da volta e o que diferencia uma cedilha da outra.
    """
    p = {
        "tamanho": ref * rnd.uniform(0.35, 0.60),
        "desvio_descida": rnd.uniform(-0.10, 0.15),
        "barriga": rnd.uniform(0.40, 0.85),
        "volta": rnd.uniform(0.10, 0.50),
        "profundidade": rnd.uniform(0.85, 1.15),
    }
    s = p["tamanho"]
    pts = _bezier([(0.0, 0.0),
                   (p["desvio_descida"] * s, 0.40 * s),
                   (p["barriga"] * s, 0.85 * s * p["profundidade"]),
                   (-p["volta"] * s, 1.0 * s * p["profundidade"])])
    return pts, p


FORMAS = {"til": til, "agudo": agudo, "grave": grave,
          "circunflexo": circunflexo, "cedilha": cedilha}
