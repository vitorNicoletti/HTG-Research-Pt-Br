"""Lista canonica de palavras da sonda de diacriticos (Fase 2).

Compartilhada entre DiffusionPen e VATr++ de proposito: a Fase 4 exige que os
dois geradores recebam exatamente as mesmas palavras, senao a comparacao entre
eles nao se sustenta.

Nao editar sem registrar no LOG.md - mudar a lista invalida comparacoes com
execucoes anteriores.
"""

# Pares minimos: mesma palavra com e sem diacritico. E a comparacao mais
# informativa da sonda, entao ficam explicitos aqui e nao so implicitos nos grupos.
PARES_MINIMOS = [
    ("nacao", "nação"),
    ("coracao", "coração"),
]

GRUPOS = {
    # Controle: so ASCII. Se estas falharem, o problema nao e diacritico -
    # e o setup (pesos, patch de atencao) e a sonda inteira e invalida.
    "controle": ["casa", "pato", "vento", "nacao", "coracao"],

    # Til: diacritico superscrito, sobre a vogal.
    "til": ["mão", "pão", "nação", "irmã", "põe"],

    # Cedilha: traco subscrito. Compete visualmente com descendentes (g, p, q).
    "cedilha": ["coração", "ação", "março", "praça", "força"],

    # Agudo / circunflexo: superscrito, mas mais fino que o til.
    "agudo_circunflexo": ["você", "café", "três", "histórico", "português"],
}

SEEDS = [0, 1, 2]  # 3 seeds separa falha sistematica de variacao aleatoria

# Caracteres nao-ASCII que a sonda exercita, com nome, para os relatorios.
DIACRITICOS = {
    "ã": "a com til",
    "õ": "o com til",
    "ç": "c cedilha",
    "é": "e agudo",
    "ê": "e circunflexo",
    "ó": "o agudo",
    "ú": "u agudo",
}


def todas_palavras():
    """Todas as palavras da sonda, sem duplicatas, em ordem estavel."""
    vistas, saida = set(), []
    for palavras in GRUPOS.values():
        for p in palavras:
            if p not in vistas:
                vistas.add(p)
                saida.append(p)
    return saida


def itens():
    """Gera (grupo, palavra, seed) para toda a sonda."""
    for grupo, palavras in GRUPOS.items():
        for palavra in palavras:
            for seed in SEEDS:
                yield grupo, palavra, seed


def nome_arquivo(grupo, palavra, seed):
    """Nome de arquivo estavel e ASCII-safe (o proprio nome nao pode depender
    de o filesystem lidar bem com acento)."""
    slug = "".join(f"u{ord(c):04x}" if ord(c) > 127 else c for c in palavra)
    return f"{grupo}__{slug}__seed{seed}.png"


def caracteres_nao_ascii():
    """Conjunto de caracteres nao-ASCII exercitados pela sonda."""
    return sorted({c for p in todas_palavras() for c in p if ord(c) > 127})


if __name__ == "__main__":
    print(f"grupos: {len(GRUPOS)}  palavras unicas: {len(todas_palavras())}")
    print(f"seeds: {SEEDS}")
    print(f"total de imagens: {len(list(itens()))}")
    print(f"nao-ASCII exercitados: {' '.join(caracteres_nao_ascii())}")
    print()
    for grupo, palavras in GRUPOS.items():
        print(f"  {grupo:<20} {' '.join(palavras)}")
    print()
    print("pares minimos:")
    for a, b in PARES_MINIMOS:
        print(f"  {a} <-> {b}")
