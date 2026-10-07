"""Vocabulario portugues particionado (vocabulario_pt/palavras.tsv).

A particao e por GRUPO de palavra (esqueleto sem acento com o plural
dobrado) e foi congelada por scripts/preparar_vocabulario_pt.py antes de
qualquer imagem ser gerada. Use conferir() em todo ponto que poe palavra no
treino ou tira palavra para avaliar: ele levanta erro em vez de deixar o
vazamento passar em silencio.
"""

import os
import unicodedata

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO = os.path.join(RAIZ, "vocabulario_pt", "palavras.tsv")


def esqueleto(p):
    """Palavra sem diacritico (c cedilha -> c)."""
    return "".join(c for c in unicodedata.normalize("NFD", p)
                   if unicodedata.category(c) != "Mn")


def grupo(p):
    """Chave de grupo: esqueleto com plural dobrado (oes/aes/aos -> ao; s final)."""
    e = esqueleto(p)
    for suf in ("oes", "aes", "aos"):
        if e.endswith(suf) and len(e) > 3:
            return e[:-3] + "ao"
    if e.endswith("s") and len(e) > 3:
        return e[:-1]
    return e


class Vocabulario:
    def __init__(self, caminho=ARQUIVO):
        self.palavras = {}         # palavra -> dict da linha
        self.split_do_grupo = {}
        with open(caminho, encoding="utf-8") as f:
            cab = f.readline().rstrip("\n").split("\t")
            for l in f:
                d = dict(zip(cab, l.rstrip("\n").split("\t")))
                d["frequencia"] = int(d["frequencia"])
                d["acentuada"] = d["acentuada"] == "1"
                self.palavras[d["palavra"]] = d
                self.split_do_grupo[d["grupo"]] = d["split"]

    def split(self, palavra):
        """Split do grupo da palavra. Palavra de grupo desconhecido -> None."""
        return self.split_do_grupo.get(grupo(unicodedata.normalize("NFC", palavra).lower()))

    def lista(self, split, acentuada=None):
        return [p for p, d in self.palavras.items() if d["split"] == split
                and (acentuada is None or d["acentuada"] == acentuada)]

    def conferir(self, palavras, permitido, contexto):
        """Levanta ValueError se alguma palavra (rotulo portugues) tiver o grupo
        fora de `permitido`. Palavras do IAM em ingles sao de outro dominio e
        nao passam por aqui."""
        permitido = {permitido} if isinstance(permitido, str) else set(permitido)
        ruins = sorted({p for p in palavras if self.split(p) not in permitido})
        if ruins:
            raise ValueError(f"{contexto}: {len(ruins)} palavra(s) fora de {sorted(permitido)} "
                             f"(vazamento entre splits): {ruins[:20]}")
