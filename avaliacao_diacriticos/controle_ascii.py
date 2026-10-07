"""
controle_ascii.py -- controle negativo NO PROPRIO DOMINIO.

O controle negativo do Passo 4 e o DiffusionPen puro do IAM. Ele responde
"o gerador que nunca viu portugues omite o acento?", mas nao responde
"o DETECTOR inventa acento onde nao ha?", porque imagem gerada e recorte real
diferem em tudo: traco, contraste, e a presenca da pauta (0% no IAM, 88% no
real). Parte do escore 0.141 do IAM pode ser so ruido de renderizacao.

Aqui o negativo e construido de modo que a UNICA diferenca seja o acento:
pega-se uma palavra REAL do BRESSAY que comprovadamente nao tem diacritico e
finge-se que ela tem. Mesmo papel, mesma pauta, mesmo punho, mesma
normalizacao. Tudo que o E1 marcar aqui e falso positivo, por construcao.

    python avaliacao_diacriticos/controle_ascii.py \\
        --dir avaliacao_diacriticos/reais_test \\
        --out-dir avaliacao_diacriticos/reais_ascii_negativo

A saida e um manifest no mesmo formato, entao o avaliar.py roda sem mudanca.
As imagens sao ligadas por symlink -- nenhum pixel e copiado nem alterado.
"""
import argparse
import json
import os
import random
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402

# onde cada marca pode pousar, em portugues
BASES = {
    "til": "ao",
    "agudo": "aeiou",
    "grave": "a",
    "circunflexo": "aeo",
    "cedilha": "c",
}
COMBINANTE = {"til": "̃", "agudo": "́", "grave": "̀",
              "circunflexo": "̂", "cedilha": "̧"}


def injeta(palavra, marca, rng):
    """Poe `marca` numa letra compativel de `palavra`. None se nao couber.

    A cedilha so vale em 'c' seguido de a/o/u -- em 'ce'/'ci' ela nao existe
    em portugues, e pedir o detector para procurar cedilha ali seria um
    negativo mais facil do que o caso real.
    """
    pos = []
    for i, c in enumerate(palavra.lower()):
        if c not in BASES[marca]:
            continue
        if marca == "cedilha" and not (i + 1 < len(palavra)
                                       and palavra[i + 1].lower() in "aou"):
            continue
        pos.append(i)
    if not pos:
        return None
    i = rng.choice(pos)
    return unicodedata.normalize(
        "NFC", palavra[:i] + palavra[i] + COMBINANTE[marca] + palavra[i + 1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    origem = os.path.abspath(a.dir)
    rng = random.Random(a.seed)

    # distribuicao de marcas do grupo POSITIVO, para o negativo nao ficar
    # concentrado numa marca so
    positivas = []
    ascii_reais = []
    for linha in open(os.path.join(a.dir, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(linha)
        if r.get("acentuada"):
            positivas += [n for _, n, _ in M.diacriticos(r["palavra"])]
        elif not M.diacriticos(r["palavra"]):
            ascii_reais.append(r)

    marcas = sorted(set(positivas)) or sorted(BASES)
    saida = open(os.path.join(a.out_dir, "manifest.jsonl"), "w", encoding="utf-8")
    n = 0
    pulados = 0
    cont = {}
    # Sortear a marca entre as que CABEM na palavra, em vez de percorrer as
    # marcas em rodizio: com rodizio a cedilha cai quase sempre em palavra sem
    # "c" seguido de a/o/u e o grupo fica com n=1, que nao da para interpretar.
    # O sorteio favorece a marca mais rara entre as viaveis, para equilibrar.
    for r in ascii_reais:
        viaveis = [m for m in marcas if injeta(r["palavra"], m, rng) is not None]
        if not viaveis:
            pulados += 1
            continue
        marca = min(viaveis, key=lambda m: (cont.get(m, 0), rng.random()))
        falsa = injeta(r["palavra"], marca, rng)
        alvo = os.path.join(a.out_dir, r["arquivo"])
        if not os.path.lexists(alvo):
            os.symlink(os.path.join(origem, r["arquivo"]), alvo)
        d = dict(r)
        d.update({"palavra": falsa, "acentuada": True,
                  "palavra_verdadeira": r["palavra"],
                  "controle_negativo_ascii": True})
        saida.write(json.dumps(d, ensure_ascii=False) + "\n")
        cont[marca] = cont.get(marca, 0) + 1
        n += 1
    saida.close()
    print(f"{n} negativos construidos em {a.out_dir} "
          f"({pulados} palavras puladas por nao ter letra compativel)")
    print("marcas:", cont)
    print("\nTudo que o E1 marcar aqui e falso positivo: as palavras sao reais")
    print("e comprovadamente nao tem diacritico nenhum.")


if __name__ == "__main__":
    main()
