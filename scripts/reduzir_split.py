"""Monta uma reparticao menor do BRESSAY a partir de bressay_split/.

Reduz SO o treino. Validacao e teste sao copiados sem mudanca, para que a
avaliacao continue comparavel com os runs feitos no split inteiro.

O corte e por escritor: de cada um sorteia-se a fracao pedida das palavras,
com um piso de --min_por_escritor (quem tem menos que o piso fica com tudo).
Assim todos os escritores continuam presentes, e o sorteio das 5 imagens de
estilo no bressay_dataset.py nao depende tanto de reposicao.

Uso:
    python scripts/reduzir_split.py --fracao 0.25 --destino ./bressay_split_25

Depois, no treino:
    python scripts/treinar.py experimentos/bressay_25.json   (dados.split = ./bressay_split_25)
"""

import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

DIAC = set("àáâãçéêíóôõúüÀÁÂÃÇÉÊÍÓÔÕÚÜ")


def ler_tsv(caminho):
    linhas = []
    for l in caminho.read_text(encoding="utf-8").splitlines():
        partes = l.split("\t")
        if len(partes) >= 3:
            linhas.append(partes[:3])
    return linhas


def resumo(nome, linhas):
    escritores = collections.Counter(p[1] for p in linhas)
    ac = sum(any(c in DIAC for c in p[2]) for p in linhas)
    n = sorted(escritores.values())
    print(f"{nome:6} amostras={len(linhas):6}  escritores={len(escritores):4}  "
          f"com diacritico={ac} ({100 * ac / len(linhas):.1f}%)  "
          f"por escritor min/mediana/max={n[0]}/{n[len(n) // 2]}/{n[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", default="./bressay_split")
    ap.add_argument("--destino", required=True)
    ap.add_argument("--fracao", type=float, required=True,
                    help="fracao das palavras de cada escritor mantida no treino")
    ap.add_argument("--min_por_escritor", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not 0 < args.fracao <= 1:
        sys.exit("--fracao tem de estar em (0, 1]")

    origem, destino = Path(args.origem), Path(args.destino)
    if destino.resolve() == origem.resolve():
        sys.exit("destino igual a origem: isso sobrescreveria o split inteiro")

    treino = ler_tsv(origem / "splits" / "train.tsv")
    por_escritor = collections.defaultdict(list)
    for p in treino:
        por_escritor[p[1]].append(p)

    rnd = random.Random(args.seed)
    reduzido = []
    for wid in sorted(por_escritor, key=int):
        itens = por_escritor[wid]
        k = max(round(len(itens) * args.fracao), args.min_por_escritor)
        reduzido += rnd.sample(itens, min(k, len(itens)))
    rnd.shuffle(reduzido)

    (destino / "splits").mkdir(parents=True, exist_ok=True)
    with open(destino / "splits" / "train.tsv", "w", encoding="utf-8") as f:
        for p in reduzido:
            f.write("\t".join(p) + "\n")
    for nome in ("val.tsv", "test.tsv"):
        shutil.copyfile(origem / "splits" / nome, destino / "splits" / nome)
    shutil.copyfile(origem / "writers_dict.json", destino / "writers_dict.json")

    (destino / "reducao.json").write_text(json.dumps({
        "origem": str(origem), "fracao": args.fracao,
        "min_por_escritor": args.min_por_escritor, "seed": args.seed,
        "treino_original": len(treino), "treino_reduzido": len(reduzido),
    }, indent=1), encoding="utf-8")

    resumo("antes", treino)
    resumo("depois", reduzido)
    print(f"\nsplit reduzido em {destino}  (val e test copiados sem mudanca)")


if __name__ == "__main__":
    main()
