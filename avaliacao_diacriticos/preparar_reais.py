"""
preparar_reais.py -- controle contra circularidade.

Monta um conjunto de recortes REAIS do BRESSAY no mesmo formato de manifest
que o gerador produz, para que a metrica rode identica nos dois. Serve para
dois controles do planejamento:

  * presenca (E1) em manuscrito humano de verdade deve ficar perto de 100%.
    Se nao ficar, o problema e do detector, nao do gerador.
  * CER (E2) do MESMO reconhecedor em imagem real e o piso de leitura. Sem
    ele, um CER alto nas imagens geradas nao distingue "gerador ruim" de
    "reconhecedor que nao le este tipo de letra".

As imagens passam pelo MESMO load_image do dataset usado no treino/geracao
(normalizacao por percentis + pad/resize para 256x64), senao a comparacao
estaria medindo pre-processamento.
"""
import argparse
import json
import os
import random
import sys
import unicodedata

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "diffusionpen", "DiffusionPen"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrica as M  # noqa: E402
from utils.bressay_dataset import BRESSAY_Dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-acentuadas", type=int, default=120)
    ap.add_argument("--n-ascii", type=int, default=120)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    import argparse as _ap
    args = _ap.Namespace(max_samples=0)
    ds = BRESSAY_Dataset(os.path.join(RAIZ, "bressay_split"), a.split,
                         transforms=None, args=args)

    com, sem = [], []
    for caminho, transcr, wid in ds.data:
        if len(transcr) < a.min_len or not transcr.isalpha():
            continue
        d = M.diacriticos(transcr)
        (com if d else sem).append((caminho, transcr, wid))

    rng = random.Random(a.seed)
    rng.shuffle(com)
    rng.shuffle(sem)
    # cobre as marcas de forma equilibrada em vez de deixar o agudo dominar
    por_marca = {}
    for item in com:
        for _, nome, _ in M.diacriticos(item[1]):
            por_marca.setdefault(nome, []).append(item)
    escolhidas, vistos = [], set()
    marcas = sorted(por_marca)
    i = 0
    while len(escolhidas) < a.n_acentuadas and any(por_marca.values()):
        m = marcas[i % len(marcas)]
        i += 1
        if not por_marca[m]:
            continue
        it = por_marca[m].pop()
        if it[0] in vistos:
            continue
        vistos.add(it[0])
        escolhidas.append(it)

    manifest = open(os.path.join(a.out_dir, "manifest.jsonl"), "w",
                    encoding="utf-8")
    n = 0
    for grupo, itens in (("acc", escolhidas), ("asc", sem[:a.n_ascii])):
        for caminho, transcr, wid in itens:
            im = ds.load_image(caminho)
            nome = f"real_{grupo}_{n:04d}.png"
            im.save(os.path.join(a.out_dir, nome))
            manifest.write(json.dumps({
                "arquivo": nome, "palavra": transcr,
                "par_id": f"real{n:04d}", "acentuada": grupo == "acc",
                "escritor": int(wid), "semente": -1, "real": True,
                "origem": caminho, "nan": False, "std": None,
                "colapsada": False,
            }, ensure_ascii=False) + "\n")
            n += 1
    manifest.close()
    print(f"{n} recortes reais em {a.out_dir} "
          f"({len(escolhidas)} acentuados, {len(sem[:a.n_ascii])} ASCII)")
    cont = {}
    for _, t, _ in escolhidas:
        for _, nome, _ in M.diacriticos(t):
            cont[nome] = cont.get(nome, 0) + 1
    print("marcas cobertas:", cont)


if __name__ == "__main__":
    main()
