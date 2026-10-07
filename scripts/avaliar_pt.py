"""Protocolo fixo de avaliacao dos acentos em portugues (escrito ANTES do modelo novo).

Palavras de UM split do vocabulario (val para escolher checkpoint, teste uma
vez no fim), escritores do iam_test (fora do treino), varias sementes:
  acentuada -- palavra com acento do split (ex.: "pântano")
  esqueleto -- a mesma sem acento ("pantano"): o par minimo, mesmo escritor e
               mesmo ruido inicial; mede o vazamento
  sem_acento -- palavra do split que nao tem acento ("tarde")
No teste entram tambem os pares da sonda (comum/palavras.py PARES_MINIMOS).
Palavras com i/j ficam de fora: o pingo conta como marca solta.

Medidas por painel:
  marca      -- ha marca solta acima/abaixo do corpo (scripts/medir_marcas.py)
  CER        -- leitura livre de um leitor CTC SEPARADO (cabeca LSTM; nao e o
                alinhador que filtrou a base) contra o esqueleto
Por modelo: taxa de marca em cada tipo, diferenca pareada acentuada menos
esqueleto (mesmo escritor e semente), CER medio por tipo.

Anti-vazamento (o script para com erro se falhar):
  - toda palavra avaliada e do --split pedido;
  - nenhuma palavra avaliada (pelo grupo) aparece nos split.txt de --bases,
    exceto os grupos de --aceitar_grupos, conferidos a mao e registrados no
    resumo.json (ex.: "central" na base de acentos sobre o IAM real, que tem a
    palavra inglesa real -- a mesma dos originais do IAM em todo treino);
  - os escritores de avaliacao nao estao no iam_train_val.

    python scripts/avaliar_pt.py --split val --modelo iam=<ckpt> --modelo pt=<ckpt> \\
        --bases iam_pt --leitor modelos/leitor_iam_lstm.pt --saida saidas/avaliacao_pt_val
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
from acentos_sinteticos import alinhamento, geometria  # noqa: E402
from acentos_sinteticos.vocabulario import Vocabulario, esqueleto, grupo  # noqa: E402
from medir_marcas import marcas  # noqa: E402

N_ACENTUADAS = 30
N_SEM_ACENTO = 20
N_ESCRITORES = 20
SEMENTES = (42, 43)
TAM = (3, 10)


def levenshtein(a, b):
    ant = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(ant[j] + 1, cur[j - 1] + 1, ant[j - 1] + (ca != cb)))
        ant = cur
    return ant[-1]


def escolher_palavras(voc, split):
    ok = lambda p: TAM[0] <= len(p) <= TAM[1] and not set("ij") & set(esqueleto(p))  # noqa: E731
    rnd = random.Random(0)
    ac = rnd.sample(sorted(p for p in voc.lista(split, True) if ok(p)), N_ACENTUADAS)
    sem = rnd.sample(sorted(p for p in voc.lista(split, False) if ok(p)), N_SEM_ACENTO)
    if split == "teste":
        from comum.palavras import PARES_MINIMOS
        ac += [com for _, com in PARES_MINIMOS if com not in ac]
    itens = [(p, "acentuada", esqueleto(p)) for p in ac] + \
            [(esqueleto(p), "esqueleto", esqueleto(p)) for p in ac] + \
            [(p, "sem_acento", p) for p in sem]
    return itens


def recorte_tinta(g, margem=3):
    """Painel 64x256 -> recorte justo na tinta, como os recortes do IAM que o leitor viu."""
    m = geometria.mascara_tinta(g)
    if m.sum() < 20:
        return None
    ys, xs = np.where(m)
    return g[max(0, ys.min() - margem):ys.max() + 1 + margem,
             max(0, xs.min() - margem):xs.max() + 1 + margem]


def conferir_bases(itens, bases, aceitos=()):
    grupos = {grupo(p) for p, _, _ in itens}
    for b in bases:
        with open(os.path.join(b, "split.txt"), encoding="utf-8") as f:
            vistos = {grupo(l.rstrip("\n").split(",", 2)[2]) for l in f if l.count(",") >= 2}
        inter = (grupos & vistos) - set(aceitos)
        if inter:
            raise SystemExit(f"VAZAMENTO: {len(inter)} grupos avaliados estao na base {b}: {sorted(inter)[:20]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "teste"), required=True)
    ap.add_argument("--modelo", action="append", required=True, help="rotulo=ckpt (.pt do EMA)")
    ap.add_argument("--bases", nargs="*", default=[], help="bases de treino para conferir vazamento")
    ap.add_argument("--aceitar_grupos", nargs="*", default=[],
                    help="grupos presentes nas bases aceitos apos conferencia manual (ficam no resumo.json)")
    ap.add_argument("--leitor", required=True, help="leitor CTC separado (treinar_alinhador.py --cabeca lstm)")
    ap.add_argument("--style", default=os.path.join(RAIZ, "DiffusionPen/style_models/iam_style_diffusionpen.pth"))
    ap.add_argument("--saida", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from PIL import Image
    from comum.diffusionpen import EscritoresIAM, GeradorDiffusionPen

    voc = Vocabulario()
    itens = escolher_palavras(voc, a.split)
    voc.conferir([p for p, _, _ in itens], a.split, f"avaliacao ({a.split})")
    conferir_bases(itens, a.bases, a.aceitar_grupos)

    treino = EscritoresIAM("iam_train_val.txt")
    teste = EscritoresIAM("iam_test.txt")
    comuns = set(treino.escritores) & set(teste.escritores)
    if comuns:
        raise SystemExit(f"escritores de teste tambem no treino: {sorted(comuns)[:10]}")
    escritores = random.Random(0).sample(teste.escritores, N_ESCRITORES)

    leitor = alinhamento.Alinhador(a.leitor, "cpu")
    os.makedirs(a.saida, exist_ok=True)
    linhas = ["modelo\tpalavra\ttipo\tescritor\tsemente\tmarca\tleitura\tcer"]
    res = defaultdict(list)          # (modelo, tipo) -> [(marca, cer)]
    pares = defaultdict(dict)        # modelo -> {(esqueleto, escritor, semente, tipo): marca}
    for m in a.modelo:
        rot, ckpt = m.split("=", 1)
        ger = GeradorDiffusionPen(ckpt, a.style, a.device)
        pasta = os.path.join(a.saida, "paineis", rot)
        os.makedirs(pasta, exist_ok=True)
        for k, (palavra, tipo, esq) in enumerate(itens):
            for s in SEMENTES:
                # mesma semente e mesmos escritores para todas as palavras e
                # todos os modelos: o par acentuada/esqueleto difere so no texto
                refs = [teste.referencias(e, random.Random(s * 1000 + j)) for j, e in enumerate(escritores)]
                imgs = ger.gerar([palavra] * len(escritores), refs, s)
                for e, img in zip(escritores, imgs):
                    g = (img.mean(0).numpy() * 255).astype(np.float32)
                    ac, ab, _, _ = marcas(g)
                    marca = int(ac + ab > 0)
                    r = recorte_tinta(g)
                    lido = leitor.ler(r) if r is not None else ""
                    cer = levenshtein(lido, esq) / max(1, len(esq))
                    linhas.append(f"{rot}\t{palavra}\t{tipo}\t{e}\t{s}\t{marca}\t{lido}\t{cer:.3f}")
                    res[(rot, tipo)].append((marca, cer))
                    pares[rot][(esq, e, s, tipo)] = marca
                    Image.fromarray(g.astype(np.uint8)).save(os.path.join(pasta, f"{k:03d}_{e}_{s}.png"))
            print(f"  {rot}: {k + 1}/{len(itens)} {palavra}", flush=True)
        del ger

    with open(os.path.join(a.saida, "paineis.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    rots = list(dict.fromkeys(m.split("=", 1)[0] for m in a.modelo))
    resumo = {"split": a.split, "palavras": [p for p, _, _ in itens], "escritores": escritores,
              "grupos_aceitos_nas_bases": a.aceitar_grupos,
              "sementes": list(SEMENTES), "modelos": {}}
    tab = ["| modelo | marca: acentuada | marca: esqueleto | marca: sem acento | "
           "diferenca pareada | CER acentuada | CER esqueleto | CER sem acento |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rots:
        d = {}
        for t in ("acentuada", "esqueleto", "sem_acento"):
            v = res[(r, t)]
            d[t] = {"n": len(v), "marca": float(np.mean([x[0] for x in v])),
                    "cer": float(np.mean([x[1] for x in v]))}
        dif = [pares[r][(k[0], k[1], k[2], "acentuada")] - pares[r][(k[0], k[1], k[2], "esqueleto")]
               for k in pares[r] if k[3] == "acentuada"]
        d["diferenca_pareada"] = float(np.mean(dif))
        resumo["modelos"][r] = d
        tab.append(f"| {r} | {d['acentuada']['marca']:.0%} | {d['esqueleto']['marca']:.0%} | "
                   f"{d['sem_acento']['marca']:.0%} | {d['diferenca_pareada']:+.0%} | "
                   f"{d['acentuada']['cer']:.2f} | {d['esqueleto']['cer']:.2f} | {d['sem_acento']['cer']:.2f} |")
    texto = "\n".join(tab)
    print(texto)
    with open(os.path.join(a.saida, "resumo.md"), "w", encoding="utf-8") as f:
        f.write(f"Split {a.split}; {len(escritores)} escritores do iam_test x {len(SEMENTES)} sementes.\n\n{texto}\n")
    with open(os.path.join(a.saida, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
