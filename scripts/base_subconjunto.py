"""Recorte de uma base de acentos para uma maquina com o IAM incompleto.

    python scripts/base_subconjunto.py --origem iam_pt_alinhado --saida iam_pt_sub

O leitor iam_acentuado tira as referencias de estilo das palavras REAIS do
escritor de cada amostra e, com iam_originais > 0, sorteia originais de todo o
iam_train_val. Numa maquina em que so parte do IAM foi extraida, qualquer
imagem ausente derruba o treino. Este script monta uma base derivada que
funciona com dados.iam_originais 0.0:

  - so as amostras da base de origem cujos escritores tem TODAS as imagens
    do iam_train_val no disco (as referencias de estilo deles existem);
  - mais N palavras originais desses mesmos escritores, copiadas para dentro
    da base (--originais, por padrao uma por amostra da base, a proporcao
    aproximada dos treinos com 30% de originais).

As imagens da base de origem nao sao copiadas: `imagens` vira um link para a
pasta de origem. O manifesto e filtrado junto, entao os pares e a mascara do
acento continuam valendo. O resumo.json NAO aponta o vocabulario (as palavras
originais sao inglesas e reprovariam na conferencia); os rotulos portugueses
ja foram conferidos quando a base de origem foi gerada.
"""
import argparse
import json
import os
import random
import shutil

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE = os.path.join(RAIZ, "DiffusionPen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origem", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--originais", type=int, default=-1, help="-1 = uma por amostra da base")
    ap.add_argument("--tipos", nargs="*", default=None,
                    help="tipos do manifesto a manter (acentuada, par, sem_acento); padrao = todos; vazio = nenhum")
    ap.add_argument("--total", type=int, default=0,
                    help="completa com originais ate este numero de amostras (ignora --originais); as originais "
                         "sao LINKS para o IAM, e as primeiras sao as mesmas da base sem --total")
    ap.add_argument("--semente", type=int, default=0)
    a = ap.parse_args()

    imagens = os.environ.get("IAM_IMAGES", os.path.join(CLONE, "iam_data", "words"))
    por_escritor = {}
    with open(os.path.join(CLONE, "utils", "splits_words", "iam_train_val.txt"), encoding="utf-8") as f:
        for l in f:
            p = l.rstrip("\n").split(",")
            if len(p) >= 3:
                por_escritor.setdefault(p[1], []).append((p[0], ",".join(p[2:])))
    completos = {e for e, v in por_escritor.items() if all(os.path.isfile(os.path.join(imagens, c)) for c, _ in v)}

    with open(os.path.join(a.origem, "split.txt"), encoding="utf-8") as f:
        linhas = [l.rstrip("\n") for l in f if l.count(",") >= 2]
    mantidas = [l for l in linhas if l.split(",")[1] in completos]
    n_todas = len(mantidas)                      # antes do filtro de tipos: define as originais "de sempre"
    if a.tipos is not None:
        with open(os.path.join(a.origem, "manifesto.jsonl"), encoding="utf-8") as f:
            tipo_de = {x["arquivo"]: x.get("tipo") for x in map(json.loads, f)}
        mantidas = [l for l in mantidas if tipo_de.get(l.split(",")[0]) in a.tipos]
    arquivos = {l.split(",")[0] for l in mantidas}

    os.makedirs(os.path.join(a.saida, "originais"), exist_ok=True)
    link = os.path.join(a.saida, "imagens")
    if not os.path.islink(link):
        os.symlink(os.path.abspath(os.path.join(a.origem, "imagens")), link)

    cand = sorted((e, c, t) for e in completos for c, t in por_escritor[e])
    if a.total:
        # as primeiras n_todas sao as mesmas do recorte padrao; o resto vem em ordem sorteada
        base = random.Random(a.semente).sample(cand, min(n_todas, len(cand)))
        resto = [x for x in cand if x not in set(base)]
        random.Random(a.semente + 1).shuffle(resto)
        escolhidas = (base + resto)[:max(0, a.total - len(mantidas))]
    else:
        n = len(mantidas) if a.originais < 0 else a.originais
        escolhidas = random.Random(a.semente).sample(cand, min(n, len(cand)))
    novas = []
    for i, (e, c, t) in enumerate(escolhidas):
        nome = f"originais/{i:05d}.png"
        destino = os.path.join(a.saida, nome)
        if a.total:
            if not os.path.islink(destino):
                os.symlink(os.path.abspath(os.path.join(imagens, c)), destino)
        else:
            shutil.copyfile(os.path.join(imagens, c), destino)
        novas.append(f"{nome},{e},{t}")

    with open(os.path.join(a.saida, "split.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(mantidas + novas) + "\n")
    tipos = {}
    with open(os.path.join(a.origem, "manifesto.jsonl"), encoding="utf-8") as f, \
            open(os.path.join(a.saida, "manifesto.jsonl"), "w", encoding="utf-8") as g:
        for l in f:
            x = json.loads(l)
            if x["arquivo"] in arquivos:
                g.write(l)
                tipos[x.get("tipo", "?")] = tipos.get(x.get("tipo", "?"), 0) + 1
    with open(os.path.join(a.origem, "resumo.json"), encoding="utf-8") as f:
        origem = json.load(f)
    with open(os.path.join(a.saida, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump({"descricao": f"Recorte de {a.origem} para maquina com IAM incompleto: so escritores com todas "
                                f"as imagens do iam_train_val no disco, mais palavras originais deles",
                   "origem": a.origem, "escritores_completos": len(completos),
                   "amostras_da_base": len(mantidas), "tipos": tipos, "originais": len(novas),
                   "semente": a.semente, "pares_alinhados": bool(origem.get("pares_alinhados"))},
                  f, ensure_ascii=False, indent=1)
    print(f"{a.saida}: {len(mantidas)} amostras de {a.origem} ({tipos}) + {len(novas)} originais, "
          f"{len(completos)} escritores completos")


if __name__ == "__main__":
    main()
