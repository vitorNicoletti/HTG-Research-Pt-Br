from PIL import Image
from pathlib import Path
import collections
import json
import numpy as np
import random
import re
import statistics
import sys

#  ajuste aqui
BRESSAY = Path("./bressay")
SAIDA = Path("./bressay_split")

# min 11 p10 17 mediana 25 max 60
CORTE = 0
ALT_MIN, LARG_MIN = 28, 8
PX_CHAR_MIN = 0
SEED = 42

WORDS = BRESSAY / "data" / "words"
MARCACOES = re.compile(r"(@@|##|\$\$|--)")
DIAC = set("àáâãçéêíóôõúüÀÁÂÃÇÉÊÍÓÔÕÚÜ")
TINTA_MIN = 0.1


def ler_texto(p):
    for enc in ("utf-8", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return None

def ler_particao(nome):
    f = BRESSAY / "sets" / nome
    return set(ler_texto(f).split()) if f.exists() else set()

def conferir_ids(paginas):
    """Valida os nomes de pasta contra os arquivos de particao oficiais."""
    sets_dir = BRESSAY / "sets"
    oficial = set()
    for nome in ("training.txt", "validation.txt", "test.txt"):
        f = sets_dir / nome
        if f.exists():
            oficial |= set(ler_texto(f).split())

    so_disco, so_sets = paginas - oficial, oficial - paginas
    print(f"[ids] pastas em words/ : {len(paginas)}")
    print(f"[ids] IDs em sets/*.txt: {len(oficial)}")
    if not so_disco and not so_sets:
        print("[ids] conferem exatamente\n")
    else:
        print(f"[ids] !! so no disco: {len(so_disco)} {sorted(so_disco)[:5]}")
        print(f"[ids] !! so nos sets: {len(so_sets)} {sorted(so_sets)[:5]}\n")


def main():
    random.seed(SEED)
    if not WORDS.is_dir():
        sys.exit(f"nao encontrei {WORDS}")

    paginas = {d.name: d for d in sorted(WORDS.iterdir()) if d.is_dir()}
    conferir_ids(set(paginas))

    if CORTE > 0:
        medianas = {}
        for pg, d in paginas.items():
            hs = []
            for png in d.glob("*.png"):
                try:
                    hs.append(Image.open(png).size[1])
                except Exception:
                    pass
            if hs:
                medianas[pg] = statistics.median(hs)
        selecionadas = {pg: paginas[pg] for pg, m in medianas.items() if m >= CORTE}
        print(
            f"[corte] mediana >= {CORTE}px: {len(selecionadas)} de {len(paginas)} paginas"
        )
    else:
        selecionadas = paginas
        print(f"[corte] desligado: {len(selecionadas)} paginas")

    # transcricao + verificacao de qualidade
    por_escritor = collections.defaultdict(list)
    descartados, n_marcacao = [], 0

    for pg, d in selecionadas.items():
        for png in sorted(d.glob("*.png")):
            txt = png.with_suffix(".txt")
            if not txt.exists():
                descartados.append((png, "", "sem .txt"))
                continue
            t = (ler_texto(txt) or "").strip()
            if not t:
                descartados.append((png, "", "transcricao vazia"))
                continue
            if MARCACOES.search(t):
                descartados.append((png, t, f"marcacao"))
                n_marcacao += 1
                continue
            try:
                im = Image.open(png)
                w, h = im.size
            except Exception as e:
                descartados.append((png, t, f"erro ao abrir: {e}"))
                continue

            px_por_char = w / max(len(t), 1)
            if h < ALT_MIN or w < LARG_MIN: #or px_por_char < PX_CHAR_MIN:
                descartados.append((png, t, f"baixa densidade {w}x{h} len={len(t)}"))
                continue

            if len(t) < 2:
                descartados.append((png, t, f"menos de 2 letras"))
                continue

            g = np.asarray(im.convert("L"), dtype=np.float32)
            lo, hi = np.percentile(g, 3), np.percentile(g, 40)
            if hi - lo < 8:
                descartados.append((png, t, "imagem plana demais"))
                continue

            g = np.clip((g - lo) / (hi - lo), 0, 1) * 255
            frac = float((g < 128).mean())
            if not (TINTA_MIN <= frac):
                descartados.append((png, t, f"tinta fora da faixa ({frac:.4f})"))
                continue

            por_escritor[pg].append((f"{pg}/{png.name}", t))

    if not por_escritor:
        sys.exit("nenhuma palavra sobreviveu aos filtros")

    # saidas
    (SAIDA / "splits").mkdir(parents=True, exist_ok=True)
    wdict = {pg: i for i, pg in enumerate(sorted(por_escritor))}
    (SAIDA / "writers_dict.json").write_text(
        json.dumps(wdict, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    pg_train = ler_particao("training.txt")
    pg_val   = ler_particao("validation.txt")
    pg_test  = ler_particao("test.txt")

    treino, val, teste = [], [], []
    for pg, itens in por_escritor.items():
        destino = treino if pg in pg_train else val if pg in pg_val else teste if pg in pg_test else None
        if destino is None:
            continue
        destino += [(c, wdict[pg], t) for c, t in itens]

    for nome, dados in (("train.tsv", treino), ("val.tsv", val), ("test.tsv", teste)):
        random.shuffle(dados)
        with open(SAIDA / "splits" / nome, "w", encoding="utf-8") as f:
            for c, wid, t in dados:
                f.write(f"{c}\t{wid}\t{t}\n")

    with open(SAIDA / "descartados.tsv", "w", encoding="utf-8") as f:
        for png, t, motivo in descartados:
            f.write(f"{png}\t{t}\t{motivo}\n")

    # relatorio
    tot = len(treino) + len(val) + len(teste)
    ac = sum(1 for _, _, t in treino + val + teste if any(c in DIAC for c in t))
    print(f"treino / val / teste : {len(treino)} / {len(val)} / {len(teste)}")
    porescr = sorted(len(v) for v in por_escritor.values())
    longas = sum(1 for _, _, t in treino + val if len(t) > 3)

    for m, n in collections.Counter(
        m.split("(")[0].strip() for _, _, m in descartados
    ).most_common():
        print(f"    {m}: {n}")
    print(
        f"palavras por escritor: min={porescr[0]} "
        f"mediana={porescr[len(porescr)//2]} max={porescr[-1]}"
    )
    print(f"    escritores com <30: {sum(1 for n in porescr if n < 30)}")

    print(f"\nescritores           : {len(wdict)}   (style_classes)")
    print(f"treino/validacao/teste : {len(treino)} / {len(val)} / {len(teste)}")
    print(f"com diacritico       : {ac} ({100*ac/tot:.1f}%)")
    print(f"com mais de 3 chars  : {longas} ({100*longas/tot:.1f}%)")
    print(f"pulados por marcacao : {n_marcacao}")
    print(f"descartados          : {len(descartados)}")

if __name__ == "__main__":
    main()
