"""Figura: os 4 passos do E1 numa imagem real."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "avaliacao_diacriticos")
import metrica as M, e1
from teste_sensibilidade_diff import pinta_acento

D = "avaliacao_diacriticos/ger_iam_n696"
itens = [json.loads(l) for l in open(f"{D}/manifest.jsonl", encoding="utf-8")]
gem = {(x["par_id"], x["escritor"], x["semente"]): x for x in itens if not x["acentuada"]}

# escolhe um par de TIL com bastante tinta (imagem menos degenerada)
alvo = None
for r in itens:
    if not r["acentuada"] or "til" not in [n for _, n, _ in M.diacriticos(r["palavra"])]:
        continue
    g = gem.get((r["par_id"], r["escritor"], r["semente"]))
    if g and r["std"] > 0.20:
        alvo = (r, g); break
r, g = alvo
acc = M.carregar_tinta(f"{D}/{r['arquivo']}")
asc = M.carregar_tinta(f"{D}/{g['arquivo']}")
idx, _, onde = [d for d in M.diacriticos(r["palavra"]) if d[1] == "til"][0]
pint = pinta_acento(asc, r["palavra"], idx, onde, 1.0)

a, b = e1.tinta(acc), e1.tinta(asc)
p = e1.tinta(pint)
perfil = b.sum(axis=1)
corpo = np.where(perfil >= 0.5 * perfil.max())[0]
topo, base = corpo[0], corpo[-1] + 1
cols = np.where(b.any(axis=0))[0]
larg = (cols[-1] + 1 - cols[0]) / len(M.sem_acento(r["palavra"]))
c0 = max(0, int(cols[0] + (idx - 0.5) * larg)); c1 = int(cols[0] + (idx + 1.5) * larg)

e_real = e1.e1(acc, asc, r["palavra"])[[d[1] for d in M.diacriticos(r["palavra"])].index("til")]
e_pint = e1.e1(pint, asc, r["palavra"])[[d[1] for d in M.diacriticos(r["palavra"])].index("til")]

fig, ax = plt.subplots(6, 1, figsize=(9, 9.2))
fig.suptitle(f'E1 passo a passo — par mínimo "{r["palavra"]}" / "{M.sem_acento(r["palavra"])}"\n'
             f'mesmo escritor, mesma semente: a única variável é o texto',
             fontsize=13, y=0.985)

def mostra(k, img, titulo, binaria=False):
    ax[k].imshow(1 - img if binaria else 1 - img, cmap="gray", vmin=0, vmax=1,
                 interpolation="nearest")
    ax[k].set_title(titulo, fontsize=10, loc="left", pad=4)
    ax[k].set_xticks([]); ax[k].set_yticks([])

mostra(0, acc, f'1. o que o modelo gerou para "{r["palavra"]}"')
mostra(1, asc, f'1. e para "{M.sem_acento(r["palavra"])}" — a gêmea')

mostra(2, b.astype(float), "2. tinta (Otsu) e corpo da letra: altura-x e linha de base")
for y, cor, rot in ((topo, "tab:blue", "altura-x"), (base, "tab:red", "linha de base")):
    ax[2].axhline(y, color=cor, lw=1.4)
    ax[2].text(258, y, f" {rot}", color=cor, fontsize=8, va="center")

car = M.sem_acento(r["palavra"])[idx]
mostra(3, b.astype(float),
       f"3. só a faixa acima da altura-x, na coluna do {idx+1}º caractere (o '{car}' de '{r['palavra']}')")
ax[3].add_patch(Rectangle((c0, -0.5), c1 - c0, topo + 0.5, ec="tab:green",
                          fc="tab:green", alpha=0.25, lw=1.8))

def diferenca(k, mask_acc, titulo):
    """Palavra em cinza claro ao fundo; vermelho = tinta a mais, azul = a menos."""
    d = mask_acc.astype(int) - b.astype(int)
    img = np.ones((*b.shape, 3))
    img[b] = 0.80                                  # a palavra, de referencia
    img[d > 0] = (0.85, 0.10, 0.10)
    img[d < 0] = (0.10, 0.30, 0.85)
    ax[k].imshow(img, interpolation="nearest")
    ax[k].set_title(titulo, fontsize=9.5, loc="left", pad=4)
    ax[k].add_patch(Rectangle((c0, -0.5), c1 - c0, topo + 0.5,
                              ec="tab:green", fc="none", lw=1.8))
    ax[k].set_xticks([]); ax[k].set_yticks([])

diferenca(4, a, f"4. diferença real (vermelho = tinta a mais, azul = a menos)"
              f"    E1 = {e_real:+.3f}")
diferenca(5, p, f"5. o mesmo par, com um til nominal pintado"
              f"    E1 = {e_pint:+.3f}")

fig.tight_layout(rect=[0, 0.055, 1, 0.945])
fig.text(0.5, 0.006,
         "O retângulo verde é a única região que conta. Na linha 4 não sobra tinta "
         "vermelha dentro dele: o modelo do IAM não desenhou o til.\n"
         "Repare que o 'd' tem haste alta nas DUAS imagens, então ele some sozinho "
         "na subtração — é por isso que ascendente não vira falso positivo.",
         ha="center", fontsize=9, style="italic")
fig.savefig("avaliacao_diacriticos/figura_e1.png", dpi=130)
print("palavra:", r["palavra"], "| E1 real:", round(e_real, 4), "| E1 pintado:", round(e_pint, 4))
print("salvo: avaliacao_diacriticos/figura_e1.png")
