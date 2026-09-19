"""
testes_metrica.py -- testes de resposta conhecida para a geometria do E1.

Nao usa o gerador: desenha figuras sinteticas em que a resposta certa e obvia
por construcao. Servem para separar "a metrica esta errada" de "o gerador
esta ruim" quando os numeros reais vierem baixos.

    python avaliacao_diacriticos/testes_metrica.py
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrica as M  # noqa: E402

H, W = 64, 256
falhas = []


def branco():
    return np.zeros((H, W), dtype=np.float32)


def caixa(img, y0, y1, x0, x1, v=1.0):
    img[y0:y1, x0:x1] = v


def corpo(n_chars, y_topo=30, y_base=46, x0=20, larg=30):
    """n_chars blocos solidos lado a lado = o corpo da palavra."""
    img = branco()
    for i in range(n_chars):
        caixa(img, y_topo, y_base, x0 + i * larg + 4, x0 + i * larg + larg - 4)
    return img


def checa(nome, cond, detalhe=""):
    print(("  OK   " if cond else "  FALHA") + f" {nome} {detalhe}")
    if not cond:
        falhas.append(nome)


print("1. linha_base_e_altura_x recupera o corpo desenhado")
img = corpo(5, y_topo=30, y_base=46)
yx, yb = M.linha_base_e_altura_x(M.binariza(img))
checa("altura-x = 30", yx == 30, f"(obtido {yx})")
checa("linha de base = 46", yb == 46, f"(obtido {yb})")

print("\n2. ascendente NAO conta como acento (coluna errada)")
# palavra "dacao": 'd' no indice 0 tem haste alta; o til estaria no indice 2
img = corpo(5)
caixa(img, 10, 30, 24, 30)          # haste do 'd', coluna 0
r = M.e1_por_faixa(img, "dacao".replace("a", "a"))   # sem acento: nada a medir
checa("palavra sem acento -> lista vazia", r == [], f"(obtido {r})")
# agora pede o til no indice 2, onde nao ha nada acima
img2 = corpo(5)
caixa(img2, 10, 30, 24, 30)         # ascendente na coluna 0
r = M.e1_por_faixa(img2, "dacão".replace("dac", "dac"))
alvo = [d for d in r if d["marca"] == "til"]
checa("til pedido no indice 3 com ascendente no 0 -> massa 0",
      alvo and alvo[0]["massa"] == 0, f"(obtido {alvo})")

print("\n3. til desenhado na coluna certa E detectado")
img3 = corpo(5)
caixa(img3, 10, 30, 24, 30)         # ascendente na coluna 0 (ruido)
caixa(img3, 20, 26, 20 + 2 * 30 + 6, 20 + 2 * 30 + 24)   # til sobre o indice 2
r = M.e1_por_faixa(img3, "dacão".replace("dac", "dac"))
alvo = [d for d in r if d["marca"] == "til"][0]
checa("til no indice 3 -> massa > 0", alvo["massa"] > 0, f"(massa {alvo['massa']})")

print("\n4. cedilha e procurada ABAIXO da linha de base")
img4 = corpo(5)
caixa(img4, 46, 56, 20 + 2 * 30 + 8, 20 + 2 * 30 + 16)   # cedilha sob o indice 2
r = M.e1_por_faixa(img4, "daçao")
alvo = [d for d in r if d["marca"] == "cedilha"][0]
checa("cedilha detectada abaixo da base", alvo["massa"] > 0, f"(massa {alvo['massa']})")
img5 = corpo(5)
caixa(img5, 20, 30, 20 + 2 * 30 + 8, 20 + 2 * 30 + 16)   # tinta ACIMA, nao abaixo
r = M.e1_por_faixa(img5, "daçao")
alvo = [d for d in r if d["marca"] == "cedilha"][0]
checa("tinta acima nao conta como cedilha", alvo["massa"] == 0, f"(massa {alvo['massa']})")

print("\n5. e1_por_diff isola o acento do que os dois gemeos ja tinham")
asc = corpo(5)
caixa(asc, 10, 30, 24, 30)                               # ascendente nos DOIS
acc = asc.copy()
caixa(acc, 20, 26, 20 + 2 * 30 + 6, 20 + 2 * 30 + 24)    # so a acentuada tem til
r = M.e1_por_diff(acc, asc, "dacão".replace("dac", "dac"))
alvo = [d for d in r if d["marca"] == "til"][0]
checa("delta > 0 quando so a acentuada tem til", alvo["delta"] > 0, f"(delta {alvo['delta']})")
checa("acima_do_topo > 0", alvo["acima_do_topo"] > 0, f"({alvo['acima_do_topo']})")
r = M.e1_por_diff(asc, asc, "dacão".replace("dac", "dac"))
alvo = [d for d in r if d["marca"] == "til"][0]
checa("delta = 0 quando os gemeos sao iguais", alvo["delta"] == 0, f"(delta {alvo['delta']})")

print("\n6. o caso do pingo do i: agudo tem que ter MAIS massa que o pingo")
asc = corpo(4)
caixa(asc, 22, 28, 20 + 2 * 30 + 12, 20 + 2 * 30 + 18)   # pingo do i, indice 2
acc = corpo(4)
caixa(acc, 16, 28, 20 + 2 * 30 + 8, 20 + 2 * 30 + 22)    # agudo, maior
r = M.e1_por_diff(acc, asc, "país")
alvo = [d for d in r if d["marca"] == "agudo"][0]
checa("delta > 0 (agudo maior que o pingo)", alvo["delta"] > 0,
      f"(massa_acc {alvo['massa_acc']} vs massa_asc {alvo['massa_asc']})")
r = M.e1_por_diff(asc, asc, "país")
alvo = [d for d in r if d["marca"] == "agudo"][0]
checa("delta = 0 se a acentuada so tem o pingo (acento omitido)",
      alvo["delta"] == 0, f"(delta {alvo['delta']})")

print("\n7. imagem em branco nao gera acento fantasma")
r = M.e1_por_faixa(branco(), "não")
checa("imagem vazia -> nao medivel", not r[0]["medivel"], f"({r[0]})")

print("\n8. CER e dobra ASCII")
checa("cer identico = 0", M.cer("nacao", "nacao") == 0.0)
checa("cer 1 substituicao em 5 = 0.2", abs(M.cer("nacao", "nocao") - 0.2) < 1e-9)
checa("dobra tira acento e caixa", M.dobra_ascii("Ação!") == "acao")
checa("CER com dobra ignora o acento",
      M.cer(M.dobra_ascii("ação"), M.dobra_ascii("acao")) == 0.0)

print("\n" + ("TODOS OS TESTES PASSARAM" if not falhas
               else f"FALHARAM: {falhas}"))
sys.exit(1 if falhas else 0)
