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
checa("delta > 0 quando so a acentuada tem til", alvo["delta_rel"] > 0,
      f"(delta_rel {alvo['delta_rel']})")
r = M.e1_por_diff(asc, asc, "dacão".replace("dac", "dac"))
alvo = [d for d in r if d["marca"] == "til"][0]
checa("delta = 0 quando os gemeos sao iguais", alvo["delta_rel"] == 0,
      f"(delta_rel {alvo['delta_rel']})")

print("\n6. o caso do pingo do i: agudo tem que ter MAIS massa que o pingo")
asc = corpo(4)
caixa(asc, 22, 28, 20 + 2 * 30 + 12, 20 + 2 * 30 + 18)   # pingo do i, indice 2
acc = corpo(4)
caixa(acc, 16, 28, 20 + 2 * 30 + 8, 20 + 2 * 30 + 22)    # agudo, maior
r = M.e1_por_diff(acc, asc, "país")
alvo = [d for d in r if d["marca"] == "agudo"][0]
checa("delta > 0 (agudo maior que o pingo)", alvo["delta_rel"] > 0,
      f"(delta_rel {alvo['delta_rel']})")
r = M.e1_por_diff(asc, asc, "país")
alvo = [d for d in r if d["marca"] == "agudo"][0]
checa("delta = 0 se a acentuada so tem o pingo (acento omitido)",
      alvo["delta_rel"] == 0, f"(delta_rel {alvo['delta_rel']})")

print("\n7. gemeo fora de lugar nao vira acento")
base = corpo(5)
caixa(base, 10, 30, 24, 30)
movida = branco()
movida[:, 6:] = base[:, :-6]           # a MESMA palavra, 6 px para a direita
r = M.e1_por_diff(movida, base, "dacão".replace("dac", "dac"))
alvo = [d for d in r if d["marca"] == "til"][0]
checa("mesma palavra deslocada 6 px -> delta 0", abs(alvo["delta_rel"]) < 0.02,
      f"(delta_rel {alvo['delta_rel']})")

print("\n8. imagem em branco nao gera acento fantasma")
r = M.e1_por_faixa(branco(), "não")
checa("imagem vazia -> nao medivel", not r[0]["medivel"], f"({r[0]})")

print("\n9. CER e dobra ASCII")
checa("cer identico = 0", M.cer("nacao", "nacao") == 0.0)
checa("cer 1 substituicao em 5 = 0.2", abs(M.cer("nacao", "nocao") - 0.2) < 1e-9)
checa("dobra tira acento e caixa", M.dobra_ascii("Ação!") == "acao")
checa("CER com dobra ignora o acento",
      M.cer(M.dobra_ascii("ação"), M.dobra_ascii("acao")) == 0.0)

print("\n9. linha pautada do papel")
# palavra "dacao" com til no indice 2, MAIS a pauta atravessando a imagem
base = corpo(5)
caixa(base, 20, 26, 20 + 2 * 30 + 6, 20 + 2 * 30 + 24)    # til
r_sem = M.e1_por_faixa(base, "dacão".replace("dac", "dac"))[0]

pautada = base.copy()
caixa(pautada, 46, 49, 0, W)                              # pauta de 3 px na base
r_com = M.e1_por_faixa(pautada, "dacão".replace("dac", "dac"))[0]
r_cru = M.e1_por_faixa(pautada, "dacão".replace("dac", "dac"),
                       tirar_pauta=False)[0]
checa("pauta detectada", M.tem_pauta(M.binariza(pautada)))
checa("sem pauta nao dispara falso positivo",
      not M.tem_pauta(M.binariza(base)))
checa("com remocao, o escore volta ao valor sem pauta",
      abs(r_com["massa_rel"] - r_sem["massa_rel"]) < 0.02,
      f"(sem={r_sem['massa_rel']:.3f} com={r_com['massa_rel']:.3f})")
checa("SEM remocao o escore se distorce",
      abs(r_cru["massa_rel"] - r_sem["massa_rel"]) > 0.02,
      f"(sem={r_sem['massa_rel']:.3f} cru={r_cru['massa_rel']:.3f})")
checa("a geometria volta ao corpo desenhado com a remocao",
      M.linha_base_e_altura_x(M.mascara_de_tinta(pautada)[0]) == (30, 46),
      f"(obtido {M.linha_base_e_altura_x(M.mascara_de_tinta(pautada)[0])})")

print("\n10. na cedilha a pauta ESCONDE o acento, nao o inventa")
# A faixa da cedilha e abaixo da linha de base. Sem remover a pauta, ela vira
# a fileira mais cheia e puxa a linha de base para baixo, de modo que a
# cedilha fica DENTRO do corpo estimado e deixa de ser contada. E o oposto do
# que acontece com as marcas de cima, que inflam. Bate com o controle real:
# a cedilha foi a unica marca que ficou MENOR com pauta (0.214) do que sem
# pauta (0.302), enquanto agudo e til inflaram.
ped = corpo(5)
caixa(ped, 47, 55, 20 + 2 * 30 + 8, 20 + 2 * 30 + 16)     # cedilha de verdade
r_sem = M.e1_por_faixa(ped, "daçao")[0]
checa("cedilha sem pauta e detectada", r_sem["massa"] > 0,
      f"(massa {r_sem['massa']})")

pautada2 = ped.copy()
caixa(pautada2, 46, 49, 0, W)                             # pauta sobre a base
r_com = M.e1_por_faixa(pautada2, "daçao")[0]
r_cru = M.e1_por_faixa(pautada2, "daçao", tirar_pauta=False)[0]
checa("com remocao a cedilha continua detectada", r_com["massa"] > 0,
      f"(massa {r_com['massa']})")
# O mecanismo: sem remover a pauta, o "corpo" estimado colapsa sobre a
# propria pauta e a linha de base desce ate ela. O quanto isso muda a massa
# contada depende de onde a cedilha cai em relacao a base deslocada, entao a
# assercao aqui e sobre a GEOMETRIA, que e a causa, e nao sobre a massa, que e
# um efeito que varia caso a caso.
geo_cru = M.linha_base_e_altura_x(M.binariza(pautada2))
geo_ok = M.linha_base_e_altura_x(M.mascara_de_tinta(pautada2)[0])
checa("SEM remocao a linha de base desce ate a pauta",
      geo_cru[1] >= 46 and (geo_cru[1] - geo_cru[0]) < 8,
      f"(corpo cru={geo_cru}, {geo_cru[1]-geo_cru[0]} px)")
checa("COM remocao a linha de base volta ao corpo real",
      geo_ok == (30, 46), f"(corpo corrigido={geo_ok})")

# e o caso simetrico: pauta sozinha nao pode inventar cedilha
so_pauta = corpo(5)
caixa(so_pauta, 46, 49, 0, W)
checa("pauta sozinha nao vira cedilha",
      M.e1_por_faixa(so_pauta, "daçao")[0]["massa"] == 0)

print("\n11. traco de letra grosso NAO e confundido com pauta")
# 7 fileiras cheias = corpo de palavra curta, nao pauta (medido em "que"/"para")
grosso = branco()
caixa(grosso, 30, 46, 100, 160)          # bloco solido: cobre 100% da bbox
checa("bloco de 16 px nao e pauta", not M.tem_pauta(M.binariza(grosso)))
fino = branco()
caixa(fino, 30, 46, 100, 160)
caixa(fino, 50, 53, 90, 170)             # pauta fina, mais larga que a letra
checa("banda de 3 px e pauta", M.tem_pauta(M.binariza(fino)))
m2, n = M.remover_pauta(M.binariza(fino))
checa("remocao tira so as 3 fileiras da pauta", n == 3, f"(n={n})")
checa("o corpo da letra sobrevive", m2[30:46, 100:160].all())

print("\n" + ("TODOS OS TESTES PASSARAM" if not falhas
               else f"FALHARAM: {falhas}"))
sys.exit(1 if falhas else 0)
