"""
Cedilha pela tinta abaixo do c, numa imagem so.

A tentativa anterior (cedilha.py) comparava a acentuada com o esqueleto nas
colunas do esqueleto. No modelo gerado as duas imagens nao tem as letras no
mesmo lugar, e a cedilha de outra letra ou a palavra redesenhada entravam na
coluna do c.

Aqui nao ha par. Como no detector do colega (scripts/medir_marcas.py), o corpo
das letras vem de acentos_sinteticos.geometria, e as colunas de cada letra vem
do alinhador lendo o esqueleto na propria imagem acentuada
(scripts/medir_posicao_acento.py). A medida de cada c e a tinta abaixo de
base + TOL alturas-x nas colunas do c, dividida pela tinta do corpo do c nas
mesmas colunas. TOL e o do medir_marcas.

Com --medida altura (setima tentativa) a medida e outra: quantas fileiras
seguidas, a partir do corte para baixo, tem tinta nas colunas do c, divididas
pela altura-x. A cedilha e estreita e funda; um c inteiro um pouco mais baixo
que a palavra, ou com o traco de baixo grosso, e largo e raso. Fileiras
seguidas para que um pixel solto la embaixo nao conte.

Com --medida base_local (oitava tentativa) a medida e a altura, mas o corte
sai de uma linha de base inclinada: uma reta, por minimos quadrados, pela
fileira mais baixa com tinta de cada letra sem descendente (nem g, j, p, q,
y, f, z, nem os proprios c). O corte de cada c e essa reta no meio da fatia
dele mais TOL alturas-x. Um c inteiro mais baixo que as vizinhas deixa de
contar como tinta a mais.

Negativo e limiar como em cedilha.py: os c sem cedilha das acentuadas do
modelo de 40 epocas, limiar no p95 deles. Positivo: as cedilhas da base
iam_pt_alinhado; imagem que nao alinha conta como nao percebida.

Grava resultados/cedilha_abaixo_40ep.json e folhas do modelo de 40 epocas,
resultados/cedilha_abaixo_folha_{A,B,C,D}.png: acertos, perdidas, falsos
alarmes e c sem cedilha abaixo do limiar. Com --medida altura ou base_local os
nomes ganham o sufixo da medida e sai tambem a folha E, com as cedilhas pedidas que um limiar de 0,15
perderia. Os paineis sao numerados e listados no JSON. O JSON traz tambem os
limiares fixos testados a mao (LIMIARES_A_MAO).

    python avaliacao_diacriticos/experimentos/cedilha_abaixo.py [--medida altura|base_local]
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
RAIZ = os.path.dirname(MODULO)
sys.path.insert(0, MODULO)
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))

import e1  # noqa: E402
import e1_alinhador  # noqa: E402
from acentos_sinteticos import alinhamento, geometria  # noqa: E402
from anotacao_dupla import AVALIACAO, ler_paineis  # noqa: E402
from medir_marcas import TOL  # noqa: E402
from medir_posicao_acento import recorte  # noqa: E402
from utils.iam_acentuado_dataset import preprocessar_iam  # noqa: E402

BASE = os.path.join(RAIZ, "iam_pt_alinhado")
MEDIDA = "tinta"   # ou "altura" ou "base_local"; vem da linha de comando
# limiares testados a mao, fixados antes de rodar; nao saem da regra do p95
LIMIARES_A_MAO = {"tinta": [0.05, 0.10, 0.15, 0.20],
                  "altura": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                  "base_local": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]}
# letras que nao entram na linha de base local: descendentes e o proprio c
FORA_DA_BASE = set("gjpqyfzc")
# folha E: cedilhas pedidas que se perdem ao subir o limiar ate este valor
LIMIAR_FOLHA_E = 0.15


def fileiras_seguidas(bloco):
    """Quantas fileiras do topo do bloco para baixo tem tinta, sem buraco."""
    vazias = np.where(~bloco.any(axis=1))[0]
    return int(vazias[0]) if len(vazias) else bloco.shape[0]


def base_local(geo, fatias, letras):
    """Reta y = f(x) pelo fundo da tinta das letras sem descendente. Sem dois
    pontos, a linha de base horizontal da geometria."""
    xs, ys = [], []
    for (xa, xb), letra in zip(fatias, letras):
        if letra in FORA_DA_BASE:
            continue
        linhas = np.where(geo.mask[:, int(round(xa)):int(round(xb))].any(axis=1))[0]
        if len(linhas):
            xs.append((xa + xb) / 2)
            ys.append(linhas.max())
    if len(xs) < 2:
        return lambda x: geo.base
    inclinacao, altura = np.polyfit(xs, ys, 1)
    return lambda x: inclinacao * x + altura


def medidas_nos_c(g, letras):
    """Para cada c: (posicao, medida, desenho, fundo). None se nao alinhar.
    desenho = (xa, xb, linha de corte) em pixels do painel, para a folha;
    fundo = fileira mais baixa com tinta na coluna do c menos a linha de base
    usada ali, em pixels (positivo: o c desce abaixo da base)."""
    r, x0 = recorte(g)
    if r is None:
        return None
    geo = geometria.analisar(r)
    al = ALINHADOR.alinhar(r, letras) if geo is not None else None
    if al is None:
        return None
    fatias = geometria.ajustar_aos_vales(geo, alinhamento.fatias_do_alinhamento(al, geo))
    corte = int(round(geo.base + TOL * geo.altura_x)) + 1
    reta = base_local(geo, fatias, letras) if MEDIDA == "base_local" else None
    saida = []
    for k, (xa, xb) in enumerate(fatias):
        if letras[k] != "c":
            continue
        a, b = int(round(xa)), max(int(round(xb)), int(round(xa)) + 1)
        base_aqui = reta((xa + xb) / 2) if reta is not None else geo.base
        if reta is not None:
            corte = max(0, int(round(base_aqui + TOL * geo.altura_x)) + 1)
        linhas = np.where(geo.mask[:, a:b].any(axis=1))[0]
        fundo = float(linhas.max() - base_aqui) if len(linhas) else 0.0
        if MEDIDA in ("altura", "base_local"):
            medida = fileiras_seguidas(geo.mask[corte:, a:b]) / geo.altura_x
        else:
            abaixo = geo.mask[corte:, a:b].sum()
            corpo = geo.mask[geo.topo_x:geo.base + 1, a:b].sum()
            medida = float(abaixo) / corpo if corpo else 0.0
        # o recorte tira MARGEM px de cima; somar de volta para desenhar no painel
        y0 = int(np.where(geometria.mascara_tinta(g))[0].min()) - 3
        saida.append((k, medida, (a + x0, b + x0, corte + max(0, y0)), fundo))
    return saida


def medir_no_modelo():
    """c das acentuadas do modelo de 40 epocas, com cedilha pedida ou nao."""
    saida = []
    for linha in ler_paineis(AVALIACAO):
        letras = e1.sem_acento(linha["palavra"])
        if linha["tipo"] != "acentuada" or "c" not in letras:
            continue
        g = np.asarray(Image.open(linha["caminho"]).convert("L"), dtype=np.float32)
        pedidas = {p for p, acima in e1.diacriticos(linha["palavra"]) if not acima}
        for k, medida, desenho, fundo in medidas_nos_c(g, letras) or []:
            saida.append({"palavra": linha["palavra"], "posicao": k, "pedida": k in pedidas,
                          "medida": medida, "caminho": linha["caminho"], "desenho": desenho,
                          "fundo": fundo})
    return saida


def medir_na_base(tipo):
    """Letras com cedilha na base, no formato do treino. None = nao alinhou.
    tipo "acentuada": a cedilha desenhada; tipo "par": o mesmo c sem cedilha."""
    with open(os.path.join(BASE, "manifesto.jsonl"), encoding="utf-8") as f:
        itens = [json.loads(l) for l in f]
    saida = []
    for item in itens:
        if item["tipo"] != tipo or "ç" not in item.get("alvo", ""):
            continue
        imagem = preprocessar_iam(Image.open(os.path.join(BASE, item["arquivo"])).convert("RGB"),
                                  item["rotulo"])
        g = np.asarray(imagem.convert("L"), dtype=np.float32)
        letras = e1.sem_acento(item["rotulo"])
        medidas = {k: m for k, m, _, _ in medidas_nos_c(g, letras) or []}
        for k, letra in enumerate(item["alvo"]):
            if letra == "ç":
                saida.append(medidas.get(k))
    return saida


def folha(itens, letra, titulo, caminho):
    """Um painel por c, numerado (A1, A2...), com a fatia do c em azul e o
    corte em vermelho. Devolve a lista dos paineis para o JSON."""
    E = 2
    im = Image.new("RGB", (2 * (256 * E + 8), 20 + (len(itens) + 1) // 2 * (64 * E + 18)), "white")
    d = ImageDraw.Draw(im)
    d.text((4, 4), titulo, fill="black")
    lista = []
    for i, m in enumerate(itens):
        nome = f"{letra}{i + 1}"
        p = Image.open(m["caminho"]).convert("RGB")
        dp = ImageDraw.Draw(p)
        a, b, corte = m["desenho"]
        dp.rectangle([a, 0, b - 1, 63], outline=(0, 90, 255))
        dp.line([(a, corte), (b - 1, corte)], fill=(230, 0, 0))
        x, y = (i % 2) * (256 * E + 8), 20 + (i // 2) * (64 * E + 18)
        # a fonte padrao nao tem c cedilha nem til: rotulo sem acento
        d.text((x, y), f"{nome}  {e1.sem_acento(m['palavra'])}  letra {m['posicao'] + 1}  {m['medida']:.3f}",
               fill="black")
        im.paste(p.resize((256 * E, 64 * E), Image.NEAREST), (x, y + 14))
        lista.append({"painel": nome, "palavra": m["palavra"], "letra": m["posicao"] + 1,
                      "medida": m["medida"], "imagem": os.path.relpath(m["caminho"], MODULO)})
    im.save(caminho)
    return lista


def sortear(itens, n, semente=0):
    """n itens sorteados, em ordem crescente de medida."""
    if len(itens) > n:
        escolha = np.random.default_rng(semente).choice(len(itens), n, replace=False)
        itens = [itens[k] for k in escolha]
    return sorted(itens, key=lambda m: m["medida"])


def folhas(no_modelo, limiar, pasta):
    """Folhas do modelo de 40 epocas: acertos, perdidas, falsos alarmes
    (todos), c sem cedilha abaixo do limiar e, em E, todas as cedilhas pedidas
    entre o limiar e LIMIAR_FOLHA_E (fora da medida tinta)."""
    grupos = [
        ("A", "acertos: cedilha pedida, medida acima do limiar (16 sorteadas)",
         sortear([m for m in no_modelo if m["pedida"] and m["medida"] > limiar], 16)),
        ("B", "perdidas: cedilha pedida, medida no limiar ou abaixo (16 sorteadas)",
         sortear([m for m in no_modelo if m["pedida"] and m["medida"] <= limiar], 16)),
        ("C", "falsos alarmes: c sem cedilha pedida, medida acima do limiar (todos)",
         sortear([m for m in no_modelo if not m["pedida"] and m["medida"] > limiar], 10 ** 6)),
        ("D", "c sem cedilha pedida, medida no limiar ou abaixo (16 sorteados)",
         sortear([m for m in no_modelo if not m["pedida"] and m["medida"] <= limiar], 16)),
        ("E", f"cedilha pedida, medida entre o limiar e {LIMIAR_FOLHA_E} (todas)",
         sortear([m for m in no_modelo if m["pedida"] and limiar < m["medida"] <= LIMIAR_FOLHA_E], 10 ** 6)),
    ]
    if MEDIDA == "tinta":
        grupos = grupos[:4]
    paineis = {}
    for letra, titulo, itens in grupos:
        caminho = os.path.join(pasta, f"cedilha_abaixo{SUFIXO}_folha_{letra}.png")
        paineis[letra] = folha(itens, letra, f"{letra}  {titulo}  limiar {limiar:.4f}", caminho)
    return paineis


def main():
    global ALINHADOR, MEDIDA, SUFIXO
    ap = argparse.ArgumentParser()
    ap.add_argument("--medida", choices=["tinta", "altura", "base_local"], default="tinta")
    MEDIDA = ap.parse_args().medida
    SUFIXO = "" if MEDIDA == "tinta" else "_" + MEDIDA
    ALINHADOR = e1_alinhador.carregar_alinhador()
    no_modelo = medir_no_modelo()
    negativos = [m for m in no_modelo if not m["pedida"]]
    pedidas = np.array([m["medida"] for m in no_modelo if m["pedida"]])
    limiar = float(np.percentile([m["medida"] for m in negativos], 95))

    base = medir_na_base("acentuada")
    par = np.array([m for m in medir_na_base("par") if m is not None])
    alinhadas = np.array([m for m in base if m is not None])
    # quanto o fundo do c sem cedilha desce abaixo da base usada, no c que abre
    # a palavra e no c do meio
    fundo_do_c = {}
    for nome, primeira in (("primeira_letra", True), ("meio", False)):
        v = [m["fundo"] for m in negativos if (m["posicao"] == 0) == primeira]
        fundo_do_c[nome] = {"n": len(v), "mediana_px": float(np.median(v)),
                            "p90_px": float(np.percentile(v, 90))}
    resultado = {
        "medida": MEDIDA,
        "fundo_do_c_sem_cedilha_abaixo_da_base": fundo_do_c,
        "tol_alturas_x": TOL,
        "negativo": {"c_sem_cedilha": len(negativos),
                     "mediana": float(np.median([m["medida"] for m in negativos])), "p95": limiar},
        "base_iam_pt_alinhado": {
            "cedilhas": len(base), "alinhadas": len(alinhadas),
            "mediana": float(np.median(alinhadas)),
            "acima_do_limiar": float((alinhadas > limiar).sum() / len(base)),
            "mesmo_c_no_par_sem_cedilha": {"n": len(par), "acima_do_limiar": float(np.mean(par > limiar))}},
        "modelo_40ep_cedilhas_pedidas": {
            "n": len(pedidas), "mediana": float(np.median(pedidas)),
            "acima_do_limiar": float(np.mean(pedidas > limiar))},
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=1))
    pasta = os.path.join(MODULO, "resultados")
    resultado["folhas"] = folhas(no_modelo, limiar, pasta)

    # varredura a mao; os c sem cedilha que passam sao nomeados pelo painel da
    # folha C, que tem todos os que passam do p95
    medida_c = [(q["painel"], q["medida"]) for q in resultado["folhas"]["C"]]
    resultado["limiares_a_mao"] = [
        {"limiar": t,
         "base_cedilhas": float((alinhadas > t).sum() / len(base)),
         "base_mesmo_c_no_par": float(np.mean(par > t)),
         "modelo_cedilhas_pedidas": float(np.mean(pedidas > t)),
         "modelo_c_sem_cedilha": [nome for nome, m in medida_c if m > t]}
        for t in LIMIARES_A_MAO[MEDIDA]]
    for linha in resultado["limiares_a_mao"]:
        print(linha)
    with open(os.path.join(pasta, f"cedilha_abaixo{SUFIXO}_40ep.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
