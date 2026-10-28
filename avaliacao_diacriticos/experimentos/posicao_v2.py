"""
Etapa 2 da metrica final, correcao: variantes da atribuicao de marca a letra,
escolhidas com base no diagnostico (diagnosticar_posicao.py) e validadas sem
reusar as imagens do ajuste.

As palavras da anotacao sao divididas em duas metades fixas (palavras em
ordem alfabetica, alternadas). Tudo que tem parametro e escolhido na metade
A; o numero que vale e o da metade B.

Variantes, cada uma somada a anterior:
  base     scripts/medir_posicao_acento.py: marca solta acima do corpo
           (scripts/medir_marcas.py), letra cuja coluna contem o centro
  vogal    a marca vai para a vogal cuja coluna mais se sobrepoe a ela; acento
           em portugues so cai em vogal. Sem vogal sobreposta, fica a letra
           da base
  alta     aceita marca acima do corpo mais alta que a altura-x, que o
           detector descarta como haste. A marca precisa estar inteira acima
           do corpo, entao nao e haste
  junta    componentes acima do corpo separados por poucos pixels viram uma
           marca so (til ou circunflexo partido)
  area     area minima da marca, escolhida na metade A
  tinta    alem das manchas soltas, uma vogal conta como marcada quando ha
           tinta acima do corpo na coluna dela, dividida pela tinta do corpo
           da vogal, acima de um limiar escolhido na metade A. E o mesmo
           desenho da cedilha (tinta abaixo do c), e pega o acento que encosta
           na letra e por isso nao e mancha solta

Medidas, contra o gabarito letra por letra (gabarito_por_letra):
  por letra   em cada vogal, a pessoa ve marca e o detector acha marca
  por imagem  as quatro classes (so o certo, certo e a mais, so a mais, nenhum)

Grava resultados/posicao_v2_40ep.txt.

    python avaliacao_diacriticos/experimentos/posicao_v2.py
"""
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
MODULO = os.path.dirname(AQUI)
sys.path.insert(0, MODULO)
sys.path.insert(0, AQUI)

from diagnosticar_posicao import gabarito_por_letra  # noqa: E402
from sinais import Detector, sem_acento, sinais_pedidos, tem_pingo  # noqa: E402
from medir_marcas import ALTURA_MAX, AREA_MIN, TOL  # noqa: E402
from acentos_sinteticos import geometria  # noqa: E402

RES = os.path.join(MODULO, "resultados")
VOGAIS = set("aeiou")
GAP_JUNTA = 3          # px entre caixas para juntar componentes
AREAS = [12, 20, 30, 40, 60]
LIMIARES_TINTA = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40]


def tinta_acima(g, fatias, x0, palavra):
    """{letra 1..n: tinta acima do corte / tinta do corpo} nas vogais."""
    geo = geometria.analisar(g)
    if geo is None:
        return {}
    corte = int(round(geo.topo_x - TOL * geo.altura_x))
    saida = {}
    for k, (fa, fb) in enumerate(fatias):
        if palavra[k] not in VOGAIS:
            continue
        a, b = int(round(fa + x0)), max(int(round(fb + x0)), int(round(fa + x0)) + 1)
        acima = geo.mask[:max(0, corte), a:b].sum()
        corpo = geo.mask[geo.topo_x:geo.base + 1, a:b].sum()
        saida[k + 1] = float(acima) / corpo if corpo else 0.0
    return saida


def componentes_acima(g, alta):
    """[(x, y, w, h, area)] das manchas soltas acima do corpo, no painel."""
    geo = geometria.analisar(g)
    if geo is None:
        return []
    hx = geo.altura_x
    n, _, st, _ = cv2.connectedComponentsWithStats(geo.mask.astype(np.uint8), 8)
    saida = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if area < AREA_MIN:
            continue
        if not alta and h >= ALTURA_MAX * hx:
            continue
        if y + h <= geo.topo_x + TOL * hx:
            saida.append([int(x), int(y), int(w), int(h), int(area)])
    return saida


def juntar(caixas):
    caixas = sorted(caixas)
    saida = []
    for c in caixas:
        if saida and c[0] <= saida[-1][0] + saida[-1][2] + GAP_JUNTA:
            a = saida[-1]
            x0, y0 = min(a[0], c[0]), min(a[1], c[1])
            x1, y1 = max(a[0] + a[2], c[0] + c[2]), max(a[1] + a[3], c[1] + c[3])
            saida[-1] = [x0, y0, x1 - x0, y1 - y0, a[4] + c[4]]
        else:
            saida.append(list(c))
    return saida


def letras(caixas, fatias, x0, palavra, vogal):
    """Conjunto das letras (1..n) com marca."""
    saida = set()
    for x, y, w, h, _ in caixas:
        a, b = x - x0, x + w - x0
        cx = (a + b) / 2
        k = next((i for i, (fa, fb) in enumerate(fatias) if fa <= cx < fb),
                 0 if cx < fatias[0][0] else len(fatias) - 1)
        if vogal and palavra[k] not in VOGAIS:
            sobre = [(min(b, fb) - max(a, fa), i) for i, (fa, fb) in enumerate(fatias) if palavra[i] in VOGAIS]
            sobre = [s for s in sobre if s[0] > 0]
            if sobre:
                k = max(sobre)[1]
        saida.add(k + 1)
    return saida


def classe(palavra, marcadas):
    pedidas = {k + 1 for k, s in sinais_pedidos(palavra) if s != "cedilha"}
    certas = bool(pedidas) and pedidas <= marcadas
    a_mais = bool(marcadas - pedidas)
    return "so o certo" if certas and not a_mais else "certo e a mais" if certas else "so a mais" if a_mais else "nenhum"


def cohen(a, b):
    n = len(a)
    obs = sum(x == y for x, y in zip(a, b)) / n
    cl = set(a) | set(b)
    acaso = sum((a.count(c) / n) * (b.count(c) / n) for c in cl)
    return float("nan") if acaso == 1 else (obs - acaso) / (1 - acaso)


def avaliar(dados, cfg, imagens):
    """kappa por letra (vogais), kappa da classe e imagens com letras iguais."""
    ph, pd, ch, cd, iguais = [], [], [], [], 0
    for img in imagens:
        palavra, gab, g, analise, pedida = dados[img]
        geo, fatias, x0 = analise
        caixas = componentes_acima(g, cfg["alta"])
        caixas = [c for c in caixas if c[4] >= cfg["area"]]
        if cfg["junta"]:
            caixas = juntar(caixas)
        det = letras(caixas, fatias, x0, palavra, cfg["vogal"])
        if cfg.get("tinta") is not None:
            det |= {k for k, m in tinta_acima(g, fatias, x0, palavra).items() if m > cfg["tinta"]}
        for k, l in enumerate(palavra):
            if l in VOGAIS:
                ph.append(k + 1 in gab); pd.append(k + 1 in det)
        ch.append(classe(pedida, gab)); cd.append(classe(pedida, det))
        iguais += gab == det
    vp = sum(h and d for h, d in zip(ph, pd)); fp = sum(d and not h for h, d in zip(ph, pd))
    fn = sum(h and not d for h, d in zip(ph, pd))
    return {"letra_kappa": cohen(ph, pd), "letra_prec": vp / max(1, vp + fp), "letra_rec": vp / max(1, vp + fn),
            "classe_kappa": cohen(ch, cd), "iguais": iguais, "n": len(imagens)}


def main():
    chave = {r["imagem"]: r for r in csv.DictReader(open(os.path.join(RES, "anotacao_40ep_chave.csv"), encoding="utf-8"))}
    gab = gabarito_por_letra()
    d = Detector()
    dados = {}
    for img, item in chave.items():
        palavra = sem_acento(item["palavra"])
        if tem_pingo(item["palavra"]):
            continue
        g = np.asarray(Image.open(os.path.join(MODULO, item["painel"])).convert("L"), dtype=np.float32)
        analise = d.fatias(g, palavra)
        if analise is None:
            continue
        dados[img] = (palavra, gab[img], g, analise, item["palavra"])
    palavras = sorted({sem_acento(chave[i]["palavra"]) for i in dados})
    metade_a = set(palavras[0::2])
    A = [i for i in dados if sem_acento(chave[i]["palavra"]) in metade_a]
    B = [i for i in dados if i not in A]
    so_acento = lambda imgs: [i for i in imgs if any(s != "cedilha" for _, s in sinais_pedidos(chave[i]["palavra"]))]

    variantes = [("base", dict(vogal=False, alta=False, junta=False, area=AREA_MIN)),
                 ("+ vogal", dict(vogal=True, alta=False, junta=False, area=AREA_MIN)),
                 ("+ alta", dict(vogal=True, alta=True, junta=False, area=AREA_MIN)),
                 ("+ junta", dict(vogal=True, alta=True, junta=True, area=AREA_MIN))]
    # area minima: escolhida na metade A, pelo kappa da classe nas palavras com acento em cima
    base_cfg = dict(variantes[-1][1])
    escolha = max(AREAS, key=lambda a: avaliar(dados, dict(base_cfg, area=a), so_acento(A))["classe_kappa"])
    variantes.append((f"+ area >= {escolha}", dict(base_cfg, area=escolha)))

    cfg_t = dict(variantes[3][1])
    lim = max(LIMIARES_TINTA, key=lambda v: avaliar(dados, dict(cfg_t, tinta=v), A)["letra_kappa"])
    variantes.append((f"+ tinta > {lim}", dict(cfg_t, tinta=lim)))
    out = [f"Atribuicao de marca a letra, contra o gabarito humano letra por letra (modelo de 40 epocas).",
           f"Metade A: {len(A)} imagens ({len(metade_a)} palavras), usada para escolher a area minima.",
           f"Metade B: {len(B)} imagens, nao usada em nenhuma escolha.", ""]
    for nome_grupo, filtro in (("palavras com acento em cima", so_acento), ("todas as imagens sem i e j", lambda x: x)):
        out.append(nome_grupo.upper())
        out.append(f"  {'variante':16s} | {'metade':6s} | kappa por letra | precisao | recall | kappa da classe | letras iguais")
        for nome, cfg in variantes:
            for rotulo, imgs in (("A", A), ("B", B)):
                r = avaliar(dados, cfg, filtro(imgs))
                out.append(f"  {nome:16s} | {rotulo:6s} | {r['letra_kappa']:+.2f}           | {r['letra_prec']:.2f}     | "
                           f"{r['letra_rec']:.2f}   | {r['classe_kappa']:+.2f}           | {r['iguais']} de {r['n']}")
        out.append("")
    texto = "\n".join(out)
    open(os.path.join(RES, "posicao_v2_40ep.txt"), "w", encoding="utf-8").write(texto + "\n")
    print(texto)


if __name__ == "__main__":
    main()
