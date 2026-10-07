"""O acento caiu na letra CERTA? Mede a posicao das marcas nos paineis do avaliar_pt.py.

A medida de marca do avaliar_pt.py (scripts/medir_marcas.py) so diz se ha uma
marca solta acima/abaixo do corpo: "hávera" conta como acerto para "haverá".
Aqui cada marca e atribuida a uma letra:
  1. o alinhador CTC (modelos/alinhador_iam.pt) alinha o ESQUELETO ("havera")
     no recorte da palavra gerada; fronteiras entre letras no meio dos disparos,
     ajustadas aos vales de tinta -- o mesmo localizador que desenhou os
     acentos da base (acentos_sinteticos.gerador);
  2. as marcas sao as do medir_marcas (acima ou abaixo do corpo), e cada uma
     vai para a letra cuja fatia contem o centro da sua caixa.
Por painel acentuado:
  certa    -- toda letra acentuada tem marca do lado certo (acima; abaixo na cedilha)
  parcial  -- so parte delas (palavras com mais de um sinal)
  errada   -- ha marca, mas em nenhuma letra acentuada (ou do lado errado)
  nenhuma  -- nenhuma marca
  extra    -- (a parte) ha marca em alguma letra que nao devia ter
Paineis sem acento (esqueleto, sem_acento): taxa de marca (falso positivo).
Por sinal: fracao das letras com o sinal pedido que receberam marca do lado certo.

O alinhador so localiza letras; quem decide se ha acento e o detector de marcas.
--conferir_base mede o metodo em imagens da base, onde a posicao e conhecida.

    python scripts/medir_posicao_acento.py --avaliacao avaliacao_peso_val:peso5_16ep \\
        --avaliacao avaliacao_pt_val:iam --saida saidas/posicao_acento
    python scripts/medir_posicao_acento.py --conferir_base iam_pt_alinhado --saida ...
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
from acentos_sinteticos import alinhamento, geometria  # noqa: E402
from acentos_sinteticos.vocabulario import esqueleto  # noqa: E402
from medir_marcas import marcas  # noqa: E402

TIPO = {"ã": "til", "õ": "til", "ç": "cedilha", "á": "agudo", "é": "agudo", "í": "agudo",
        "ó": "agudo", "ú": "agudo", "à": "grave", "â": "circunflexo", "ê": "circunflexo",
        "ô": "circunflexo"}
MARGEM = 3


def recorte(g):
    """Painel -> (recorte justo na tinta, deslocamento x); None sem tinta (= avaliar_pt.recorte_tinta)."""
    m = geometria.mascara_tinta(g)
    if m.sum() < 20:
        return None, 0
    ys, xs = np.where(m)
    x0 = max(0, xs.min() - MARGEM)
    return g[max(0, ys.min() - MARGEM):ys.max() + 1 + MARGEM, x0:xs.max() + 1 + MARGEM], x0


def letras_das_marcas(g, texto, alinhador):
    """-> ([(indice_da_letra, 'acima'|'abaixo')], motivo). Fatias no recorte, marcas no painel."""
    r, x0 = recorte(g)
    if r is None:
        return [], "sem_tinta"
    geo = geometria.analisar(r)
    al = alinhador.alinhar(r, texto) if geo is not None else None
    if al is None:
        return [], "sem_alinhamento"
    fats = geometria.ajustar_aos_vales(geo, alinhamento.fatias_do_alinhamento(al, geo))
    _, _, caixas, _ = marcas(g)
    saida = []
    for x, y, w, h, lado in caixas:
        cx = x + w / 2 - x0
        k = next((i for i, (a, b) in enumerate(fats) if a <= cx < b),
                 0 if cx < fats[0][0] else len(fats) - 1)
        saida.append((k, lado))
    return saida, "ok"


def classificar(palavra, marcas_letra):
    esq = esqueleto(palavra)
    pedidas = {i: ("abaixo" if TIPO[c] == "cedilha" else "acima")
               for i, (c, b) in enumerate(zip(palavra, esq)) if c != b}
    acertos = {i for i, lado in pedidas.items() if (i, lado) in marcas_letra}
    extra = any(k not in pedidas for k, _ in marcas_letra)
    if not marcas_letra:
        cls = "nenhuma"
    elif len(acertos) == len(pedidas):
        cls = "certa"
    elif acertos:
        cls = "parcial"
    else:
        cls = "errada"
    sinais = [(TIPO[palavra[i]], i in acertos) for i in pedidas]
    return cls, extra, sinais


def resumir(registros):
    """registros: [(tipo_painel, cls, extra, sinais, motivo)] -> dict."""
    ac = [r for r in registros if r[0] == "acentuada" and r[4] == "ok"]
    out = {"paineis_acentuados": len(ac),
           "sem_alinhamento": sum(1 for r in registros if r[0] == "acentuada" and r[4] != "ok")}
    c = Counter(r[1] for r in ac)
    for k in ("certa", "parcial", "errada", "nenhuma"):
        out[k] = round(c[k] / max(1, len(ac)), 4)
    out["extra_na_acentuada"] = round(sum(r[2] for r in ac) / max(1, len(ac)), 4)
    por_sinal = defaultdict(list)
    for r in ac:
        for t, ok in r[3]:
            por_sinal[t].append(ok)
    out["acerto_por_sinal"] = {t: {"n": len(v), "acerto": round(float(np.mean(v)), 4)}
                               for t, v in sorted(por_sinal.items())}
    for t in ("esqueleto", "sem_acento"):
        v = [r for r in registros if r[0] == t and r[4] == "ok"]
        if v:
            out[f"marca_em_{t}"] = round(sum(r[1] != "nenhuma" for r in v) / len(v), 4)
    return out


def medir_avaliacao(pasta, rotulo, alinhador):
    resumo = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
    k_de = {p: k for k, p in enumerate(resumo["palavras"])}
    regs = []
    with open(os.path.join(pasta, "paineis.tsv"), encoding="utf-8") as f:
        next(f)
        for l in f:
            m, p, tipo, e, s, *_ = l.rstrip("\n").split("\t")
            if m != rotulo:
                continue
            g = np.asarray(Image.open(os.path.join(pasta, "paineis", rotulo, f"{k_de[p]:03d}_{e}_{s}.png"))
                           .convert("L"), dtype=np.float32)
            ml, motivo = letras_das_marcas(g, esqueleto(p), alinhador)
            if tipo == "acentuada":
                cls, extra, sinais = classificar(p, ml)
            else:
                cls, extra, sinais = ("nenhuma" if not ml else "marca"), False, []
            regs.append((tipo, cls, extra, sinais, motivo))
    return resumir(regs)


def conferir_base(base, n, alinhador, seed=0):
    """Imagens acentuadas da base, pre-processadas como no treino (64x256): posicao conhecida."""
    sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
    from utils.iam_acentuado_dataset import preprocessar_iam
    man = [json.loads(l) for l in open(os.path.join(base, "manifesto.jsonl"), encoding="utf-8")]
    ac = [x for x in man if x["tipo"] == "acentuada" and not set("ij") & set(esqueleto(x["rotulo"]))]
    regs = []
    for x in random.Random(seed).sample(ac, min(n, len(ac))):
        img = preprocessar_iam(Image.open(os.path.join(base, x["arquivo"])).convert("RGB"), x["rotulo"])
        g = np.asarray(img.convert("L"), dtype=np.float32)
        ml, motivo = letras_das_marcas(g, esqueleto(x["rotulo"]), alinhador)
        cls, extra, sinais = classificar(x["rotulo"], ml)
        regs.append(("acentuada", cls, extra, sinais, motivo))
    return resumir(regs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avaliacao", action="append", default=[], help="pasta:rotulo de um avaliar_pt.py")
    ap.add_argument("--conferir_base", action="append", default=[], help="base com manifesto acentuada/par")
    ap.add_argument("--n_base", type=int, default=800)
    ap.add_argument("--alinhador", default=os.path.join(RAIZ, "modelos/alinhador_iam.pt"))
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    alinhador = alinhamento.Alinhador(a.alinhador, "cpu")

    rel = {}
    for b in a.conferir_base:
        rel[f"base:{b}"] = conferir_base(b, a.n_base, alinhador)
        print(f"base:{b}", json.dumps(rel[f"base:{b}"], ensure_ascii=False), flush=True)
    for av in a.avaliacao:
        pasta, rot = av.rsplit(":", 1)
        rel[rot] = medir_avaliacao(pasta, rot, alinhador)
        print(rot, json.dumps(rel[rot], ensure_ascii=False), flush=True)
    with open(os.path.join(a.saida, "posicao.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)

    tab = ["| modelo | certa | parcial | errada | nenhuma | marca em outra letra | agudo | til | cedilha | circunflexo | marca no esqueleto | marca sem acento |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pc = lambda v: "—" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    for k, r in rel.items():
        s = r["acerto_por_sinal"]
        tab.append(f"| {k} | {pc(r['certa'])} | {pc(r['parcial'])} | {pc(r['errada'])} | {pc(r['nenhuma'])} | "
                   f"{pc(r['extra_na_acentuada'])} | "
                   + " | ".join(pc(s.get(t, {}).get("acerto")) for t in ("agudo", "til", "cedilha", "circunflexo"))
                   + f" | {pc(r.get('marca_em_esqueleto'))} | {pc(r.get('marca_em_sem_acento'))} |")
    texto = "\n".join(tab)
    open(os.path.join(a.saida, "posicao.md"), "w", encoding="utf-8").write(texto + "\n")
    print(texto)


if __name__ == "__main__":
    main()
