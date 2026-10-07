"""
checar_alinhamento.py -- PORTAO (Passo 1).

Mede se os gemeos de um par minimo saem espacialmente alinhados. Se saem,
o eixo E1 pode ser feito por diferenca de imagens. Se nao saem, e preciso
cair na deteccao por faixa/coluna.

Criterios reportados por par (imagens ja em tinta=1, fundo=0):
  * dx*, dy*  : deslocamento que maximiza a correlacao cruzada 2D
  * IoU@0     : intersecao/uniao das mascaras de tinta SEM deslocar
  * IoU@best  : o mesmo apos deslocar pelo (dx*, dy*)
  * corr_col@0: correlacao de Pearson dos perfis de tinta por coluna
  * res@0     : fracao de tinta do gemeo ASCII que NAO e explicada pela
                acentuada (mede quanta massa "sobra" fora do diacritico)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image


def carregar_tinta(caminho):
    """PNG -> array float 0..1 onde 1 = tinta (escuro)."""
    g = np.asarray(Image.open(caminho).convert("L"), dtype=np.float32) / 255.0
    return 1.0 - g


def binariza(t):
    """Mascara de tinta por Otsu.

    Limiar relativo (min/max) nao serve: numa imagem quase vazia ele promove
    o ruido do fundo a "tinta" e a metrica passa a contar acento onde nao ha
    nada. Otsu com uma guarda de contraste minimo evita isso -- se a imagem
    nao tem dois modos, ela e declarada sem tinta.
    """
    import cv2
    if float(t.max()) - float(t.min()) < 0.10:
        return np.zeros_like(t, dtype=bool)
    u8 = (np.clip(t, 0, 1) * 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m.astype(bool)


def melhor_shift(a, b, max_dx=24, max_dy=8):
    """(dx, dy) que maximiza sum(a * shift(b)). dx>0 = b deslocado p/ direita."""
    melhor = (0, 0, -1.0)
    H, W = a.shape
    for dy in range(-max_dy, max_dy + 1):
        for dx in range(-max_dx, max_dx + 1):
            bs = np.zeros_like(b)
            ys0, ys1 = max(0, dy), min(H, H + dy)
            xs0, xs1 = max(0, dx), min(W, W + dx)
            bs[ys0:ys1, xs0:xs1] = b[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
            s = float((a * bs).sum())
            if s > melhor[2]:
                melhor = (dx, dy, s)
    return melhor[0], melhor[1]


def desloca(b, dx, dy):
    H, W = b.shape
    out = np.zeros_like(b)
    ys0, ys1 = max(0, dy), min(H, H + dy)
    xs0, xs1 = max(0, dx), min(W, W + dx)
    out[ys0:ys1, xs0:xs1] = b[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def iou(ma, mb):
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--contato", default=None, help="PNG de inspecao visual")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    reg = defaultdict(dict)
    meta = {}
    with open(os.path.join(a.dir, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chave = (d["par_id"], d["escritor"], d["semente"])
            reg[chave]["acc" if d["acentuada"] else "asc"] = d["arquivo"]
            meta.setdefault(chave, {})["acc" if d["acentuada"] else "asc"] = d["palavra"]

    linhas, tiras = [], []
    for chave in sorted(reg):
        r = reg[chave]
        if "acc" not in r or "asc" not in r:
            continue
        A = carregar_tinta(os.path.join(a.dir, r["acc"]))
        B = carregar_tinta(os.path.join(a.dir, r["asc"]))
        mA, mB = binariza(A), binariza(B)

        dx, dy = melhor_shift(A, B)
        Bs = desloca(B, dx, dy)
        mBs = binariza(Bs)

        pA, pB = A.sum(axis=0), B.sum(axis=0)
        if pA.std() > 1e-6 and pB.std() > 1e-6:
            corr = float(np.corrcoef(pA, pB)[0, 1])
        else:
            corr = float("nan")

        # tinta do ASCII nao explicada pela acentuada (dilatando a acentuada 1px)
        from scipy.ndimage import binary_dilation
        mAd = binary_dilation(mA, iterations=1)
        res = float((mB & ~mAd).sum() / max(1, mB.sum()))

        linhas.append({
            "par_id": chave[0], "escritor": chave[1], "semente": chave[2],
            "acc": meta[chave]["acc"], "asc": meta[chave]["asc"],
            "dx": dx, "dy": dy,
            "iou_0": round(iou(mA, mB), 4),
            "iou_best": round(iou(mA, mBs), 4),
            "corr_col_0": round(corr, 4),
            "res_asc_nao_explicado_0": round(res, 4),
            "tinta_acc": int(mA.sum()), "tinta_asc": int(mB.sum()),
        })

        if a.contato:
            sep = np.ones((3, A.shape[1])) * 0.25
            dif = np.abs(A - B)
            tiras += [1 - A, sep, 1 - B, sep, 1 - dif / max(1e-6, dif.max()),
                      np.ones((8, A.shape[1]))]

    hdr = (f"{'par':22s} {'dx':>4s} {'dy':>4s} {'IoU@0':>7s} {'IoU@best':>9s} "
           f"{'corr_col':>9s} {'res_asc':>8s} {'tintaA':>7s} {'tintaB':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for l in linhas:
        print(f"{l['acc'] + '/' + l['asc']:22s} {l['dx']:4d} {l['dy']:4d} "
              f"{l['iou_0']:7.3f} {l['iou_best']:9.3f} {l['corr_col_0']:9.3f} "
              f"{l['res_asc_nao_explicado_0']:8.3f} {l['tinta_acc']:7d} {l['tinta_asc']:7d}")

    if linhas:
        import statistics as st
        print("\nMEDIANAS: |dx|=%d  |dy|=%d  IoU@0=%.3f  IoU@best=%.3f  corr_col=%.3f" % (
            st.median([abs(l["dx"]) for l in linhas]),
            st.median([abs(l["dy"]) for l in linhas]),
            st.median([l["iou_0"] for l in linhas]),
            st.median([l["iou_best"] for l in linhas]),
            st.median([l["corr_col_0"] for l in linhas]),
        ))

    if a.contato and tiras:
        img = np.clip(np.concatenate(tiras, axis=0), 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(a.contato)
        print("contato:", a.contato, "(ordem por par: acentuada / ascii / |diff|)")
    if a.json_out:
        json.dump(linhas, open(a.json_out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
