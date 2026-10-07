"""Nitidez do traco nos paineis gerados pelo avaliar_pt.py.

Mede, por painel (semente 42, metade das palavras):
  tinta cinza -- entre os pixels de tinta (mais escuros que o papel - 40), a
                 fracao que NAO e tinta firme (papel - 120): traco borrado
                 tem mais cinza intermediario;
  gradiente   -- modulo medio do gradiente (Sobel) na borda do traco: borda
                 nitida tem gradiente alto.
Serviu para mostrar que as bases GERADAS pelo DiffusionPen borram o traco
(0,68 de tinta cinza contra 0,59 do IAM original) e que voltar ao IAM real
devolve a nitidez (docs/07_experimentos_com_acentos.md, fases 7 e 8).

    python diagnostico/nitidez.py --modelo avaliacao_pt_val:iam --modelo avaliacao_peso_val:peso5_16ep
"""

import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="pasta_da_avaliacao:rotulo")
    a = ap.parse_args()
    for m in a.modelo:
        pasta, rot = m.rsplit(":", 1)
        cinza, grad = [], []
        for arq in sorted(glob.glob(os.path.join(pasta, "paineis", rot, "*_42.png")))[::2]:
            g = cv2.imread(arq, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            papel = np.median(g)
            tinta = g < papel - 40
            if tinta.sum() < 50:
                continue
            cinza.append(1 - (g < papel - 120).sum() / tinta.sum())
            gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
            t8 = tinta.astype(np.uint8)
            borda = cv2.dilate(t8, np.ones((3, 3))).astype(bool) & ~cv2.erode(t8, np.ones((3, 3))).astype(bool)
            grad.append(np.hypot(gx, gy)[borda].mean())
        print(f"{rot:20s} n={len(cinza):4d}  tinta cinza: {np.mean(cinza):.3f}  gradiente na borda: {np.mean(grad):.1f}")


if __name__ == "__main__":
    main()
