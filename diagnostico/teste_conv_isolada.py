"""Teste mínimo: esta GPU calcula UMA convolução corretamente?

Sem modelo, sem dataset, sem nada. Só uma nn.Conv2d, entrada sintética,
CPU contra GPU. Roda em segundos.

    python diagnostico/teste_conv_isolada.py

Leitura: ~1e-6 é ruído normal de fp32; >= 1e-2 é erro de verdade.
Na RX 6600 XT este teste PASSA — o defeito só aparece no modelo completo
(veja teste_lote_unet.py).
"""
import copy

import torch
import torch.nn as nn

torch.manual_seed(0)
conv = nn.Conv2d(320, 320, 3, padding=1)
x = torch.randn(8, 320, 8, 32)


def passo(m, inp):
    m.zero_grad(set_to_none=True)
    y = m(inp)
    y.pow(2).sum().backward()
    return y.detach().cpu(), m.weight.grad.detach().cpu().clone()


def rel(a, b):
    return float((a - b).norm() / a.norm())


y_cpu, g_cpu = passo(conv, x)

conv_gpu = copy.deepcopy(conv).cuda()
x_gpu = x.cuda()
y1, g1 = passo(conv_gpu, x_gpu)   # 1a chamada
y2, g2 = passo(conv_gpu, x_gpu)   # 2a chamada, entrada identica
y3, g3 = passo(conv_gpu, x_gpu)   # 3a

print("UMA convolucao 320->320, 3x3, entrada 8x320x8x32\n")
print(f"forward   GPU 1a vs 2a : {rel(y1, y2):.2e}")
print(f"gradiente GPU 1a vs 2a : {rel(g1, g2):.2e}")
print(f"gradiente GPU 2a vs 3a : {rel(g2, g3):.2e}")
print()
print(f"forward   CPU vs GPU(2a): {rel(y_cpu, y2):.2e}")
print(f"gradiente CPU vs GPU(2a): {rel(g_cpu, g2):.2e}")
print()
print("referencia: ~1e-6 = ruido normal de fp32 | >=1e-2 = errado")
