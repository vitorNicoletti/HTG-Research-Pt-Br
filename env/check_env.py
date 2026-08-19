"""Fase 0 - verificacao de ambiente GPU (ROCm).

Sai com codigo != 0 se a GPU nao estiver realmente sendo usada.
Nunca cai para CPU silenciosamente.

Uso:
    python env/check_env.py
"""

import sys

import torch

print("torch:", torch.__version__)
print("hip/rocm:", getattr(torch.version, "hip", None))
print("cuda (build):", torch.version.cuda)
print("cuda disponivel (API ROCm usa o mesmo nome):", torch.cuda.is_available())

if not torch.cuda.is_available():
    sys.exit("FALHA: GPU nao visivel ao PyTorch")

if getattr(torch.version, "hip", None) is None:
    sys.exit("FALHA: torch nao foi compilado com ROCm/HIP (wheel errado instalado)")

print("device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
# gcnArchName e o campo que expoe 'gfx1200'; get_device_name da o nome comercial.
print("arquitetura:", getattr(props, "gcnArchName", "desconhecida"))
print("VRAM total (GB):", round(props.total_memory / 1e9, 2))

# teste real de computacao na GPU
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = a @ a
torch.cuda.synchronize()
if not torch.isfinite(b).all():
    sys.exit("FALHA: matmul fp16 produziu NaN/Inf - kernel quebrado para esta arch")
print("matmul fp16 OK, pico VRAM (GB):",
      round(torch.cuda.max_memory_allocated() / 1e9, 2))

# bf16, usado no treino
try:
    c = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    _ = c @ c
    torch.cuda.synchronize()
    print("bf16 OK")
except Exception as e:
    print("AVISO: bf16 falhou:", e)

# SDPA - substitui o xformers no ROCm (ver patch da Fase 1)
try:
    q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    print("scaled_dot_product_attention OK")
except Exception as e:
    print("AVISO: SDPA falhou:", e)

print("\nAMBIENTE OK")
