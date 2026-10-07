"""O gradiente da GPU erra so na ESCALA ou tambem na DIRECAO?

Distincao decisiva: o treino usa clip_grad_norm_, que NORMALIZA a magnitude.
Se o erro fosse so de escala, o clip corrigiria e treinar seria viavel.

Mede o cosseno entre o gradiente da CPU e o da GPU, com as mesmas entradas e
os mesmos pesos.

    CKPT=... python diagnostico/teste_direcao_gradiente.py

LEITURA
    cosseno ~1,0  -> so a escala esta errada; o clip corrigiria
    cosseno baixo -> a direcao esta errada; cada passo anda para o lado errado

Resultado nesta maquina, lote 8: cosseno 0,357, com 93,6% da norma do
gradiente da GPU ortogonal ao gradiente correto.
"""

import os, sys, argparse, torch, torch.nn as nn
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "DiffusionPen"))
os.chdir(RAIZ)
from unet import UNetModel

CKPT = os.environ.get("CKPT", "./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt")
B = int(os.environ.get("LOTE", "8"))
class Passa(nn.Module):
    def __init__(s,m): super().__init__(); s.module=m
    def forward(s,*a,**k): return s.module(*a,**k)
def monta(dev):
    a=argparse.Namespace(device=dev,img_size=(64,256),channels=4,emb_dim=320,num_heads=4,
        num_res_blocks=1,latent=True,img_feat=True,interpolation=False,mix_rate=None,
        model_name="diffusionpen",color=True,level="word",unet="unet_latent",batch_size=B,save_path="")
    emb=(lambda m: DataParallel(m,device_ids=[0])) if dev!="cpu" else Passa
    te=emb(CanineModel.from_pretrained("google/canine-c")).to(dev); te.requires_grad_(False); te.eval()
    u=UNetModel(image_size=a.img_size,in_channels=4,model_channels=320,out_channels=4,
        num_res_blocks=1,attention_resolutions=(1,1),channel_mult=(1,1),num_heads=4,
        num_classes=339,context_dim=320,vocab_size=79,text_encoder=te,args=a)
    u=emb(u).to(dev); u.load_state_dict(torch.load(CKPT,map_location=dev,weights_only=True)); u.eval(); return u
tok=CanineTokenizer.from_pretrained("google/canine-c")
torch.manual_seed(0)
X=torch.randn(B,4,8,32); A=torch.randn(B,4,8,32); T=torch.randint(0,1000,(B,)).long()
SF=torch.randn(B*5,1280); Y=torch.zeros(B).long()
txt=tok(["nao"]*B,padding="max_length",truncation=True,return_tensors="pt",max_length=40)
mse=nn.MSELoss(); g={}
for dev in ("cpu","cuda:0"):
    u=monta(dev); t={k:v.to(dev) for k,v in txt.items()}
    for _ in range(2):
        u.zero_grad(set_to_none=True)
        mse(A.to(dev),u(X.to(dev),timesteps=T.to(dev),context=t,y=Y.to(dev),style_extractor=SF.to(dev))).backward()
    g[dev]={n:p.grad.detach().float().cpu().clone() for n,p in u.named_parameters() if p.grad is not None}
    u.zero_grad(set_to_none=True); del u
    if dev!="cpu": torch.cuda.empty_cache()
c=torch.cat([v.flatten() for v in g["cpu"].values()])
d=torch.cat([g["cuda:0"][k].flatten() for k in g["cpu"]])
cos=float(torch.dot(c,d)/(c.norm()*d.norm()))
print(f"\nlote {B} | cosseno entre gradiente CPU e GPU: {cos:.6f}")
print(f"  norma CPU {float(c.norm()):.3f} | norma GPU {float(d.norm()):.3f} | razao {float(d.norm()/c.norm()):.2f}x")
print(f"\n  cosseno ~1,0  -> so a ESCALA esta errada; clip_grad_norm corrigiria")
print(f"  cosseno baixo -> a DIRECAO esta errada; o treino anda para o lado errado")
# quanto do gradiente da GPU e "ruido" ortogonal ao correto
proj = float(torch.dot(c,d)/c.norm()**2)
resid = float((d - proj*c).norm()/d.norm())
print(f"\n  fracao da norma da GPU que e ortogonal ao gradiente correto: {resid*100:.1f}%")
