"""O acento pesa pouco na perda do treino? O modelo usa o diacritico do texto?

Usa os PARES da base portuguesa (scripts/gerar_base_pt.py): a mesma imagem
com os sinais ("fogão") e sem ("fogao"). A diferenca entre as duas e
exatamente o acento, o que da a mascara de onde ele esta.

Reproduz a conta do treino (diffusionpen_mods/train.py, laco de treino):
  latente z = vae.encode(img).latent_dist * 0.18215 (aqui a MEDIA da
  distribuicao, nao uma amostra, para a medida nao ter ruido do VAE);
  x_t = DDIMScheduler(SD 1.5).add_noise(z, eps, t); perda = MSE(eps, eps_hat)
  com media em todas as posicoes; texto pelo tokenizador CANINE com
  max_length 40; estilo = 5 referencias do mesmo escritor.

Tres medidas:
  1. PESO DO ACENTO NA PERDA (so depende dos dados, nao do modelo):
     a) fracao da AREA do latente (8x32) ocupada pelo acento;
     b) se o modelo desenhasse a palavra SEM acento quando o certo era com,
        quanto a perda subiria. Na parametrizacao eps, prever z_par no lugar
        de z_acc acrescenta SNR(t) * mean((z_acc - z_par)^2) a perda, com
        SNR(t) = abar/(1-abar). Comparado com a perda tipica do modelo no
        mesmo t, isso diz se esse erro "aparece" no treino.
  2. O MODELO USA O DIACRITICO? Na imagem acentuada com ruido, perda com o
     texto certo ("fogão") e com o errado ("fogao"), no total e so na regiao
     do acento; e quanto eps_hat muda trocando o texto, dentro e fora da
     regiao do acento. Mesma coisa no par (texto certo "fogao", errado "fogão").
  3. O CANINE DISTINGUE? Similaridade das representacoes de texto de
     "fogão" x "fogao" contra "fogão" x outra palavra de mesmo tamanho, na
     saida do CANINE (congelado) e depois da text_lin do UNet (treinavel).

Os pares sao dados de TREINO de proposito: a pergunta e sobre o sinal que o
treino recebe, nao sobre generalizacao.

    python diagnostico/diag_peso_acento.py --modelo iam=<ema.pt> --modelo pt=<ema.pt> \\
        --base iam_pt --saida diagnostico/resultados/peso_acento
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from comum.diffusionpen import TRANSFORM, EscritoresIAM, GeradorDiffusionPen  # noqa: E402

LIMIAR_PIXEL = 40          # diferenca de cinza (0..255) que conta como acento
TS = (25, 100, 250, 500, 750, 950)
RUIDOS_POR_T = 2
LOTE = 16
TIPO_SINAL = {"ã": "til", "õ": "til", "ç": "cedilha", "á": "agudo", "é": "agudo", "í": "agudo",
              "ó": "agudo", "ú": "agudo", "à": "grave", "â": "circunflexo", "ê": "circunflexo",
              "ô": "circunflexo"}


def carregar_pares(base, n, seed):
    """[(img_acentuada, img_par, rotulo_acentuado, rotulo_par, escritor, tipos)]."""
    from utils.iam_acentuado_dataset import preprocessar_iam
    m = [json.loads(l) for l in open(os.path.join(base, "manifesto.jsonl"), encoding="utf-8")]
    par = {x["tarefa"]: x for x in m if x["tipo"] == "par"}
    ac = [x for x in m if x["tipo"] == "acentuada" and x["tarefa"] in par]
    saida, com_margem = [], 0
    for x in random.Random(seed).sample(ac, min(n, len(ac))):
        p = par[x["tarefa"]]
        a_img = Image.open(os.path.join(base, x["arquivo"])).convert("L")
        p_img = Image.open(os.path.join(base, p["arquivo"])).convert("L")
        if a_img.size != p_img.size:
            # o sinal passou da borda e a tela ganhou margem: poe o par na
            # mesma tela, na mesma posicao, com a cor do papel, para as duas
            # imagens ficarem alinhadas depois do pre-processamento
            com_margem += 1
            ox, oy = x["margens_esq_cima"]
            g = np.asarray(p_img, dtype=np.float32)
            papel = int(np.median(g[g >= np.percentile(g, 60)]))
            tela = Image.new("L", a_img.size, papel)
            tela.paste(p_img, (ox, oy))
            assert ox + p_img.width <= a_img.width and oy + p_img.height <= a_img.height, x["arquivo"]
            p_img = tela
        ia = preprocessar_iam(a_img.convert("RGB"), x["rotulo"])
        ip = preprocessar_iam(p_img.convert("RGB"), p["rotulo"])
        tipos = sorted({TIPO_SINAL[s["letra"]] for s in x.get("sinais", [])})
        saida.append((ia, ip, x["rotulo"], p["rotulo"], x["escritor"], tipos))
    return saida, com_margem


def mascara_latente(ia, ip, forma_lat):
    """Mascara do acento no latente: celula com algum pixel que mudou > LIMIAR_PIXEL."""
    a = np.asarray(ia.convert("L"), dtype=np.int16)
    b = np.asarray(ip.convert("L"), dtype=np.int16)
    px = np.abs(a - b) > LIMIAR_PIXEL                  # (64, 256)
    h, w = forma_lat
    fy, fx = px.shape[0] // h, px.shape[1] // w
    return px.reshape(h, fy, w, fx).any(axis=(1, 3)), px


def media_em(x, m):
    """Media de x (B,C,H,W) nas posicoes da mascara m (B,H,W); NaN se vazia."""
    mm = m[:, None].expand_as(x).float()
    s = mm.sum()
    return float((x * mm).sum() / s) if s > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, help="rotulo=ckpt do EMA")
    ap.add_argument("--base", required=True)
    ap.add_argument("--n_pares", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", default=os.path.join(RAIZ, "DiffusionPen/style_models/iam_style_diffusionpen.pth"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    pares, com_margem = carregar_pares(a.base, a.n_pares, a.seed)
    esc = EscritoresIAM("iam_train_val.txt")
    refs = [esc.referencias(e, random.Random(a.seed * 1000 + k)) for k, (_, _, _, _, e, _) in enumerate(pares)]
    print(f"{len(pares)} pares da base {a.base}; {com_margem} com margem extra na acentuada "
          f"(no treino esses pares entram em escalas diferentes)", flush=True)

    relatorio = {"pares": len(pares), "pares_com_margem": com_margem, "ts": list(TS), "ruidos_por_t": RUIDOS_POR_T,
                 "limiar_pixel": LIMIAR_PIXEL, "modelos": {}}
    for rm in a.modelo:
        rot, ckpt = rm.split("=", 1)
        g = GeradorDiffusionPen(ckpt, a.style, a.device)
        dev, abar = g.device, g.ddim.alphas_cumprod.to(g.device)
        R = defaultdict(list)
        canine = defaultdict(list)
        for b0 in range(0, len(pares), LOTE):
            lote = pares[b0:b0 + LOTE]
            ia = torch.stack([TRANSFORM(p[0]) for p in lote]).to(dev)
            ip = torch.stack([TRANSFORM(p[1]) for p in lote]).to(dev)
            with torch.no_grad():
                za = g.vae.encode(ia).latent_dist.mean * 0.18215
                zp = g.vae.encode(ip).latent_dist.mean * 0.18215
                estilo = g.feat(torch.stack([t for k in range(len(lote)) for t in refs[b0 + k]]).to(dev))
            forma = za.shape[-2:]
            mk = [mascara_latente(p[0], p[1], forma) for p in lote]
            m = torch.from_numpy(np.stack([x[0] for x in mk])).to(dev)          # (B,8,32)
            R["area_acento_latente"] += [float(x[0].mean()) for x in mk]
            R["area_acento_pixels"] += [float(x[1].mean()) for x in mk]
            dz2 = (za - zp) ** 2
            R["dz2_total"] += dz2.mean(dim=(1, 2, 3)).tolist()
            dentro = (dz2 * m[:, None]).sum(dim=(1, 2, 3)) / dz2.sum(dim=(1, 2, 3))
            R["fracao_dz2_na_mascara"] += dentro.tolist()

            tok = lambda txt: g.tokenizer(list(txt), padding="max_length", truncation=True,  # noqa: E731
                                          return_tensors="pt", max_length=g.texto_max_len).to(dev)
            ta, tp = tok([p[2] for p in lote]), tok([p[3] for p in lote])
            # palavra de controle: outro rotulo acentuado do mesmo lote (rotacao)
            to = tok([lote[(k + 1) % len(lote)][2] for k in range(len(lote))])

            # 3. CANINE e text_lin
            with torch.no_grad():
                enc, lin = g.ema.module.text_encoder, g.ema.module.text_lin
                ha, hp, ho = (enc(**t).last_hidden_state for t in (ta, tp, to))
                for nome, (xa, xp, xo) in (("canine", (ha, hp, ho)), ("text_lin", (lin(ha), lin(hp), lin(ho)))):
                    va, vp, vo = (x.mean(dim=1) for x in (xa, xp, xo))
                    cos = torch.nn.functional.cosine_similarity
                    canine[f"{nome}_cos_acentuada_vs_esqueleto"] += cos(va, vp).tolist()
                    canine[f"{nome}_cos_acentuada_vs_outra"] += cos(va, vo).tolist()
                    canine[f"{nome}_dist_rel_esqueleto"] += ((va - vp).norm(dim=1) / va.norm(dim=1)).tolist()
                    canine[f"{nome}_dist_rel_outra"] += ((va - vo).norm(dim=1) / va.norm(dim=1)).tolist()

            # 1b e 2, por t
            gen = torch.Generator(device="cpu").manual_seed(a.seed * 7 + b0)
            for t in TS:
                tt = torch.full((len(lote),), t, device=dev, dtype=torch.long)
                snr = float(abar[t] / (1 - abar[t]))
                R[f"t{t}_penalidade_sem_acento"] += (snr * dz2.mean(dim=(1, 2, 3))).tolist()
                for _ in range(RUIDOS_POR_T):
                    eps = torch.randn(za.shape, generator=gen).to(dev)
                    for img, z, certo, errado in (("acentuada", za, ta, tp), ("par", zp, tp, ta)):
                        xt = g.ddim.add_noise(z, eps, tt)
                        with torch.no_grad():
                            e_c = g.ema(xt, timesteps=tt, context=certo, y=torch.zeros_like(tt), style_extractor=estilo)
                            e_e = g.ema(xt, timesteps=tt, context=errado, y=torch.zeros_like(tt), style_extractor=estilo)
                        lc, le = (eps - e_c) ** 2, (eps - e_e) ** 2
                        sens = (e_c - e_e) ** 2
                        R[f"t{t}_{img}_perda_texto_certo"].append(float(lc.mean()))
                        R[f"t{t}_{img}_perda_texto_errado"].append(float(le.mean()))
                        R[f"t{t}_{img}_perda_certo_na_mascara"].append(media_em(lc, m))
                        R[f"t{t}_{img}_perda_errado_na_mascara"].append(media_em(le, m))
                        R[f"t{t}_{img}_sens_na_mascara"].append(media_em(sens, m))
                        R[f"t{t}_{img}_sens_fora_mascara"].append(media_em(sens, ~m))
            print(f"  {rot}: {min(b0 + LOTE, len(pares))}/{len(pares)}", flush=True)
        relatorio["modelos"][rot] = {"ckpt": ckpt,
                                     **{k: float(np.nanmean(v)) for k, v in R.items()},
                                     **{k: float(np.mean(v)) for k, v in canine.items()}}
        del g
        torch.cuda.empty_cache()

    with open(os.path.join(a.saida, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    # relatorio legivel
    rots = list(relatorio["modelos"])
    M = relatorio["modelos"]
    L = [f"# Peso do acento na perda — {len(pares)} pares da base {os.path.basename(a.base.rstrip('/'))}", ""]
    r0 = M[rots[0]]
    L += [f"Pares cuja imagem acentuada ganhou margem (o sinal passou da borda): {com_margem} de "
          f"{len(pares)}. Aqui eles foram realinhados; no treino entram em escalas diferentes.", ""]
    L += ["## 1. Quanto do latente o acento ocupa (depende so dos dados)", "",
          f"- area do acento: {r0['area_acento_pixels']:.2%} dos pixels; "
          f"{r0['area_acento_latente']:.2%} das posicoes do latente 8x32",
          f"- da diferenca entre os latentes com e sem acento, {r0['fracao_dz2_na_mascara']:.0%} "
          f"cai nas posicoes do acento (o resto e o VAE espalhando)", ""]
    L += ["## 1b. Quanto a perda sobe se o modelo desenhar SEM acento (penalidade) "
          "contra a perda tipica do modelo com o texto certo", "",
          "| t | penalidade | " + " | ".join(f"perda {r}" for r in rots) + " | " +
          " | ".join(f"razao {r}" for r in rots) + " |",
          "|---|---|" + "---|" * (2 * len(rots))]
    for t in TS:
        pen = r0[f"t{t}_penalidade_sem_acento"]
        perdas = [M[r][f"t{t}_acentuada_perda_texto_certo"] for r in rots]
        L.append(f"| {t} | {pen:.5f} | " + " | ".join(f"{p:.4f}" for p in perdas) + " | " +
                 " | ".join(f"{pen / p:.1%}" for p in perdas) + " |")
    L += ["", "## 2. O modelo usa o diacritico do texto? (imagem acentuada; texto certo x errado)", "",
          "| modelo | t | perda certo | perda errado | aumento | na mascara: certo | na mascara: errado | "
          "sensibilidade na mascara | sensibilidade fora | razao dentro/fora |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rots:
        for t in TS:
            x = M[r]
            pc, pe = x[f"t{t}_acentuada_perda_texto_certo"], x[f"t{t}_acentuada_perda_texto_errado"]
            sd, sf = x[f"t{t}_acentuada_sens_na_mascara"], x[f"t{t}_acentuada_sens_fora_mascara"]
            L.append(f"| {r} | {t} | {pc:.4f} | {pe:.4f} | {(pe - pc) / pc:+.2%} | "
                     f"{x[f't{t}_acentuada_perda_certo_na_mascara']:.4f} | "
                     f"{x[f't{t}_acentuada_perda_errado_na_mascara']:.4f} | {sd:.2e} | {sf:.2e} | "
                     f"{sd / sf if sf > 0 else float('nan'):.1f}x |")
    L += ["", "## 3. O codificador de texto distingue com e sem acento?", "",
          "| modelo | camada | cos(acentuada, esqueleto) | cos(acentuada, outra palavra) | "
          "dist. rel. ao esqueleto | dist. rel. a outra |", "|---|---|---|---|---|---|"]
    for r in rots:
        for c in ("canine", "text_lin"):
            x = M[r]
            L.append(f"| {r} | {c} | {x[f'{c}_cos_acentuada_vs_esqueleto']:.4f} | "
                     f"{x[f'{c}_cos_acentuada_vs_outra']:.4f} | {x[f'{c}_dist_rel_esqueleto']:.3f} | "
                     f"{x[f'{c}_dist_rel_outra']:.3f} |")
    texto = "\n".join(L) + "\n"
    open(os.path.join(a.saida, "resultado.md"), "w", encoding="utf-8").write(texto)
    print(texto)


if __name__ == "__main__":
    main()
