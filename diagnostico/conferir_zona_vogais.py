"""Confere a zona "vogais" (--zona vogais) ANTES de treinar com ela.

  1. integridade: zona disjunta do acento; acentuada e par com a mesma zona;
     nenhum px de tinta do alvo dentro da zona; quantas amostras ficaram sem
     mascara (alinhamento fraco, zona_vogais.json);
  2. HASTES: fracao das celulas da zona que caem sobre letras com haste ou
     perna (b d f h k l t / g j p q y), comparada com a zona "vazia" antiga
     nas mesmas imagens (ACHADOS 15: ela encurtou as hastes);
  3. COBERTURA DO VAZAMENTO: nas palavras sem acento pedido do ultimo modelo
     avaliado (--paineis), fracao das marcas falsas cujo centro cai na faixa
     das vogais/c (sem excluir tinta) -- o que a zona passaria a cobrar;
  4. peso na loss do lote, com os pesos pedidos;
  5. figura: zona (azul), acento (vermelho), fronteiras das letras (cinza).
Carrega o dataset de diffusionpen_mods/ (nao do clone).

    python diagnostico/conferir_zona_vogais.py --base iam_acentuado_teto25_pares \
        --paineis avaliacao_pares_val:pares_12ep --saida diagnostico/resultados/zona_vogais
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
CLONE = os.path.join(RAIZ, "DiffusionPen")
sys.path.insert(0, CLONE)
esp = importlib.util.spec_from_file_location(
    "iam_acentuado_mods", os.path.join(RAIZ, "diffusionpen_mods", "utils", "iam_acentuado_dataset.py"))
mods = importlib.util.module_from_spec(esp)
esp.loader.exec_module(mods)
mods.SPLIT_ORIGINAIS = os.path.join(CLONE, "utils", "splits_words", "iam_train_val.txt")
mods.WRITERS_DICT = os.path.join(CLONE, "writers_dict_train.json")
os.environ.setdefault("IAM_IMAGES", os.path.join(CLONE, "iam_data", "words"))
from acentos_sinteticos import alinhamento, geometria, zona_vogais  # noqa: E402
from acentos_sinteticos.vocabulario import esqueleto  # noqa: E402
from medir_marcas import marcas  # noqa: E402

HASTE = set("bdfhkltgjpqyBDFHKLTGJPQY")


def painel(caminho, texto):
    return mods.preprocessar_iam(Image.open(caminho).convert("RGB"), texto)


def celulas_por_letra(z, fats, texto):
    """Conta celulas da zona z (8,32) pelo tipo da letra sob o centro da celula."""
    h, w = z.shape
    c = Counter()
    for y, x in zip(*np.nonzero(z)):
        cx = (x + 0.5) * 256 / w
        k = next((i for i, (a, b) in enumerate(fats) if a <= cx < b), None)
        if k is None:
            c["fora_da_palavra"] += 1
        elif texto[k] in HASTE:
            c["haste"] += 1
        elif texto[k] in zona_vogais.VOGAIS or texto[k] in zona_vogais.CEDILHAVEIS:
            c["vogal_ou_c"] += 1
        else:
            c["outra"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--peso_acento", type=float, default=5.0)
    ap.add_argument("--peso_zona", type=float, default=5.0)
    ap.add_argument("--iam_originais", type=float, default=1.0)
    ap.add_argument("--n_originais", type=int, default=3000)
    ap.add_argument("--paineis", default="", help="pasta_avaliacao:rotulo para medir a cobertura do vazamento")
    ap.add_argument("--alinhador", default=os.path.join(RAIZ, "modelos/alinhador_iam.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    rnd = random.Random(a.seed)
    alin = alinhamento.Alinhador(a.alinhador, "cpu")

    args = SimpleNamespace(peso_acento=a.peso_acento, peso_zona=a.peso_zona, zona="vogais",
                           iam_originais=a.iam_originais, max_samples=0)
    ds = mods.IAMAcentuadoDataset(a.base, transforms=None, args=args)
    man = [json.loads(l) for l in open(os.path.join(a.base, "manifesto.jsonl"), encoding="utf-8")]
    tipo = {os.path.join(a.base, x["arquivo"]): x["tipo"] for x in man}
    originais = [d for d in ds.data if not d[3]]
    amostra_orig = rnd.sample(originais, min(a.n_originais, len(originais)))
    for c, *_ in amostra_orig:
        tipo[c] = "original_iam"

    # 1. integridade e cobertura
    cob, tinta_na_zona, problemas, zonas, figura = {}, Counter(), [], {}, []
    for c, _, t, sint in [d for d in ds.data if d[3]] + amostra_orig:
        tp = tipo[c]
        ac = ds.mascara(c)
        z = ds.zona(c, t, sint, ac)
        cob.setdefault(tp, []).append(float(z.mean()))
        if (z * ac).any():
            problemas.append(f"zona encosta no acento: {c}")
        g = np.asarray(painel(c, t).convert("L"), dtype=np.float32)
        zpx = np.kron(z.numpy(), np.ones((8, 8))) > 0
        tinta_na_zona[tp] += int((geometria.mascara_tinta(g) & zpx).sum())
        if tp in ("acentuada", "par"):
            zonas[c] = z
        figura.append((c, t, tp, g, ac.numpy() > 0, z.numpy() > 0))
    for c, ((ca, _), (cp, _)) in ds.pares.items():
        if c == ca and ca in zonas and cp in zonas and not torch.equal(zonas[ca], zonas[cp]):
            problemas.append(f"zona do par difere da acentuada: {ca}")

    # 2. hastes: zona vogais x zona vazia antiga, nas mesmas originais
    hastes = {"vogais": Counter(), "vazia": Counter()}
    for c, _, t, _ in amostra_orig:
        g_img = painel(c, t)
        g = np.asarray(g_img.convert("L"), dtype=np.float32)
        fats, _, motivo = zona_vogais.fatias_no_painel(g, t, alin)
        if fats is None:
            continue
        zv, _ = zona_vogais.mascara(g, t, alin)
        hastes["vogais"] += celulas_por_letra(zv, fats, t)
        hastes["vazia"] += celulas_por_letra(mods.mascara_zona(g_img), fats, t)
    frac_haste = {k: {**{kk: int(vv) for kk, vv in v.items()},
                      "fracao_haste": round(v["haste"] / max(1, sum(v.values())), 4)} for k, v in hastes.items()}

    # 3. cobertura do vazamento nos paineis do ultimo modelo
    vaz = Counter()
    if a.paineis:
        pasta, rot = a.paineis.rsplit(":", 1)
        res = json.load(open(os.path.join(pasta, "resumo.json"), encoding="utf-8"))
        k_de = {p: k for k, p in enumerate(res["palavras"])}
        for l in list(open(os.path.join(pasta, "paineis.tsv"), encoding="utf-8"))[1:]:
            m, p, tp, e, s, mk, *_ = l.rstrip("\n").split("\t")
            if m != rot or tp == "acentuada" or mk != "1":
                continue
            g = np.asarray(Image.open(os.path.join(pasta, "paineis", rot, f"{k_de[p]:03d}_{e}_{s}.png"))
                           .convert("L"), dtype=np.float32)
            faixa, motivo = zona_vogais.mascara(g, esqueleto(p), alin, excluir_tinta=False)
            if motivo != "ok":
                vaz["sem_alinhamento"] += 1
                continue
            for x, y, w, h, lado in marcas(g)[2]:
                cy, cx = int((y + h / 2) // 8), int((x + w / 2) // 8)
                vaz["marca_na_faixa" if faixa[min(cy, 7), min(cx, 31)] else "marca_fora"] += 1
    cobertura_vaz = round(vaz["marca_na_faixa"] / max(1, vaz["marca_na_faixa"] + vaz["marca_fora"]), 4)

    # 4. peso na loss
    n_tipo = {"original_iam": len(originais)}
    for c, _, _, sint in ds.data:
        if sint:
            n_tipo[tipo[c]] = n_tipo.get(tipo[c], 0) + 1
    cob_lote = sum(np.mean(v) * n_tipo[k] for k, v in cob.items()) / len(ds.data)
    cob_ac = sum(float(ds.mascara(c).mean()) for c, _, _, s in ds.data if s and c in ds.pares) / len(ds.data)
    um = 1 - cob_lote - cob_ac
    total = um + a.peso_acento * cob_ac + a.peso_zona * cob_lote

    motivos = json.load(open(os.path.join(a.base, "zona_vogais.json"), encoding="utf-8"))["motivos"]
    rel = {"base": a.base, "peso_acento": a.peso_acento, "peso_zona": a.peso_zona,
           "motivos_do_pre_calculo": motivos, "amostras": n_tipo,
           "zona_por_amostra": {k: {"media": round(float(np.mean(v)), 4), "vazia": round(float(np.mean(np.array(v) == 0)), 4)}
                                for k, v in cob.items()},
           "px_de_tinta_do_alvo_na_zona": dict(tinta_na_zona),
           "celulas_por_letra_nas_originais": frac_haste,
           "vazamento_do_modelo": {"paineis": a.paineis, **dict(vaz), "fracao_das_marcas_na_faixa": cobertura_vaz},
           "zona_no_lote": round(cob_lote, 4), "acento_no_lote": round(cob_ac, 4),
           "parte_da_loss_do_lote": {"zona": round(a.peso_zona * cob_lote / total, 4),
                                     "acento": round(a.peso_acento * cob_ac / total, 4), "resto": round(um / total, 4)},
           "n_problemas": len(problemas), "problemas": problemas[:30]}
    with open(os.path.join(a.saida, "zona_vogais.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)
    print(json.dumps(rel, ensure_ascii=False, indent=2))

    # 5. figura
    amostra = rnd.sample([f for f in figura if f[2] == "acentuada"], 10) + \
        rnd.sample([f for f in figura if f[2] == "original_iam"], 20)
    tela = Image.new("RGB", (2 * 264, 24 + 15 * 78), "white")
    d = ImageDraw.Draw(tela)
    d.text((4, 4), "azul = zona vogais (peso_zona); vermelho = acento; cinza = fronteira das letras", fill="black")
    for i, (c, t, tp, g, ac, z) in enumerate(amostra):
        x0, y0 = (i % 2) * 264, 24 + (i // 2) * 78
        im = Image.fromarray(g.astype(np.uint8)).convert("RGB")
        dd = ImageDraw.Draw(im, "RGBA")
        for m, cor in ((z, (0, 90, 255, 70)), (ac, (255, 0, 0, 80))):
            for yy, xx in zip(*np.nonzero(m)):
                dd.rectangle([xx * 8, yy * 8, xx * 8 + 7, yy * 8 + 7], fill=cor)
        texto = esqueleto(t) if tp == "acentuada" else t
        fats, _, _ = zona_vogais.fatias_no_painel(g, texto, alin)
        for a_, _ in (fats or [])[1:]:
            dd.line([(a_, 0), (a_, 63)], fill=(120, 120, 120, 160))
        tela.paste(im, (x0, y0 + 12))
        d.text((x0 + 2, y0), f"{tp}: {t}", fill="black")
    tela.save(os.path.join(a.saida, "zona_vogais.png"))
    if problemas:
        sys.exit(f"ERRO: {len(problemas)} problemas")
    print("ok")


if __name__ == "__main__":
    main()
