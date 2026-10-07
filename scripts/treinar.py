"""Entrada unica do fine-tune: le um arquivo de experimento e roda o treino.

Todos os parametros de execucao ficam no JSON do experimento (ver
experimentos/bressay_25.json). Nao ha valor padrao escondido: toda chave e
obrigatoria, e chave desconhecida e erro -- o arquivo e a lista completa do que
foi usado.

    python scripts/treinar.py experimentos/bressay_25.json
    python scripts/treinar.py experimentos/bressay_25.json --dry-run

dados.dataset escolhe o leitor do train.py:
  bressay        -- dados.split e a pasta do split (splits/*.tsv), dados.imagens
                    as imagens do BRESSAY, dados.preproc v1 ou v2;
  iam_acentuado  -- dados.split e a pasta da base de scripts/gerar_base_acentos.py,
                    dados.imagens as palavras do IAM (iam_data/words),
                    dados.preproc "iam" (o do IAMDataset) e dados.iam_originais
                    a fracao das palavras originais que entra junto;
                    treino.peso_acento (>= 1) multiplica o erro de ruido na
                    regiao do acento e treino.peso_zona (>= 1) nas celulas
                    vazias acima/abaixo do corpo (as duas exigem base com
                    pares alinhados).

O que faz, na ordem:
  1. valida o JSON e sincroniza o clone (scripts/aplicar_mods.sh);
  2. grava uma copia do experimento em SAVE_PATH/experimento.json. Se ja
     existir uma diferente, so aceita mudancas que nao alteram o que foi
     treinado (epocas_total, execucao, amostras); o resto exige um save_path
     novo;
  3. treina em blocos de epocas_por_bloco ate epocas_total. O primeiro bloco
     parte dos pesos_iniciais; os seguintes retomam do checkpoint. Codigo de
     saida 3 (--abort_after) relanca do ultimo checkpoint;
  4. ao fim de cada bloco: snapshot do EMA, deriva (medir_deriva.py) e
     amostras (gerar_amostras.py) em SAVE_PATH/amostras/<N>ep.

Tudo vai para dentro de SAVE_PATH: pesos, experimento.json, config.jsonl
(gravado pelo train.py a cada lancamento), treino.log e amostras.

Criterio de parada: deriva e amostras, nao o numero de epocas nem o MSE
(ver CLAUDE.md). O treino pode ser interrompido e relancado com o mesmo
comando; ele continua da ultima epoca completa.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Esquema do experimento: secao -> {chave: tipo}. Toda chave e obrigatoria.
NUM = (int, float)
ESQUEMA = {
    "descricao": str,
    "save_path": str,
    "dados": {"dataset": str, "split": str, "imagens": str, "max_samples": int,
              "preproc": str, "iam_originais": NUM},
    "modelo": {"pesos_iniciais": str, "extrator_estilo": str,
               "stable_diffusion": str},
    "treino": {"epocas_por_bloco": int, "epocas_total": int, "lr": NUM,
               "batch_size": int, "adamw_eps": NUM, "clip_grad_norm": NUM,
               "ema_beta": NUM, "ema_inicio": int, "texto_max_len": int,
               "peso_acento": NUM, "peso_zona": NUM, "zona": str},
    "execucao": {"device": str, "num_workers": int, "save_every_steps": int,
                 "abort_after": int},
    "amostras": {"palavras": list, "estilos": int, "seed": int,
                 "estilo_de": str},
}

# Valores de dados.preproc aceitos por dataset
PREPROCS = {"bressay": ("v1", "v2"), "iam_acentuado": ("iam",)}

# Chaves acrescentadas depois dos primeiros runs: um experimento.json gravado
# antes delas equivale a estes valores (todos eram do BRESSAY).
LEGADO = {"dados.dataset": "bressay", "dados.iam_originais": 0,
          "treino.peso_acento": 1, "treino.peso_zona": 1, "treino.zona": "vazia"}

# Mudancas aceitas ao retomar um save_path existente: nao alteram o que o
# modelo ja aprendeu nem como aprende.
PODE_MUDAR_AO_RETOMAR = {"descricao", "treino.epocas_total", "execucao",
                         "amostras"}


def validar(exp, esquema=ESQUEMA, prefixo=""):
    erros = []
    for chave, tipo in esquema.items():
        nome = prefixo + chave
        if chave not in exp:
            erros.append(f"falta a chave '{nome}'")
        elif isinstance(tipo, dict):
            if not isinstance(exp[chave], dict):
                erros.append(f"'{nome}' tem de ser um objeto")
            else:
                erros += validar(exp[chave], tipo, nome + ".")
        elif isinstance(exp[chave], bool) or not isinstance(exp[chave], tipo):
            erros.append(f"'{nome}' tem tipo errado: {exp[chave]!r}")
    for chave in exp:
        if chave not in esquema:
            erros.append(f"chave desconhecida '{prefixo + chave}'")
    if not prefixo and not erros:
        d = exp["dados"]
        if d["dataset"] not in PREPROCS:
            erros.append(f"dados.dataset desconhecido: {d['dataset']!r} (use {sorted(PREPROCS)})")
        elif d["preproc"] not in PREPROCS[d["dataset"]]:
            erros.append(f"dados.preproc {d['preproc']!r} nao vale para {d['dataset']} "
                         f"(use {PREPROCS[d['dataset']]})")
        if not 0 <= d["iam_originais"] <= 1:
            erros.append("dados.iam_originais tem de estar em [0, 1]")
        if exp["treino"]["zona"] not in ("vazia", "vogais"):
            erros.append("treino.zona tem de ser 'vazia' ou 'vogais'")
        for nome in ("peso_acento", "peso_zona"):
            p = exp["treino"][nome]
            if p < 1:
                erros.append(f"treino.{nome} tem de ser >= 1 (1 = loss original)")
            elif p != 1 and d["dataset"] != "iam_acentuado":
                erros.append(f"treino.{nome} != 1 so vale para dados.dataset iam_acentuado")
    return erros


def achatar(d, prefixo=""):
    saida = {}
    for k, v in d.items():
        if isinstance(v, dict):
            saida.update(achatar(v, f"{prefixo}{k}."))
        else:
            saida[prefixo + k] = v
    return saida


def conferir_retomada(exp, caminho):
    """Compara com o experimento ja gravado no save_path. Devolve a lista de
    mudancas proibidas (vazia = pode seguir)."""
    with open(caminho, encoding="utf-8") as f:
        antigo = achatar(json.load(f))
    for k, v in LEGADO.items():
        antigo.setdefault(k, v)
    novo = achatar(exp)
    proibidas = []
    for k in sorted(set(antigo) | set(novo)):
        if antigo.get(k) == novo.get(k):
            continue
        linha = f"  {k}: {antigo.get(k)!r} -> {novo.get(k)!r}"
        livre = k in PODE_MUDAR_AO_RETOMAR or k.split(".")[0] in PODE_MUDAR_AO_RETOMAR
        print(("  (ok) " if livre else "  (PROIBIDA) ") + linha.strip())
        if not livre:
            proibidas.append(k)
    return proibidas


def epocas_feitas(save_path):
    estado = os.path.join(save_path, "models", "estado.pt")
    if not os.path.isfile(estado):
        return 0
    import torch
    return torch.load(estado, map_location="cpu", weights_only=True)["epoch"] + 1


def ambiente(exp):
    env = os.environ.copy()
    env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    if exp["dados"]["dataset"] == "bressay":
        env["BRESSAY_IMAGES"] = exp["dados"]["imagens"]
    else:
        env["IAM_IMAGES"] = os.path.abspath(exp["dados"]["imagens"])
    env["SD"] = exp["modelo"]["stable_diffusion"]
    # A RX 6600 XT (gfx1032) so roda com kernels de gfx1030; em qualquer outra
    # placa o override e nocivo. Mesmo criterio do antigo treinar.sh.
    if "HSA_OVERRIDE_GFX_VERSION" not in env and shutil.which("rocminfo"):
        r = subprocess.run(["rocminfo"], capture_output=True, text=True)
        if "gfx1032" in r.stdout:
            env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
            print("placa gfx1032 detectada: HSA_OVERRIDE_GFX_VERSION=10.3.0")
    return env


def cmd_treino(exp, n, primeiro):
    t, e, d, m = exp["treino"], exp["execucao"], exp["dados"], exp["modelo"]
    cmd = [
        sys.executable, "DiffusionPen/train.py",
        # fixos: e o que define este pipeline, nao sao escolhas do experimento
        "--dataset", d["dataset"],
        "--model_name", "diffusionpen",
        "--sample_every", "0",   # a grade interna usa max_length=200; nao vale
        # do experimento
        "--save_path", exp["save_path"],
        "--dataset_folder", d["split"],
        "--max_samples", str(d["max_samples"]),
        "--style_path", m["extrator_estilo"],
        "--stable_dif_path", m["stable_diffusion"],
        "--lr", repr(t["lr"]),
        "--batch_size", str(t["batch_size"]),
        "--adamw_eps", repr(t["adamw_eps"]),
        "--clip_grad_norm", repr(t["clip_grad_norm"]),
        "--ema_beta", repr(t["ema_beta"]),
        "--ema_inicio", str(t["ema_inicio"]),
        "--texto_max_len", str(t["texto_max_len"]),
        "--device", e["device"],
        "--num_workers", str(e["num_workers"]),
        "--save_every_steps", str(e["save_every_steps"]),
        "--abort_after", str(e["abort_after"]),
        "--epochs", str(n),
    ]
    # --pretrained_path e --load_check sao mutuamente exclusivos: no train.py o
    # pretrained_path roda depois e sobrescreveria o que o load_check retomou.
    # E --load_check e type=bool: nunca passar 'False', so omitir.
    if d["dataset"] == "bressay":
        cmd += ["--preproc", d["preproc"]]
    else:
        cmd += ["--iam_originais", repr(d["iam_originais"]),
                "--peso_acento", repr(t["peso_acento"]),
                "--peso_zona", repr(t["peso_zona"]),
                "--zona", t["zona"]]
    if primeiro:
        cmd += ["--pretrained_path", m["pesos_iniciais"]]
    else:
        cmd += ["--load_check", "True"]
    return cmd


def cmd_amostras(exp, ckpt, destino):
    a, d = exp["amostras"], exp["dados"]
    # referencias de estilo do BRESSAY usam o split e o pre-processamento dele;
    # com estilo_de "iam" o gerar_amostras.py usa o pre-processamento do IAM
    bressay = ["--preproc", d["preproc"], "--dataset_folder", d["split"]]         if d["dataset"] == "bressay" else []
    return [
        sys.executable, "scripts/gerar_amostras.py",
        "--ckpt", ckpt,
        "--out", destino,
        "--style", exp["modelo"]["extrator_estilo"],
        "--texto_max_len", str(exp["treino"]["texto_max_len"]),
        *bressay,
        "--styles", str(a["estilos"]),
        "--seed", str(a["seed"]),
        "--estilo_de", a["estilo_de"],
        "--device", exp["execucao"]["device"],
        "--palavras", *a["palavras"],
    ]


def rodar(cmd, env, log):
    """Roda cmd mostrando a saida e copiando para o log (como o tee)."""
    with open(log, "ab") as f:
        f.write(("\n$ " + " ".join(cmd) + "\n").encode())
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env)
        for pedaco in iter(lambda: p.stdout.read1(4096), b""):
            sys.stdout.buffer.write(pedaco)
            sys.stdout.buffer.flush()
            f.write(pedaco)
        return p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("experimento", help="arquivo JSON do experimento")
    ap.add_argument("--dry-run", action="store_true",
                    help="valida e imprime o comando do proximo bloco, sem "
                         "gravar nem treinar nada")
    cli = ap.parse_args()

    with open(cli.experimento, encoding="utf-8") as f:
        exp = json.load(f)
    erros = validar(exp)
    if erros:
        sys.exit("experimento invalido:\n  " + "\n  ".join(erros))

    os.chdir(RAIZ)
    save_path = exp["save_path"]
    t = exp["treino"]
    for caminho in (exp["dados"]["split"], exp["dados"]["imagens"],
                    exp["modelo"]["extrator_estilo"],
                    exp["modelo"]["pesos_iniciais"]):
        if not os.path.exists(caminho):
            sys.exit(f"nao encontrado: {caminho}")

    gravado = os.path.join(save_path, "experimento.json")
    if os.path.isfile(gravado):
        print(f"retomando {save_path}; diferencas para o experimento gravado:")
        proibidas = conferir_retomada(exp, gravado)
        if proibidas:
            sys.exit("\nessas mudancas alteram o que esta sendo treinado. "
                     "Use um save_path novo para elas.")

    feitas = epocas_feitas(save_path)
    print(f"experimento: {exp['descricao']}")
    print(f"{feitas} de {t['epocas_total']} epocas feitas em {save_path}")

    if cli.dry_run:
        if feitas < t["epocas_total"]:
            n = min(t["epocas_por_bloco"], t["epocas_total"] - feitas)
            print("\nproximo bloco (dry-run):")
            print("  " + " ".join(cmd_treino(exp, n, feitas == 0)))
        return

    if subprocess.run(["bash", "scripts/aplicar_mods.sh"]).returncode != 0:
        sys.exit("nao consegui sincronizar o clone DiffusionPen/; parando")

    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    with open(gravado, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    shutil.copyfile(cli.experimento, os.path.join(save_path, "experimento_origem.json"))

    env = ambiente(exp)
    log = os.path.join(save_path, "treino.log")
    modelos = os.path.join(save_path, "models")

    while True:
        feitas = epocas_feitas(save_path)
        if feitas >= t["epocas_total"]:
            print(f"=== {feitas} epocas concluidas, teto de {t['epocas_total']} atingido ===")
            break
        n = min(t["epocas_por_bloco"], t["epocas_total"] - feitas)
        print(f"=== bloco: {feitas} epocas feitas, treinando mais {n} ===")

        codigo = rodar(cmd_treino(exp, n, feitas == 0), env, log)
        if codigo == 3:
            print("=== processo entrou no estado ruim; relancando do ultimo checkpoint ===")
            time.sleep(10)
            continue
        if codigo != 0:
            sys.exit(f"=== train.py saiu com codigo {codigo}; parando para voce olhar ===")

        agora = epocas_feitas(save_path)
        ema = os.path.join(modelos, "ema_ckpt.pt")
        shutil.copyfile(ema, os.path.join(modelos, f"ema_bloco_{agora}ep.pt"))
        print(f"=== snapshot: ema_bloco_{agora}ep.pt ===")

        rodar([sys.executable, "scripts/medir_deriva.py", ema], env, log)

        destino = os.path.join(save_path, "amostras", f"{agora}ep")
        print(f"=== gerando amostras de {agora} epocas em {destino} ===")
        rodar(cmd_amostras(exp, ema, destino), env, log)
        print(f"=== olhe {destino}: o criterio e o diacritico aparecer ===")


if __name__ == "__main__":
    main()
