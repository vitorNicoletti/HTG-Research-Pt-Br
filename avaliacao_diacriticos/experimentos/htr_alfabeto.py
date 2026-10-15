"""
htr_alfabeto.py -- portao 0 do experimento A: o modelo PODE escrever a marca?

Antes de baixar peso e rodar inferencia, da para eliminar candidato de graca
lendo o alfabeto que ele produz. Um reconhecedor cujo alfabeto nao tem "a com
til" nunca vai escrever "a com til", qualquer que seja o CER. Reprovado por
construcao, nao por medida.

Dois tipos de alfabeto:

  explicito  PyLaia (syms.txt) e Paddle (inference.yml) listam os simbolos de
             saida um a um. A ausencia e definitiva.
  byte-level o TrOCR usa BPE sobre bytes, que REPRESENTA qualquer Unicode.
             Ali a ausencia nao prova nada sobre capacidade; o que da para ver
             e se o vocabulario aprendido tem tokens com a marca, o que indica
             que o texto de treino tinha. E indicio, nao portao.

    python avaliacao_diacriticos/experimentos/htr_alfabeto.py
"""
import json
import sys
import unicodedata

MODELOS = [
    ("Teklia/pylaia-rimes", "syms.txt", "explicito"),
    ("Teklia/pylaia-belfort", "syms.txt", "explicito"),
    ("Teklia/pylaia-iam", "syms.txt", "explicito"),
    ("PaddlePaddle/latin_PP-OCRv5_mobile_rec", "inference.yml", "explicito"),
    ("agomberto/trocr-large-handwritten-fr", "vocab.json", "byte-level"),
    ("qantev/trocr-base-spanish", "vocab.json", "byte-level"),
    ("microsoft/trocr-base-handwritten", "vocab.json", "byte-level"),
]

# o que o portugues precisa, por marca
PRECISA = {
    "til": "ãõÃÕ",
    "cedilha": "çÇ",
    "agudo": "áéíóúÁÉÍÓÚ",
    "grave": "àÀ",
    "circunflexo": "âêôÂÊÔ",
}


def byte_decoder():
    """Inverso do alfabeto byte-level do GPT-2/RoBERTa."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def texto_dos_tokens(vocab):
    """Decodifica as chaves byte-level para texto de verdade."""
    dec = byte_decoder()
    saida = set()
    for tok in vocab:
        try:
            bruto = bytes(dec[c] for c in tok)
        except KeyError:
            continue
        saida.add(bruto.decode("utf-8", errors="ignore"))
    return saida


def simbolos(caminho, tipo):
    if tipo == "explicito":
        txt = open(caminho, encoding="utf-8").read()
        if caminho.endswith(".yml"):
            # o dict do Paddle vem como lista no yaml; pegar tudo entre aspas
            import re
            itens = re.findall(r"- ['\"]?(.{1,4}?)['\"]?\s*$", txt, re.M)
            return set("".join(itens))
        return set("".join(l.split()[0] for l in txt.splitlines() if l.split()))
    vocab = json.load(open(caminho, encoding="utf-8"))
    return set("".join(texto_dos_tokens(vocab)))


def main():
    from huggingface_hub import hf_hub_download
    print(f"{'modelo':42s}{'tipo':12s}" +
          "".join(f"{m[:11]:>13s}" for m in PRECISA))
    print("-" * (54 + 13 * len(PRECISA)))
    for rid, arq, tipo in MODELOS:
        try:
            caminho = hf_hub_download(rid, arq)
            alf = simbolos(caminho, tipo)
        except Exception as e:
            print(f"{rid:42s}{tipo:12s}  ERRO {type(e).__name__}: {str(e)[:40]}")
            continue
        cel = []
        for marca, chars in PRECISA.items():
            tem = sum(1 for c in chars if c in alf)
            cel.append(f"{tem}/{len(chars)}")
        print(f"{rid:42s}{tipo:12s}" + "".join(f"{c:>13s}" for c in cel))

    print("\nexplicito: 0/N e reprovacao definitiva naquela marca.")
    print("byte-level: 0/N e so indicio de que o texto de treino nao tinha a")
    print("marca; o modelo PODE representa-la, entao o teste tem de ser empirico.")


if __name__ == "__main__":
    main()
