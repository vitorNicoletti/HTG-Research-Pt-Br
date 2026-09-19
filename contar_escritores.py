import sys
from utils.word_dataset import WordLineDataset

BASEFOLDER = "/home/leo/repos/htg-tcc/bressay_split"

ds = WordLineDataset(BASEFOLDER, "train", transforms=None)
print("amostras:", len(ds))
print("escritores:", len(ds.wid2idx))
