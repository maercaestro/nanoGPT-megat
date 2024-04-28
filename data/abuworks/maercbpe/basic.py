"""
Cubaan pertama aku untuk buat tokenizer sendiri. Ini berdasarkan pada tutorial Andrej Kaparthy 
yang menunjukkan bagaiman algoritma byte-pair-encoding (BPE) berlaku 

Ini tokenizer yang paling asas, dan tidak menggunakan sebarang regex (regular expression)
ataupun karakter yang istimewa

"""

from .base import Tokenizer,get_stats,merge

class BasicTokenizer(Tokenizer):

    def __init__(self) -> None:
        super().__init__()

    
    def train(self,text,vocab_size,verbose=False):
        assert vocab_size >=256
        num_merges =vocab_size-256

        #ambil teks dan proses
        text_bytes = text.encode("utf-8") #raw bytes
        ids = list(text_bytes)

        #secara iterative pergi ke setiap element dan cari pasangan sepadan untuk dicampurkan
        merges = {} #(int,int)->int
        vocab = {idx: bytes([idx])for idx in range(256)} #int -> bytes
        for i in range(num_merges) :
            #kira berapa banyak kali setiap pasangan keluar
            stats = get_stats(ids)
            #cari pasangan dengan kiraan tertinggi
            pair = max(stats,key= stats.get)
            #bagi nama/token baru pada pasangan
            idx = 256+i
            #gantikan semua dengan ids
            ids = merge(ids,pair,idx)
            #simpan senarai merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            #prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair}->{idx}({vocab[idx]} ada {stats[pair]})")

        #simpan variables
        self.merges = merges
        self.vocab = vocab
    
    def decode(self,ids):
        #diberi satu set integer, dan ditukar jadi string
        text_bytes = b"".join(self.vocab[idx]for idx in ids)
        text = text_bytes.decode("utf-8",errors="replace")
        return text
    
    def encode(self,text):
        #bagi teks, dia pulangkan token
        text_bytes = text.encode("utf-8") #raw bytes
        ids = list(text_bytes) #senarai integer dari 0 - 255
        while len(ids) >= 2:
            #cari pasangan dengan indeks gabungan terendah
            stats = get_stats(ids)
            pair = min(stats, key = lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids,pair,idx)
        
        return ids