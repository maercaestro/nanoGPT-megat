"""
Di sini class Tokenizer di define terlebih dahulu. Termasuk beberapa fungsi pembantu (helper function) seperti
load, train dan save.

oleh Abu Huzaifah Bidin
maercaestro@gmail.com
"""

import unicodedata

def get_stats (ids, counts=None):
    """Fungsi ini akan melihat pada seluruh list nombor, dan cuba untuk
    mencari pasangan yang paling kerap berulang. Dia memerlukan senarai integer
    sebagai input, dan akan mengeluarkna senarai integer juga sebagai output"""

    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair,idx):
    """
    Fungsi ini akan mencari pasangan yang berulang dan akan menggantikan pasangan itu 
    dengan nombor yang baru. Di akhir fungsi, satu senarai yang baru akan dipulangkan kepada user
    """
     
    newids = []
    i = 0
    while i < len(ids):
        #jika tidak berapa di penghujung list dan ada pair yang sepadan, gantikan
        if ids[i] == pair[0] and i < len(ids)-1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        
        else:
            newids.append(ids[i])
            i += 1

    return newids

#fungsi pembantu (helper function)
def replace_control_char(s: str) -> str:
    #kita tak mahu print kan special characters macam \n
    #jadi kita gantikan dengan " "

    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    #print token dengan cantik, escaping control character

    s = t.decode('utf-8',errors= 'replace')
    s= replace_control_char(s)

    return s

# ________ class tokenizer yang asas seperti di bawah ____________

class Tokenizer:
    """ classs tokenizer yang paling asas"""

    def __init__(self) -> None:
        #default : vocab size 256, takde merge, takde pattern
        self.merges = {} #(int,int) ->int
        self.pattern = " " #str
        self.special_tokens = {} #str -> int (contoh: '<|penghujung|>' : 100257)
        self.vocab = self.bina_vocab() # int -> bytes
    
    def train (self, text,vocab_size,verbose = False):
        #tokenizer boleh melatih satu vokabulari dari vocab_size from teks yang kita beri
        raise NotImplementedError
    
    def encode (self, text):
        #tokenizer boleh encode satu senarai teks
        raise NotImplementedError
    
    def decode (self,ids):
        #tokenizer boleh decode list integer kepada teks
        raise NotImplementedError
    
    def bina_vocab(self):
        #vocab ditentukan oleh senarai yang dihasilkan merge
        vocab = {idx : bytes([idx]) for idx in range (256)}
        #ini yang ambil senarai merge
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        #ini yang ambil special token
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        
        return vocab
    
    def save(self, file_prefix):
        """
        Fungsi ini akan save dua file, satu .vocab satu lagi .model
        .vocab untuk memudahkan manusia untuk melihat. 
        .model yang kita akan gunakan untuk load nanti
        
        """
        #tulis dalam model, untuk digunakan masa load nanti
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("maerctoken v1\n")
            f.write(f"{self.pattern}\n")

            #tulis special token dulu
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx} \n")
            
            #senarai merge
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2} \n")
            
            #tulis vocab, untuk panduan manusia
            vocab_file = file_prefix +".vocab"
            inverted_merges = {idx: pair for pair, idx in self.merges.items()}
            with open(vocab_file,"w",encoding="utf-8") as f:
                for idx,token in self.vocab.items():
                    s  = render_token(token)
                    if idx in inverted_merges:
                        idx0, idx1 = inverted_merges[idx]
                        s0 = render_token(self.vocab[idx0])
                        s1 = render_token(self.vocab[idx1])
                        f.write(f"[{s0}][{s1}]-> [{s}]{idx} \n")
                    
                    else:
                        f.write(f"[{s}]{idx} \n")

    def load (self,model_file):
        """
        load kembali model yang dah di save kan tadi
        """

        assert model_file.endswith(".model")
        #baca model file
        merges = {}
        special_tokens ={}
        idx = 256
        with open(model_file,'r',encoding="utf-8") as f:
            #read version dulu
            version = f.readline().strip()
            assert version == "maerctoken v1"

            #read pattern pulak
            self.pattern = f.readline().strip()

            #baca special tokens
            num_special = int(f.readline().strip())

            for _ in range (num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            
            #baca merges pulak
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1,idx2)] = idx
                idx += 1
            
            self.merges = merges
            self.special_tokens = special_tokens
            self.vocab = self.bina_vocab()




    




