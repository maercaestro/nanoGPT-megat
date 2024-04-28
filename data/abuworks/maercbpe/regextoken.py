"""
Byte Pair Encoding yang menggunakan regex pattern oleh GPT 4. Secara asasnya inilah cara yang digunakan oleh OpenAI
untuk memproses teks dalam GPT 4. Dan kita boleh buat benda ni sendiri!!

 Oleh Abu huzaifah Bidin

"""

import regex as re
from .base import Tokenizer,get_stats,merge

#GPT punya split pattern seperti di bawah, sumber dari:
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern = None):
        """
        Fungsi initialization dahulu, di mana kita set kan dulu pattern default sebagai kosong  
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    

    def train(self,text,vocab_size,verbose = False):
        assert vocab_size >= 256
        num_merges = vocab_size -256

        #split teks kepada kepingan yang kecil
        text_chunks = re.findall(self.compiled_pattern,text)

        #ambil teks untuk diproses
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        #secara iterative pergi ke setiap element dan cari pasangan sepadan untuk dicampurkan
        merges = {} #(int,int)->int
        vocab = {idx: bytes([idx])for idx in range(256)} #int -> bytes
        for i in range(num_merges) :
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids,stats)
            #cari pasangan dengan kiraan tertinggi
            pair = max(stats,key=stats.get)
            #bagi nama/token baru pada pasangan
            idx = 256+i
            #gantikan semua dengan ids
            ids = [merge(chunk_ids,pair,idx) for chunk_ids in ids]
            #simpan senarai merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            #prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair}->{idx}({vocab[idx]} ada {stats[pair]})")

        #simpan variables
        self.merges = merges
        self.vocab = vocab
    
    def register_special_tokens(self,special_tokens):
        #special token ni untuk handle str -> int
        #contoh : {"<|endoftext|>"}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k,v in special_tokens.items()}

    def decode(self,ids):
        #diberi list integer, tukarkan kepada perkataan
        part_bytes =  []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id:{idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8",errors="replace")
        return text
    
    def _encode_chunk(self,text_bytes):
        #pulangkan token ids
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key = lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids,pair,idx)
        
        return ids
    
    def encode_biasa(self,text):
        """
        Biasa tu macam mana? Biasa tu maksudnya kita tak ambil peduli special character
        """

        #pisahkan teks mengikut kepingan yang kita dah initialize dalam pattern tadi
        text_chunks = re.findall(self.compiled_pattern,text)
        #setiap kepingan teks akan diencode secara asing barulah digabungkan
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        
        return ids
    def encode (self, text, allowed_special = "none_raise"):
        """
        Fungsi ini akan mengambil kira special character sekali
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special,set):
            special = {k: v  for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special ={allowed_special} not understood")
        if not special:
            #kalau tak special, kita encode gunakan cara biasa
            return self.encode_biasa(text)
        
        special_pattern = "("+"|".join(re.escape(k)for k in special)+")"
        special_chunks = re.split(special_pattern,text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_biasa(part))
        
        return ids