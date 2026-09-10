import sys, numpy as np, heapq, math
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, med
from PIL import Image
import os, time

def huffman_lens(counts):
    # counts: dict sym->count; returns dict sym->len via Huffman tree; single-symbol => len 1
    if len(counts)==1: return {s:1 for s in counts}
    heap=[(c,s) for s,c in counts.items()]; heapq.heapify(heap)
    # tree via merging with node ids
    nxt=1<<30
    H=heap[:]
    parent={}
    while len(H)>1:
        c1,n1=heapq.heappop(H); c2,n2=heapq.heappop(H)
        nn=nxt; nxt+=1
        parent[n1]=(nn,0); parent[n2]=(nn,1)
        heapq.heappush(H,(c1+c2,nn))
    lens={}
    for s in counts:
        d=0; n=s
        while n in parent: n=parent[n][0]; d+=1
        lens[s]=d
    return lens

def exact_huffman_channel(flat):
    vals,cn=np.unique(flat,return_counts=True)
    counts={int(v):int(c) for v,c in zip(vals,cn)}
    lens=huffman_lens(counts)
    data_bits=sum(counts[s]*lens[s] for s in counts)
    A=len(counts)
    # table: 16b A + per symbol 16b value + 8b len (canonical codes implicit)
    table_bits=16+A*24
    return data_bits+table_bits, A, lens

def res_medonly(arr):
    H,W,_=arr.shape
    P=np.pad(arr,((1,0),(1,0),(0,0)),mode='edge').astype(np.int16)
    res=np.zeros_like(arr)
    for i in range(H):
        for j in range(W):
            for ch in range(3):
                res[i,j,ch]=int(P[i+1,j+1,ch])-med(int(P[i+1,j,ch]),int(P[i,j+1,ch]),int(P[i,j,ch]))
    return res

d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in ["kodim01.png","kodim02.png","kodim05.png","kodim07.png","kodim13.png","kodim19.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    t0=time.perf_counter()
    r=res_medonly(rgb_to_ycocg_r(rgb))
    tb=0; det=[]
    for ch in range(3):
        b,A,_=exact_huffman_channel(r[:,:,ch].reshape(-1)); tb+=b; det.append(A)
    tot=(tb+48*8+48+7)//8; bpp=tot*8/(H*W*3)
    print(f"{fn} HUFF-EXACT bpp={bpp:.2f} bytes={tot} alph={det} t={time.perf_counter()-t0:.1f}s",flush=True)