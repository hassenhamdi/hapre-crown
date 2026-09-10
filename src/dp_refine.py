"""Hill-climb refinement of greedy quantile groups with EXACT rate criterion (data Huffman + map side)."""
import numpy as np, sys, os, math
sys.path.insert(0,"/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
def exact_total(order_res, cuts, nactive, K):
    # order_res: list of residual arrays in sorted-ctx order; cuts: sorted cut positions (len = G-1)
    bounds=[0]+cuts+[len(order_res)]
    tot=nactive*math.ceil(math.log2(K))+4+8
    for a,b in zip(bounds[:-1],bounds[1:]):
        g=np.concatenate(order_res[a:b]) if b>a else np.array([],dtype=np.int32)
        if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); tot+=tb
    return tot
def refine(key, rf, groups, K, passes=2, radius=8):
    uk,cn=np.unique(key,return_counts=True)
    ma=np.array([np.abs(rf[key==v]).mean() for v in uk])
    order=np.argsort(ma,kind="stable")
    res_list=[rf[key==u] for u in uk[order]]
    # current cuts from groups (group sizes in ctx units): need mapping groups->ctx runs; groups are sets; convert to runs in sorted order
    # rebuild: assign each sorted position its group, cuts where group changes
    pos2g={}
    for gi,gkeys in enumerate(groups):
        for u in gkeys: pos2g[np.where(uk[order]==u)[0][0]]=gi
    seq=np.array([pos2g[i] for i in range(len(res_list))])
    cuts=sorted(np.where(np.diff(seq)!=0)[0].tolist())
    # NOTE: groups may be noncontiguous in sorted order (greedy is contiguous, ok)
    best=exact_total(res_list,cuts,len(uk),K)
    n=len(res_list)
    for _ in range(passes):
        improved=False
        for ci in range(len(cuts)):
            lo=max((cuts[ci-1]+1) if ci>0 else 1, cuts[ci]-radius)
            hi=min((cuts[ci+1]-1) if ci<len(cuts)-1 else n-2, cuts[ci]+radius)
            if hi<lo: continue
            for cand in range(lo,hi+1):
                if cand==cuts[ci]: continue
                trial=sorted(cuts[:ci]+[cand]+cuts[ci+1:])
                if len(set(trial))<len(trial): continue
                t=exact_total(res_list,trial,len(uk),K)
                if t<best: best=t; cuts=trial; improved=True
        if not improved: break
    # rebuild groups from cuts
    bounds=[0]+cuts+[n]
    uks=uk[order]
    return [uks[a:b].tolist() for a,b in zip(bounds[:-1],bounds[1:])], best