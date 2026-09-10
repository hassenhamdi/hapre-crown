# Double-MDL-gate rule as implemented — src/driver_crown6.py (factual extraction only)

Source: `/home/hassenhamdi/autocompress/src/driver_crown6.py` (1382 lines). Committed config probe_b21: `12feat→8→1, L1, LS-init, 200 iters @ lr 0.003 cosine, adaptive-int16 {256…4096}` (driver:9-11,160-167).

## Thresholds / constants (code)
- `MLP_NCTX=9` (driver:165), `MLP_NPAR=113` (driver:166), `MLP_SCALES=(256,512,1024,2048,4096)` (driver:167), `MLP_ITERS=200, MLP_LR=0.003, MLP_H=8, MLP_FEAT=12` (driver:161-164).
- Skip context with `<100` samples (driver:311-315): `if nk < 100: side += 1; continue  # fallback flag still transmitted (probe parity)`.
- Quant-clip asserts: `ma*scale <= 32767+1e-9` (driver:322), int16 range assert (driver:328).

## L1 per-(channel,context) stage — subset Huffman gate (driver:316-339)
> `316:        mb_ctx = B21.hbits(med_res[m])`
> `331:        qb = B21.hbits(P.reshape(-1)[m].astype(np.int32) - pv)   # WIRE forward pv = mlp_forward_wire(Ff[m], q) (driver:330)`
> `332:        if qb + MLP_NPAR * 16 + 1 + 3 < mb_ctx:  # L1 double-gate (probe verbatim)`
> `333-337:            win[k]=1; scid[k]=si; qw[k]=qi16; pred_vec[m]=pv; side += MLP_NPAR*16+1+3`
> `338-339:        else: side += 1`

Side-cost terms L1: winner ctx costs `113*16 + 1 + 3` bits (=1808+4=1812b: 113 int16 weights + 1 flag + 3-bit scale id); loser ctx costs `1` bit fallback flag. Gate uses WIRE forward (`mlp_forward_wire`, driver:248-268), not torch (docstring driver:43-44, deviations [D3] driver:48-49).

## L2 per-channel stage — assembly+side vs MED (driver:340-351)
> `340:    asm_q = B21.hbits(P.reshape(-1).astype(np.int32) - pred_vec)`
> `341:    mb = B21.hbits(med_res)`
> `342:    thr_bits = 8 * 64`
> `343:    ch_q_total = asm_q + side + thr_bits`
> `344:    if ch_q_total >= mb or int(win.sum()) == 0:  # L2 channel gate (probe verbatim)`
> `345-348:        return {... win=zeros, scid=255, side_bits=0, has_nets=False, nwin=0, fallback=True}`
> `349-351:    return {... side_bits=side+thr_bits, has_nets=True, nwin=sum(win), fallback=False}`

Side-cost terms L2: `thr_bits=8*64=512b` (8 float64 energy thresholds) + L1 `side` accumulator. Loses (or zero winners) → whole-channel fallback to `med_plane`, `side_bits=0`. Cache reload recomputes identical side (driver:410): `side = nwin*(113*16+1+3) + (9-nwin)*1 + 8*64`.

## Per-channel dual-track exact-byte gate + family/RCT gates (driver:880-905)
> `880:        # dual-track exact assembly: MLP-variant vs no-MLP-variant, min bytes wins`
> `884-886:            bq1,n1 = assemble_channel(..., mlp_side, False, ...); variants.append((len(bq1),bq1,False,0))`
> `887-890:            if has_mlp and any(w[1]=="MLP" for w in plan["wins_all"]): bq2,n2 = assemble_channel(..., True, ...)`
> `891:            variants.sort(key=len); cands.append(min(Q,G) per channel)`
> `898-899:        cands.sort(key=len); _,bb,fam,plan,used_mlp,nmlp = cands[0]  # Q-vs-GRID min bytes`
> `911-920:    encode_image: min-over-{C6,C27,C12} RCT (cands.sort by len)`

Strict `<` for MLP variant (FORMAT:18-19: `Bytes are therefore ≤ CROWN4 per image by construction (unused MLP costs 3 flag bytes)`). No new math claims; see also `mlp_side_bytes` (driver:354-363: 64B thresholds + 9B scale ids + 226B/winning net).
