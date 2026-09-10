/* CROWN2 — exact bit-level codec core (NEW file; hapre.c is READ-ONLY, linked separately).
 *
 * WHAT: streaming decoders + bit packers for the proven CROWN stack:
 *   Q-family: sign-flipped LOCO-365 contexts -> auto-K quantile groups ->
 *             per-group best-of-E16 predictor -> per-group Huff/Golomb/rANS backend.
 *   GRID-family: zero-side energy groups e=|a-c|+|b-c| vs codec-constant grids
 *             (M-THR dictionary, 8 grids, 3b selector), same E16 + 3-way backends.
 *   G-bias: per-group Golomb-only shift adapter t=v-d (Huffman/rANS-invariant).
 *   IG (I-GATED): optional chroma sub-split by |rY| (Y decoded first) via expanded
 *             group indices; off by default (gating decided encoder-side).
 *
 * BORDER RULES (must match probe_b4_b.nbhd EXACTLY; encoder uses recon==orig):
 *   (0,0):        a=b=c=d=Ww=NNe=NE=0
 *   row 0, j>0:   a=P[j-1]; b=c=a; d=a; Ww=a; NNe=a; NE=a
 *   col 0, i>0:   b=P[i-1]; a=c=b; d=P[i-1,1] (or b if W==1); Ww=a; NNe=(i>=2)?P[i-2]:b; NE=d
 *   interior:     a=P[j-1]; b=P[i-1,j]; c=P[i-1,j-1]; d=(j+1<W)?P[i-1,j+1]:b;
 *                 Ww=(j>=2)?P[j-2]:a; NNe=(i>=2)?P[i-2,j]:b; NE=d
 *   In words: zero border; first row copies left; first col copies top. NEVER
 *   edge-replicate. d is top-right (==b at last column, ==a on row 0).
 *
 * CAUSALITY: every context (a,b,c,d,Ww,NNe,NE,key,sign,energy,ybin) is computed
 *   from already-reconstructed causal neighbors only. Encoder decisions may use
 *   the originals because lossless recon==orig makes them identical; the decoder
 *   below re-derives everything from recon. Y channel is decoded fully before
 *   Co/Cg so |rY| splits are causal.
 *
 * ALPHABET: residuals are int16 with PROVEN range |r|<=1024 for 8-bit RGB inputs
 *   (Y in 0..255, Co/Cg in -255..255, worst E16 predictor PLANE a+b-c stays inside
 *   +-1024 on these ranges). Encode aborts loudly (assert/return) outside +-1024;
 *   decode returns -4. NOTHING is ever clipped silently.
 *
 * rANS: M=14 (2^M=16384 >> max alphabet 2049; M=10 precision BUG-CLASS retired).
 *   Reused verbatim from hapre.c (linked, not duplicated).
 *
 * PREDICTOR IDS (E16 order = probe_b5_c.E16):
 *   0 MED 1 TOP 2 LEFT 3 PAETH 4 GRAD 5 GAP80 6 DG 7 AVG_AB 8 PLANE 9 GAP32
 *   10 AC 11 C 12 BC 13 D 14 GAP16 15 AVG3
 *   All divisions are FLOOR division (match numpy // via c2_fdiv); PLANE is exact.
 * BACKEND IDS: 0=Huffman 1=Golomb-Rice 2=rANS.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static inline int c2_fdiv(int a, int b){ int q=a/b, r=a%b; if(r!=0 && ((r<0)!=(b<0))) q--; return q; }

/* E16 expert bank. Must be bit-identical to probe_b5_c.predictors_X. */
int pred_e16(int x, int a,int b,int c,int d,int Ww,int NNe,int NE){
  switch(x){
    case 0: { if(c>=a&&c>=b) return a<b?a:b; if(c<=a&&c<=b) return a>b?a:b; return a+b-c; }
    case 1: return b;
    case 2: return a;
    case 3: { int p=a+b-c,pa=abs(p-a),pb=abs(p-b),pc=abs(p-c);
      if(pa<=pb&&pa<=pc) return a; else if(pb<=pc) return b; else return c; }
    case 4: return c2_fdiv(a+b,2)+c2_fdiv(b-c,4);
    case 5: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>80) return a; else if(gv-gh<-80) return b; else return c2_fdiv(a+b,2)+c2_fdiv(NE-c,4); }
    case 6: return c2_fdiv(c+d,2);
    case 7: return c2_fdiv(a+b,2);
    case 8: return a+b-c;
    case 9: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>32) return a; else if(gv-gh<-32) return b; else return c2_fdiv(a+b,2)+c2_fdiv(NE-c,4); }
    case 10: return c2_fdiv(a+c,2);
    case 11: return c;
    case 12: return c2_fdiv(b+c,2);
    case 13: return d;
    case 14: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>16) return a; else if(gv-gh<-16) return b; else return c2_fdiv(a+b,2)+c2_fdiv(NE-c,4); }
    default: { int s=a+b+c; int q=s/3, r=s%3; if(r!=0 && ((r<0)!=(3<0))) q--; return q; }
  }
}

/* Zero-border neighborhood (probe_b4_b.nbhd mirror; see header comment). */
static inline void c2_nbhd(const int16_t *P,int H,int W,int i,int j,
  int *a,int *b,int *c,int *d,int *Ww,int *NNe,int *NE){
  int A = (j>0)?P[i*W+j-1]:0;
  int B = (i>0)?P[(i-1)*W+j]:0;
  int C = 0;
  if(i==0&&j==0){A=0;B=0;C=0;}
  else if(i==0){B=A;C=A;}
  else if(j==0){A=B;C=B;}
  else{C=P[(i-1)*W+j-1];}
  int D = (j+1>=W)?B:((i==0)?A:P[(i-1)*W+j+1]);
  int WW = (j>=2)?P[i*W+j-2]:A;
  int NNE = (i>=2)?P[(i-2)*W+j]:B;
  *a=A;*b=B;*c=C;*d=D;*Ww=WW;*NNe=NNE;*NE=D;
}

static inline int c2_loco_q1(int g){ int a=g<0?-g:g; if(g==0) return 0; if(a<=2) return g<0?-1:1; if(a<=7) return g<0?-2:2; if(a<=21) return g<0?-3:3; return g<0?-4:4; }

/* Y residual magnitude from Y recon + Y metadata (for IG splits; sign-free). */
static inline int c2_yabs(const int16_t *Y,const uint8_t *ycmap,const uint8_t *ypredid,
  int yfamily,const int *ygrid,int H,int W,int i,int j){
  int a,b,c,d,Ww,NNe,NE; c2_nbhd(Y,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
  int g,x,p;
  if(yfamily==0){ int q1=c2_loco_q1(b-c),q2=c2_loco_q1(c-a),q3=c2_loco_q1(d-b);
    if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){q1=-q1;q2=-q2;q3=-q3;}
    g=ycmap[q1*81+q2*9+q3]; }
  else { int e=abs(a-c)+abs(b-c); g=0; for(int t=0;t<5;t++) if(ygrid[t]<e) g++; }
  x=ypredid[g]; p=pred_e16(x,a,b,c,d,Ww,NNe,NE);
  int r=(int)Y[i*W+j]-p; return r<0?-r:r;
}

/* Golomb-Rice pack: scan order, per-pixel k from kbygrp[keys[n]].
 * Mapping M = v>=0 ? 2v : -2v-1; code = q zeros, one 1, k-bit remainder. */
size_t golomb_pack(const int16_t *syms, const uint8_t *keys, const uint8_t *kbygrp,
                   int N, uint8_t *out, size_t outcap){
  size_t bytepos=0; uint64_t buf=0; int nbits=0;
  for(int n=0;n<N;n++){
    int v=syms[n]; int idx=v+1024; assert(idx>=0&&idx<=2048); if(idx<0||idx>2048) return 0;
    unsigned M = (v>=0)?((unsigned)v*2u):((unsigned)(-v)*2u-1u);
    int k = kbygrp[keys[n]]; assert(k>=0&&k<=12);
    unsigned q = M>>k;
    unsigned rem = (k==0)?0:(M&((1u<<k)-1u));
    while(q>0){ /* emit zero-run in bounded chunks (q can exceed 64; never shift wide) */
      int c = q>24?24:(int)q; q-=c;
      buf<<=c; nbits+=c;
      while(nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0); }
    }
    buf=(buf<<((unsigned)k+1))|((1u<<k)|rem); nbits+=k+1;
    while(nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0); }
  }
  if(nbits){ if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf<<(8-nbits)); }
  return bytepos;
}

/* Unified CROWN2 streaming channel decoder (see header + FORMAT.md).
 * hasn[x]=1 iff expanded group x owns a Huffman table (backend H and non-empty).
 * gsel[g*2+b]: base group g, IG bin b -> expanded index (identity when IG off).
 * dbias[x]: Golomb shift per expanded group (0 when unused).
 * Returns 0 ok; -1 bitstream overrun; -2 tree overflow; -3 alloc; -4 range; -5 bad meta.
 */
int crown2_decode_ch(const uint8_t *hbuf, size_t hlen,
  const uint8_t *gbuf, size_t glen,
  const int16_t *rsyms, const int *goff, int nX,
  const uint32_t *cd, const uint8_t *ln,
  const uint8_t *cmap, const uint8_t *predid, const uint8_t *backend,
  const uint8_t *kvals, const int8_t *dbias, const uint8_t *gsel,
  const uint8_t *hasn, int family, const int *grid_thr,
  const int16_t *yplane, const uint8_t *ycmap, const uint8_t *ypredid, int yfamily, const int *ygrid,
  int16_t *plane, int H, int W){
  if(nX<=0||nX>128) return -5;
  int maxnodes=4100;
  int (*ch)[4100][2]=0; int (*sv)[4100]=0;
  ch=(int(*)[4100][2])calloc((size_t)nX*4100*2,sizeof(int));
  sv=(int(*)[4100])calloc((size_t)nX*4100,sizeof(int));
  if(!ch||!sv){free(ch);free(sv);return -3;}
  for(int t=0;t<nX;t++){ for(int i=0;i<maxnodes;i++){ch[t][i][0]=ch[t][i][1]=-1;sv[t][i]=-9999;}
    if(!hasn[t]) continue;
    int nn=1;
    for(int s2=0;s2<2049;s2++){ size_t u=(size_t)t*2049+s2; uint8_t L=ln[u]; if(!L) continue; uint32_t c=cd[u];
      int node=0;
      for(int b=L-1;b>=0;b--){ int bit=(c>>b)&1;
        if(ch[t][node][bit]==-1){ if(nn>=maxnodes){free(ch);free(sv);return -2;}
          ch[t][node][bit]=nn; ch[t][nn][0]=ch[t][nn][1]=-1; sv[t][nn]=-9999; nn++; }
        node=ch[t][node][bit]; }
      sv[t][node]=s2-1024; } }
  int cur[128]; for(int t=0;t<nX;t++) cur[t]=goff[t];
  size_t hbp=0, gbp=0;
  for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int a,b,c,d,Ww,NNe,NE; c2_nbhd(plane,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
    int g, sg=1;
    if(family==0){ int q1=c2_loco_q1(b-c),q2=c2_loco_q1(c-a),q3=c2_loco_q1(d-b);
      if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){sg=-1;q1=-q1;q2=-q2;q3=-q3;}
      g=cmap[q1*81+q2*9+q3];
    } else { int e=abs(a-c)+abs(b-c); g=0; for(int t=0;t<5;t++) if(grid_thr[t]<e) g++; }
    if(g<0||g>=nX){free(ch);free(sv);return -5;}
    int x;
    if(yplane){ int yb = c2_yabs(yplane,ycmap,ypredid,yfamily,ygrid,H,W,i,j)>1?1:0;
      x=gsel[g*2+yb]; }
    else x=g;
    if(x<0||x>=nX){free(ch);free(sv);return -5;}
    int bb=backend[x];
    int sym=0;
    if(bb==0){ int node=0;
      for(;;){ if((hbp>>3)>=hlen){free(ch);free(sv);return -1;}
        int byte=hbuf[hbp>>3]; int bit=(byte>>(7-(hbp&7)))&1; hbp++; node=ch[x][node][bit];
        if(node<0){free(ch);free(sv);return -1;}
        if(sv[x][node]!=-9999){ sym=sv[x][node]; break; } }
    } else if(bb==1){ unsigned q=0;
      for(;;){ if((gbp>>3)>=glen){free(ch);free(sv);return -1;}
        int byte=gbuf[gbp>>3]; int bit=(byte>>(7-(gbp&7)))&1; gbp++;
        if(bit) break; q++; if(q>5000){free(ch);free(sv);return -1;} }
      int k=kvals[x]; unsigned rem=0;
      for(int t=0;t<k;t++){ if((gbp>>3)>=glen){free(ch);free(sv);return -1;}
        int byte=gbuf[gbp>>3]; int bit=(byte>>(7-(gbp&7)))&1; gbp++; rem=(rem<<1)|bit; }
      unsigned M=(q<<k)|rem;
      sym=((M&1)?-((int)((M+1)>>1)):((int)(M>>1)))+dbias[x];
    } else if(bb==2){ if(cur[x]>=goff[x+1]){free(ch);free(sv);return -1;}
      sym=rsyms[cur[x]++];
    } else {free(ch);free(sv);return -5;}
    if(sym<-1024||sym>1024){free(ch);free(sv);return -4;}
    int p=pred_e16(predid[x],a,b,c,d,Ww,NNe,NE);
    plane[i*W+j]=(int16_t)(p+sg*sym);
  }
  free(ch);free(sv); return 0;
}
