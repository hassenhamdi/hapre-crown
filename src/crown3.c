/* CROWN3 — exact bit-level codec core (NEW file; hapre.c / libhapre.so READ-ONLY).
 *
 * WHAT: CROWN2 line + Attack-1 winners (probe_b17):
 *   (i)  global RCT selector over {C6,C27,C12} (2 bits/image, homogeneous planes);
 *   (ii) Weighted-form 4-tap predictor as 17th expert (predid 16) on fixed G32
 *        spatial groups: per-block LS weights x16-quantized to [-16,15],
 *        Laplacian-MDL gated (decisions encoder-side, transmitted, never inferred).
 * Per-block RCT explicitly DEAD (probe M2/M4) — NOT built.
 *
 * BORDER RULES (must match probe_b17.nbhd4 EXACTLY for the Weighted stencil;
 *   E16 neighborhoods must match probe_b4_b.nbhd via c2_nbhd, as in crown2.c):
 *   L  = (j>0) ? P[i,j-1] : ((i>0) ? P[i-1,0] : 0)
 *   T  = (i>0) ? P[i-1,j] : ((j>0) ? P[i,j-1] : 0)
 *   TL = (i>0 && j>0) ? P[i-1,j-1] : 0
 *   TR = (i>0 && j+1<W) ? P[i-1,j+1] : 0
 * In words: zero border; row0 copies left (for L and T); col0 copies top (for L).
 * NEVER edge-replicate. Encoder may use originals (lossless recon==orig makes them
 * identical); decoder re-derives everything from recon (causal).
 *
 * WEIGHTED EVAL (bit-exact vs probe_b17.wpred_from_w / prove_decode):
 *   num = w0*L + w1*T + w2*TL + w3*TR;  pred = floor((num+8)/16).
 * Weights are int8 in [-16,15]; floor division matches numpy // via c3_fdiv.
 * Gated-off blocks (wuse=0) fall back to MED (pred_e16 id 0) — encoder defines
 * the WAVG residual column identically, so C/E16 search ties keep MED.
 *
 * RCT (bit-exact vs probe_b17.rct_fwd/rct_inv for the 3 carried entries):
 *   id 0 = C6  = (perm0,t6)  identity-perm YCoCg-R
 *   id 1 = C27 = (perm3,t6)  RBG-ordered YCoCg-R
 *   id 2 = C12 = (perm1,t5)  subtract-plus-average mode
 * Forward is encoder-side (Python, B17 verbatim); C carries only the inverse.
 *
 * ALPHABET: |r| <= 1024 everywhere; encoder asserts loudly, decoder returns -4.
 * Never clipped. rANS: M=14 via libhapre.so (Python-side, pre-decoded to rsyms).
 * PREDICTOR IDS: E16 order (0 MED ... 15 AVG3, same as crown2.c) + 16 WAVG.
 * BACKEND IDS: 0=Huffman 1=Golomb-Rice 2=rANS.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static inline int c3_fdiv(int a, int b){ int q=a/b, r=a%b; if(r!=0 && ((r<0)!=(b<0))) q--; return q; }

/* E16 expert bank. Bit-identical to probe_b5_c.predictors_X / crown2.c pred_e16. */
int pred_e16(int x, int a,int b,int c,int d,int Ww,int NNe,int NE){
  switch(x){
    case 0: { if(c>=a&&c>=b) return a<b?a:b; if(c<=a&&c<=b) return a>b?a:b; return a+b-c; }
    case 1: return b;
    case 2: return a;
    case 3: { int p=a+b-c,pa=abs(p-a),pb=abs(p-b),pc=abs(p-c);
      if(pa<=pb&&pa<=pc) return a; else if(pb<=pc) return b; else return c; }
    case 4: return c3_fdiv(a+b,2)+c3_fdiv(b-c,4);
    case 5: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>80) return a; else if(gv-gh<-80) return b; else return c3_fdiv(a+b,2)+c3_fdiv(NE-c,4); }
    case 6: return c3_fdiv(c+d,2);
    case 7: return c3_fdiv(a+b,2);
    case 8: return a+b-c;
    case 9: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>32) return a; else if(gv-gh<-32) return b; else return c3_fdiv(a+b,2)+c3_fdiv(NE-c,4); }
    case 10: return c3_fdiv(a+c,2);
    case 11: return c;
    case 12: return c3_fdiv(b+c,2);
    case 13: return d;
    case 14: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>16) return a; else if(gv-gh<-16) return b; else return c3_fdiv(a+b,2)+c3_fdiv(NE-c,4); }
    default: { int s=a+b+c; int q=s/3, r=s%3; if(r!=0 && ((r<0)!=(3<0))) q--; return q; }
  }
}

/* Zero-border E16 neighborhood (crown2.c c2_nbhd mirror; see header comment). */
static inline void c3_nbhd(const int16_t *P,int H,int W,int i,int j,
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

static inline int c3_loco_q1(int g){ int a=g<0?-g:g; if(g==0) return 0; if(a<=2) return g<0?-1:1; if(a<=7) return g<0?-2:2; if(a<=21) return g<0?-3:3; return g<0?-4:4; }

/* Weighted 4-tap eval on the probe_b17 stencil (recon-derived L,T,TL,TR). */
static inline int c3_wpred(const int8_t *w, int L, int T, int TL, int TR){
  int num = (int)w[0]*L + (int)w[1]*T + (int)w[2]*TL + (int)w[3]*TR;
  return c3_fdiv(num+8, 16);
}

/* JXL-style RCT catalogue (probe_b17 verbatim): perm + type. */
static const int C3_PERMS[6][3] = {{0,1,2},{1,2,0},{2,0,1},{0,2,1},{1,0,2},{2,1,0}};
/* Carried bank: id -> (perm, t). 0=C6(0,6) 1=C27(3,6) 2=C12(1,5). */
static const int C3_BANK_PERM[3] = {0, 3, 1};
static const int C3_BANK_T[3] = {6, 6, 5};

/* Inverse RCT: planes (X,Y,Z) -> interleaved RGB. Returns #out-of-range (0 ok),
 * -1 on bad rct id. Exact port of probe_b17.rct_inv for the carried bank. */
int crown3_rct_inv(const int16_t *p0, const int16_t *p1, const int16_t *p2,
  int H, int W, int rct, uint8_t *rgb){
  if(rct<0||rct>2) return -1;
  int perm = C3_BANK_PERM[rct], t = C3_BANK_T[rct];
  int bad = 0;
  for(int n=0;n<H*W;n++){
    int X=p0[n], Y=p1[n], Z=p2[n], A,B,C;
    if(t==6){ int tt=X-c3_fdiv(Z,2); int G=Z+tt; int Bc=tt-c3_fdiv(Y,2);
      A=Y+Bc; B=G; C=Bc; }
    else { A=X; B=Y+X+c3_fdiv(Z,2); C=Z+X; }
    int v[3]; v[0]=A; v[1]=B; v[2]=C;
    for(int k=0;k<3;k++){ int vv=v[k]; if(vv<0||vv>255) bad++; rgb[3*n+C3_PERMS[perm][k]]=(uint8_t)(vv&0xFF); }
  }
  return bad;
}

/* Golomb-Rice pack: scan order, per-pixel k from kbygrp[keys[n]].
 * Mapping M = v>=0 ? 2v : -2v-1; code = q zeros, one 1, k-bit remainder.
 * Vendored verbatim from crown2.c (libhapre.so's era-variant uses q ones+zero
 * and is NOT wire-compatible — verified divergent 2026-09-07). */
size_t crown3_golomb_pack(const int16_t *syms, const uint8_t *keys, const uint8_t *kbygrp,
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

/* Unified CROWN3 streaming channel decoder.
 * predid[x]==16 selects the Weighted expert: spatial G32 block b=(i>>5)*nbw+(j>>5);
 *   wuse[b]? c3_wpred(ww+4b, L,T,TL,TR from recon) : MED.
 * hasn[x]=1 iff group x owns a Huffman table (backend H and non-empty).
 * Returns 0 ok; -1 bitstream overrun; -2 tree overflow; -3 alloc; -4 range; -5 bad meta.
 */
int crown3_decode_ch(const uint8_t *hbuf, size_t hlen,
  const uint8_t *gbuf, size_t glen,
  const int16_t *rsyms, const int *goff, int nX,
  const uint32_t *cd, const uint8_t *ln,
  const uint8_t *cmap, const uint8_t *predid, const uint8_t *backend,
  const uint8_t *kvals, const int8_t *dbias, const uint8_t *hasn,
  const uint8_t *wuse, const int8_t *ww, int nbw,
  int family, const int *grid_thr,
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
    int a,b,c,d,Ww,NNe,NE; c3_nbhd(plane,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
    int g, sg=1;
    if(family==0){ int q1=c3_loco_q1(b-c),q2=c3_loco_q1(c-a),q3=c3_loco_q1(d-b);
      if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){sg=-1;q1=-q1;q2=-q2;q3=-q3;}
      g=cmap[q1*81+q2*9+q3];
    } else { int e=abs(a-c)+abs(b-c); g=0; for(int t=0;t<5;t++) if(grid_thr[t]<e) g++; }
    if(g<0||g>=nX){free(ch);free(sv);return -5;}
    int x=g;
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
    int pid=predid[x], p;
    if(pid==16){
      int blk=(i>>5)*nbw+(j>>5);
      if(wuse[blk]){
        int L=(j>0)?plane[i*W+j-1]:((i>0)?plane[(i-1)*W]:0);
        int T=(i>0)?plane[(i-1)*W+j]:((j>0)?plane[i*W+j-1]:0);
        int TL=(i>0&&j>0)?plane[(i-1)*W+j-1]:0;
        int TR=(i>0&&j+1<W)?plane[(i-1)*W+j+1]:0;
        p=c3_wpred(ww+4*blk,L,T,TL,TR);
      } else p=pred_e16(0,a,b,c,d,Ww,NNe,NE);
    } else p=pred_e16(pid,a,b,c,d,Ww,NNe,NE);
    plane[i*W+j]=(int16_t)(p+sg*sym);
  }
  free(ch);free(sv); return 0;
}
