/* CROWN6 — exact bit-level codec core (NEW file; crown4.c / libhapre.so READ-ONLY).
 *
 * WHAT: CROWN4 line + per-context micro-MLP 20th expert (predid 19).
 *   PREDICTOR IDS: E16 order (0 MED ... 15 AVG3, crown2/crown3/crown4-identical) +
 *   16 WAVG + 17 LMS5_T0 + 18 LMS5_T3 (all crown4-verbatim) + 19 MLP.
 *   BACKEND IDS: 0=H 1=G 2=R (unchanged). Magic C6 (framing disambiguation).
 *
 * MLP WIRE FORWARD (float64, FIXED order; bit-exact vs driver mlp_forward_wire):
 *   weights arrive DEQUANTIZED (driver converts int16/2^k; k in 8..12 so the
 *   division is exact in binary FP -- no precision is lost at the handoff).
 *   layout per context: W1[8][12] row-major, b1[8], W2[8], b2 (113 doubles).
 *   pre[j] = b1[j]; for i in 0..11: pre[j] += ((double)x[i]/128.0)*W1[j][i];
 *   h[j] = tanh(pre[j]); y = b2; for j in 0..7: y += W2[j]*h[j];
 *   pred = (int)floor(y*128.0 + 0.5)   [half-up; differs from torch .round()
 *   half-even ONLY on exact .5 values -- quantified in driver xcheck; the
 *   encoder ALWAYS uses this path so decode matches by construction].
 *   Build with -ffp-contract=off (no FMA contraction across the mul/add pairs,
 *   matching numpy's separate elementwise ufuncs bit-for-bit -- proven by the
 *   exhaustive C-vs-wire xcheck, 0 mismatches required before integration).
 *
 * MLP FEATURES (ZERO-border causal taps -- NOT probe_b21's edge-replication,
 *   which peeks at undecoded pixels on row 0 / col 0 and broke streaming
 *   decode at pixel (0,0); interior pixels are identical to the probe):
 *   x = [L,T,TL,TR,Ww,NNe,(L+T)//2,|L-T|,L3,T3,TL2,TR2] with
 *   tap(di,dj) = plane[i+di][j+dj] if inside, else 0
 *   (exactly numpy pad(mode="constant") slicing):
 *   L=(0,-1) T=(-1,0) TL=(-1,-1) TR=(-1,+1) Ww=(0,-2) NNe=(-2,0)
 *   L3=(0,-3) T3=(-3,0) TL2=(-2,-2) TR2=(-2,+2).
 *   (L+T)//2 uses floor division (c6_fdiv -- matches numpy // on negatives).
 *   Energy E = |L-T|+|L-TL|+|T-TL| (edge L/T/TL ints); ctx = #{thr[t] < E}
 *   over the 8 float64 thresholds (== numpy searchsorted side='left').
 *   Non-winning/empty ctx (mlp_win[ctx]==0) falls back to edge-MED(L,T,TL).
 *   All taps are causal (rows < i fully decoded; row i only cols < j), so the
 *   streaming raster decoder reproduces encoder features exactly.
 *
 * LMS5 / WAVG / E16 / rANS / Golomb-polarity / RCT-bank: crown4-verbatim
 * (see CROWN4_FORMAT.md + crown4.c header). ALPHABET |r| <= 1024 (decoder -4).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "crown6_tanhlut.h"

static inline int c6_fdiv(int a, int b){ int q=a/b, r=a%b; if(r!=0 && ((r<0)!=(b<0))) q--; return q; }

/* E16 expert bank. Bit-identical to probe_b5_c.predictors_X / crown4 pred_e16. */
int pred_e16_c6(int x, int a,int b,int c,int d,int Ww,int NNe,int NE){
  switch(x){
    case 0: { if(c>=a&&c>=b) return a<b?a:b; if(c<=a&&c<=b) return a>b?a:b; return a+b-c; }
    case 1: return b;
    case 2: return a;
    case 3: { int p=a+b-c,pa=abs(p-a),pb=abs(p-b),pc=abs(p-c);
      if(pa<=pb&&pa<=pc) return a; else if(pb<=pc) return b; else return c; }
    case 4: return c6_fdiv(a+b,2)+c6_fdiv(b-c,4);
    case 5: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>80) return a; else if(gv-gh<-80) return b; else return c6_fdiv(a+b,2)+c6_fdiv(NE-c,4); }
    case 6: return c6_fdiv(c+d,2);
    case 7: return c6_fdiv(a+b,2);
    case 8: return a+b-c;
    case 9: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>32) return a; else if(gv-gh<-32) return b; else return c6_fdiv(a+b,2)+c6_fdiv(NE-c,4); }
    case 10: return c6_fdiv(a+c,2);
    case 11: return c;
    case 12: return c6_fdiv(b+c,2);
    case 13: return d;
    case 14: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>16) return a; else if(gv-gh<-16) return b; else return c6_fdiv(a+b,2)+c6_fdiv(NE-c,4); }
    default: { int s=a+b+c; int q=s/3, r=s%3; if(r!=0 && ((r<0)!=(3<0))) q--; return q; }
  }
}

/* Zero-border E16 neighborhood (crown4 c4_nbhd-verbatim; b4_b doctrine). */
static inline void c6_nbhd(const int16_t *P,int H,int W,int i,int j,
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

static inline int c6_loco_q1(int g){ int a=g<0?-g:g; if(g==0) return 0; if(a<=2) return g<0?-1:1; if(a<=7) return g<0?-2:2; if(a<=21) return g<0?-3:3; return g<0?-4:4; }

static inline int c6_wpred(const int8_t *w, int L, int T, int TL, int TR){
  int num = (int)w[0]*L + (int)w[1]*T + (int)w[2]*TL + (int)w[3]*TR;
  return c6_fdiv(num+8, 16);
}

static inline int c6_med_b17(int a, int b, int tlb){
  int mn = a<b?a:b, mx = a<b?b:a;
  if(tlb >= mx) return mn;
  if(tlb <= mn) return mx;
  return a+b-tlb;
}

static inline int c6_lms_pred(const int *w, int L,int T,int TL,int TR,int MED){
  int dot = w[0]*L + w[1]*T + w[2]*TL + w[3]*TR + w[4]*MED;
  return c6_fdiv(dot+8, 16);
}

static inline void c6_lms_update(int *w, int L,int T,int TL,int TR,int MED,int y_true,int thr){
  int xs[5] = {L,T,TL,TR,MED};
  int pr = c6_lms_pred(w,L,T,TL,TR,MED);
  int e = y_true - pr;
  int ae = e<0?-e:e;
  if(ae > thr){
    int se = e>0?1:-1;
    int m = c6_fdiv(L+T+TL+TR+MED+2, 5);
    for(int k=0;k<5;k++){ int di=xs[k]-m; if(di>0) w[k]+=se; else if(di<0) w[k]-=se; }
    for(int k=0;k<5;k++){ if(w[k]<-8) w[k]=-8; else if(w[k]>20) w[k]=20; }
    int s=w[0]+w[1]+w[2]+w[3]+w[4];
    while(s>16){ int bi=0; for(int k=1;k<5;k++) if(w[k]>w[bi]) bi=k; w[bi]--; s--; }
    while(s<16){ int bi=0; for(int k=1;k<5;k++) if(w[k]<w[bi]) bi=k; w[bi]++; s++; }
  }
}

/* MLP wire forward: fixed order (b1 then i=0..11 adds per hidden unit).
 * tanh via FROZEN LUT (crown6_tanhlut.h): rr=|pre|*16384 (exact); rr>131072 ->
 * +/-1.0; else idx=(int)floor(rr+0.5), h=sign*C6_TANH[idx]. Bit-exact vs the
 * numpy mirror (identical constants + identical op order); resolves the 1-ulp
 * numpy-vectorized-tanh vs libm-tanh divergence found in xcheck. */
static inline int c6_mlp_fwd(const int *x, const double *W1, const double *b1,
                             const double *W2, double b2){
  double pre[8], h[8];
  for(int j=0;j<8;j++){ double acc=b1[j]; for(int i=0;i<12;i++) acc += ((double)x[i]/128.0)*W1[j*12+i]; pre[j]=acc; }
  for(int j=0;j<8;j++){ double pr=pre[j], rr=(pr<0.0?-pr:pr)*16384.0, hh;
    if(rr>131072.0) hh=pr<0.0?-1.0:1.0;
    else { int idx=(int)floor(rr+0.5); hh=(pr<0.0?-1.0:1.0)*C6_TANH[idx]; }
    h[j]=hh; }
  double y=b2; for(int j=0;j<8;j++) y += W2[j]*h[j];
  double out=y*128.0;
  return (int)floor(out+0.5);
}

/* Exported batch entry for the mandatory C-vs-wire xcheck (ctypes). */
void crown6_mlp_predict_batch(const int32_t *xs, int N,
                              const double *W1, const double *b1,
                              const double *W2, double b2, int32_t *out){
  for(int n=0;n<N;n++) out[n]=(int32_t)c6_mlp_fwd(xs+n*12, W1, b1, W2, b2);
}

/* Zero-border causal tap: plane[i+di][j+dj] or 0 outside. (probe_b21 used
 * edge-replication here, which peeks at UNDECODED pixels on row 0 / col 0 --
 * causality violation, first divergence at (0,0); zero taps are causal.) */
static inline int c6_edge(const int16_t *P,int H,int W,int i,int j,int di,int dj){
  int r=i+di, c=j+dj;
  if(r<0||r>=H||c<0||c>=W) return 0;
  return P[r*W+c];
}

static const int C6_PERMS[6][3] = {{0,1,2},{1,2,0},{2,0,1},{0,2,1},{1,0,2},{2,1,0}};
static const int C6_BANK_PERM[3] = {0, 3, 1};
static const int C6_BANK_T[3] = {6, 6, 5};

int crown6_rct_inv(const int16_t *p0, const int16_t *p1, const int16_t *p2,
  int H, int W, int rct, uint8_t *rgb){
  if(rct<0||rct>2) return -1;
  int perm = C6_BANK_PERM[rct], t = C6_BANK_T[rct];
  int bad = 0;
  for(int n=0;n<H*W;n++){
    int X=p0[n], Y=p1[n], Z=p2[n], A,B,C;
    if(t==6){ int tt=X-c6_fdiv(Z,2); int G=Z+tt; int Bc=tt-c6_fdiv(Y,2);
      A=Y+Bc; B=G; C=Bc; }
    else { A=X; B=Y+X+c6_fdiv(Z,2); C=Z+X; }
    int v[3]; v[0]=A; v[1]=B; v[2]=C;
    for(int k=0;k<3;k++){ int vv=v[k]; if(vv<0||vv>255) bad++; rgb[3*n+C6_PERMS[perm][k]]=(uint8_t)(vv&0xFF); }
  }
  return bad;
}

size_t crown6_golomb_pack(const int16_t *syms, const uint8_t *keys, const uint8_t *kbygrp,
                   int N, uint8_t *out, size_t outcap){
  size_t bytepos=0; uint64_t buf=0; int nbits=0;
  for(int n=0;n<N;n++){
    int v=syms[n]; int idx=v+1024; assert(idx>=0&&idx<=2048); if(idx<0||idx>2048) return 0;
    unsigned M = (v>=0)?((unsigned)v*2u):((unsigned)(-v)*2u-1u);
    int k = kbygrp[keys[n]]; assert(k>=0&&k<=12);
    unsigned q = M>>k;
    unsigned rem = (k==0)?0:(M&((1u<<k)-1u));
    while(q>0){
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

/* Unified CROWN6 streaming channel decoder (crown4 core + MLP expert id 19).
 * mlp_present: 1 iff MLP side info was transmitted for this channel.
 * mlp_thr[8]: float64 energy thresholds; mlp_win[9]: winning-ctx flags;
 * mlp_w[9*113]: dequantized doubles per ctx (W1 96 row-major, b1 8, W2 8, b2).
 * Returns 0 ok; -1 bitstream overrun; -2 tree overflow; -3 alloc; -4 range; -5 bad meta.
 */
int crown6_decode_ch(const uint8_t *hbuf, size_t hlen,
  const uint8_t *gbuf, size_t glen,
  const int16_t *rsyms, const int *goff, int nX,
  const uint32_t *cd, const uint8_t *ln,
  const uint8_t *cmap, const uint8_t *predid, const uint8_t *backend,
  const uint8_t *kvals, const int8_t *dbias, const uint8_t *hasn,
  const uint8_t *wuse, const int8_t *ww, int nbw,
  int family, const int *grid_thr,
  int16_t *plane, int H, int W,
  int mlp_present, const double *mlp_thr, const uint8_t *mlp_win, const double *mlp_w){
  if(nX<=0||nX>128) return -5;
  for(int t=0;t<nX;t++){ if(predid[t]>19||backend[t]>2) return -5; if(predid[t]==19&&!mlp_present) return -5; }
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
  int w0[5]={3,3,3,3,4}, w3[5]={3,3,3,3,4};
  size_t hbp=0, gbp=0;
  for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int a,b,c,d,Ww,NNe,NE; c6_nbhd(plane,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
    int g, sg=1;
    if(family==0){ int q1=c6_loco_q1(b-c),q2=c6_loco_q1(c-a),q3=c6_loco_q1(d-b);
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
    int TL0=(i>0&&j>0)?plane[(i-1)*W+j-1]:0;
    int med5=c6_med_b17(a,b,TL0);
    int pid=predid[x], p;
    if(pid==16){
      int blk=(i>>5)*nbw+(j>>5);
      if(wuse[blk]){
        int L=(j>0)?plane[i*W+j-1]:((i>0)?plane[(i-1)*W]:0);
        int T=(i>0)?plane[(i-1)*W+j]:((j>0)?plane[i*W+j-1]:0);
        int TL=(i>0&&j>0)?plane[(i-1)*W+j-1]:0;
        int TR=(i>0&&j+1<W)?plane[(i-1)*W+j+1]:0;
        p=c6_wpred(ww+4*blk,L,T,TL,TR);
      } else p=pred_e16_c6(0,a,b,c,d,Ww,NNe,NE);
    } else if(pid==17){
      p=c6_lms_pred(w0,a,b,c,d,med5);
    } else if(pid==18){
      p=c6_lms_pred(w3,a,b,c,d,med5);
    } else if(pid==19){
      int eL=c6_edge(plane,H,W,i,j,0,-1), eT=c6_edge(plane,H,W,i,j,-1,0),
          eTL=c6_edge(plane,H,W,i,j,-1,-1);
      int eE=abs(eL-eT)+abs(eL-eTL)+abs(eT-eTL);
      int ectx=0; for(int t=0;t<8;t++) if(mlp_thr[t]<(double)eE) ectx++;
      if(mlp_win[ectx]){
        int fx[12];
        fx[0]=eL; fx[1]=eT; fx[2]=eTL;
        fx[3]=c6_edge(plane,H,W,i,j,-1,1);  fx[4]=c6_edge(plane,H,W,i,j,0,-2);
        fx[5]=c6_edge(plane,H,W,i,j,-2,0);  fx[6]=c6_fdiv(fx[0]+fx[1],2);
        fx[7]=abs(fx[0]-fx[1]);             fx[8]=c6_edge(plane,H,W,i,j,0,-3);
        fx[9]=c6_edge(plane,H,W,i,j,-3,0);  fx[10]=c6_edge(plane,H,W,i,j,-2,-2);
        fx[11]=c6_edge(plane,H,W,i,j,-2,2);
        const double *mw = mlp_w + (size_t)ectx*113;
        p=c6_mlp_fwd(fx, mw, mw+96, mw+104, mw[112]);
      } else {
        int mn = eL<eT?eL:eT, mx = eL<eT?eT:eL;
        if(eTL >= mx) p = mn; else if(eTL <= mn) p = mx; else p = eL+eT-eTL;
      }
    } else p=pred_e16_c6(pid,a,b,c,d,Ww,NNe,NE);
    int yv = p+sg*sym;
    plane[i*W+j]=(int16_t)yv;
    c6_lms_update(w0,a,b,c,d,med5,yv,0);
    c6_lms_update(w3,a,b,c,d,med5,yv,3);
  }
  free(ch);free(sv); return 0;
}