/* CROWN5 — CROWN4 predictors/groups + GLOBAL raster-order adaptive entropy states.
 *
 * WHAT: Greenfield exact codec (NEW file; crown4.c / hapre.c / libhapre.so READ-ONLY,
 *   never modified). Predictor/group front end is CROWN4-verbatim (E19 Q-family with
 *   sign-flipped LOCO-365 + WIDE-K quantile, E17 GRID M-THR energy, WAVG G32,
 *   LMS5_T0/T3 recon-only states, RCT{C6,C27,C12}, dual-neighborhood borders).
 *   Entropy back end REPLACES per-group static tables (H tables / G k+dbias /
 *   rANS tables) with GLOBAL raster-order adaptive states shared across groups:
 *     acodec 0 = A-act: 16-ctx (4 activity x 4 prefix-pos) backward-adaptive binary
 *       range coder (16-bit E1/E2/E3, probe_b19_adaptive.py BinEnc/BinDec-verbatim)
 *       on Exp-Golomb binarization of M=2|r|-(r>0), prefix bins coded, suffix raw.
 *     acodec 1 = C-gol: per-activity-ctx (4 ctx) adaptive Golomb-k (JPEG-LS N/A
 *       counters, RESET=64, init N=1/A=4, raster order, probe GolombState-verbatim),
 *       code q ZEROS + one 1 + k-bit remainder MSB-first (CROWN polarity).
 *   Adaptation state is a pure function of already-decoded symbols/pixels (+ static
 *   schedules known to both sides). No transmitted init (B-fwd killed). Zero tables.
 *
 * PORT FIDELITY (probe_b19_adaptive.py):
 *   ATHR=(4,12,48) -> ab 0..3; NPOS=4; nctx=16 (A-act) / 4 (C-gol); MAXT=128 halving
 *     c=(c+1)//2 floor 1; RESET=64 halving N>>=1,A>>=1 floor N>=1; uniform init
 *     c0=c1=1, N=1/A=4. M mapping + Exp-Golomb L=(M+1).bit_length() + rem + raw.
 *   Range coder: rng=high-low+1; split=(rng*c0)//tot clamp 1..rng-1;
 *     finish: follow+=1, out(0) if low<0x4000 else out(1); decoder _read past EOF=0.
 *   Activity e=|lv-tl|+|tv-tl| with B17-style borders (lv=left else above-col0 else 0;
 *     tv=top else left else 0; tl=diag else 0) from causal recon only.
 *
 * BORDER DOCTRINE (CROWN4-verbatim, encoder==decoder):
 *   E16 nbhd/keys/energy via c5_nbhd (b4_b: zero border; row0 copies left B=A,C=A;
 *     col0 copies top A=B,C=B; TR copies L on row0, copies T in last col; Ww copies L
 *     j<2; NNe copies T i<2; NE==TR). WAVG stencil + LMS MED5 + activity use B17-style
 *     (TL0/TR0 zero at borders; lv/tv/tl rule above). L/T shared identical.
 *   LMS stencil xs=[L,T,TL,TR(b4),MED5(B17)], pred=floor((w.xs+8)/16) via c5_fdiv.
 *
 * ALPHABET: |sym|<=1024 loud (-4, never clip). META: predid<=18, acodec<=1 (-5).
 * FRAMING: exact byte counts asserted by driver (p==len(blob)); overrun -> -1.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

static inline int c5_fdiv(int a, int b){ int q=a/b, r=a%b; if(r!=0 && ((r<0)!=(b<0))) q--; return q; }

/* E16 bank, bit-identical to probe_b5_c / crown4 pred_e16_c4. */
int pred_e16_c5(int x, int a,int b,int c,int d,int Ww,int NNe,int NE){
  switch(x){
    case 0: { if(c>=a&&c>=b) return a<b?a:b; if(c<=a&&c<=b) return a>b?a:b; return a+b-c; }
    case 1: return b;
    case 2: return a;
    case 3: { int p=a+b-c,pa=abs(p-a),pb=abs(p-b),pc=abs(p-c);
      if(pa<=pb&&pa<=pc) return a; else if(pb<=pc) return b; else return c; }
    case 4: return c5_fdiv(a+b,2)+c5_fdiv(b-c,4);
    case 5: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>80) return a; else if(gv-gh<-80) return b; else return c5_fdiv(a+b,2)+c5_fdiv(NE-c,4); }
    case 6: return c5_fdiv(c+d,2);
    case 7: return c5_fdiv(a+b,2);
    case 8: return a+b-c;
    case 9: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>32) return a; else if(gv-gh<-32) return b; else return c5_fdiv(a+b,2)+c5_fdiv(NE-c,4); }
    case 10: return c5_fdiv(a+c,2);
    case 11: return c;
    case 12: return c5_fdiv(b+c,2);
    case 13: return d;
    case 14: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>16) return a; else if(gv-gh<-16) return b; else return c5_fdiv(a+b,2)+c5_fdiv(NE-c,4); }
    default: { int s=a+b+c; int q=s/3, r=s%3; if(r!=0 && ((r<0)!=(3<0))) q--; return q; }
  }
}

static inline void c5_nbhd(const int16_t *P,int H,int W,int i,int j,
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

static inline int c5_loco_q1(int g){ int a=g<0?-g:g; if(g==0) return 0; if(a<=2) return g<0?-1:1; if(a<=7) return g<0?-2:2; if(a<=21) return g<0?-3:3; return g<0?-4:4; }

static inline int c5_wpred(const int8_t *w, int L, int T, int TL, int TR){
  int num = (int)w[0]*L + (int)w[1]*T + (int)w[2]*TL + (int)w[3]*TR;
  return c5_fdiv(num+8, 16);
}
static inline int c5_med_b17(int a, int b, int tlb){
  int mn = a<b?a:b, mx = a<b?b:a;
  if(tlb >= mx) return mn;
  if(tlb <= mn) return mx;
  return a+b-tlb;
}
static inline int c5_lms_pred(const int *w, int L,int T,int TL,int TR,int MED){
  int dot = w[0]*L + w[1]*T + w[2]*TL + w[3]*TR + w[4]*MED;
  return c5_fdiv(dot+8, 16);
}
static inline void c5_lms_update(int *w, int L,int T,int TL,int TR,int MED,int y_true,int thr){
  int xs[5] = {L,T,TL,TR,MED};
  int pr = c5_lms_pred(w,L,T,TL,TR,MED);
  int e = y_true - pr;
  int ae = e<0?-e:e;
  if(ae > thr){
    int se = e>0?1:-1;
    int m = c5_fdiv(L+T+TL+TR+MED+2, 5);
    for(int k=0;k<5;k++){ int di=xs[k]-m; if(di>0) w[k]+=se; else if(di<0) w[k]-=se; }
    for(int k=0;k<5;k++){ if(w[k]<-8) w[k]=-8; else if(w[k]>20) w[k]=20; }
    int s=w[0]+w[1]+w[2]+w[3]+w[4];
    while(s>16){ int bi=0; for(int k=1;k<5;k++) if(w[k]>w[bi]) bi=k; w[bi]--; s--; }
    while(s<16){ int bi=0; for(int k=1;k<5;k++) if(w[k]<w[bi]) bi=k; w[bi]++; s++; }
  }
}

static const int C5_PERMS[6][3] = {{0,1,2},{1,2,0},{2,0,1},{0,2,1},{1,0,2},{2,1,0}};
static const int C5_BANK_PERM[3] = {0, 3, 1};
static const int C5_BANK_T[3] = {6, 6, 5};

int crown5_rct_inv(const int16_t *p0, const int16_t *p1, const int16_t *p2,
  int H, int W, int rct, uint8_t *rgb){
  if(rct<0||rct>2) return -1;
  int perm = C5_BANK_PERM[rct], t = C5_BANK_T[rct];
  int bad = 0;
  for(int n=0;n<H*W;n++){
    int X=p0[n], Y=p1[n], Z=p2[n], A,B,C;
    if(t==6){ int tt=X-c5_fdiv(Z,2); int G=Z+tt; int Bc=tt-c5_fdiv(Y,2);
      A=Y+Bc; B=G; C=Bc; }
    else { A=X; B=Y+X+c5_fdiv(Z,2); C=Z+X; }
    int v[3]; v[0]=A; v[1]=B; v[2]=C;
    for(int k=0;k<3;k++){ int vv=v[k]; if(vv<0||vv>255) bad++; rgb[3*n+C5_PERMS[perm][k]]=(uint8_t)(vv&0xFF); }
  }
  return bad;
}

/* ---- adaptive state helpers (probe-verbatim constants) ---- */
static inline int c5_act_bin(int e){
  if(e<=4) return 0; if(e<=12) return 1; if(e<=48) return 2; return 3;
}
static inline int c5_M_to_r(int M){
  if(M==0) return 0;
  if(M&1) return (M+1)>>1;
  return -(M>>1);
}

/* Binary range decoder state (16 ctx). */
typedef struct { const uint8_t *buf; size_t len; size_t bitpos;
  unsigned low, high, code; int c0[16], c1[16]; } C5BinDec;

static inline int c5_bd_read(C5BinDec *d){
  if((d->bitpos>>3) < d->len){
    int byte = d->buf[d->bitpos>>3];
    int bit = (byte >> (7-(d->bitpos&7))) & 1;
    d->bitpos++; return bit;
  }
  return 0;
}
/* returns bit or -1 on structural runaway (caller maps to -1 overrun / -6) */
static int c5_bd_dec(C5BinDec *d, int ctx){
  int c0 = d->c0[ctx], c1 = d->c1[ctx];
  int tot = c0 + c1;
  unsigned rng = d->high - d->low + 1u;
  unsigned split = (rng * (unsigned)c0) / (unsigned)tot;
  if(split==0) split=1; else if(split>=rng) split=rng-1;
  int bit;
  if(d->code < d->low + split){ bit=0; d->high = d->low + split - 1; c0++; }
  else { bit=1; d->low = d->low + split; c1++; }
  if(c0+c1 >= 128){
    c0 = (c0+1)/2; c1 = (c1+1)/2;
    if(c0<1) c0=1; if(c1<1) c1=1;
  }
  d->c0[ctx]=c0; d->c1[ctx]=c1;
  for(;;){
    if(d->high < 0x8000u){ d->low*=2; d->high=d->high*2+1; d->code=d->code*2+(unsigned)c5_bd_read(d); }
    else if(d->low >= 0x8000u){ d->low=(d->low-0x8000u)*2; d->high=(d->high-0x8000u)*2+1; d->code=(d->code-0x8000u)*2+(unsigned)c5_bd_read(d); }
    else if(d->low >= 0x4000u && d->high < 0xC000u){ d->low=(d->low-0x4000u)*2; d->high=(d->high-0x4000u)*2+1; d->code=(d->code-0x4000u)*2+(unsigned)c5_bd_read(d); }
    else break;
    d->low&=0xFFFFu; d->high&=0xFFFFu; d->code&=0xFFFFu;
  }
  return bit;
}

/* Unified CROWN5 streaming channel decoder.
 * acodec 0: arith/raw pair (A-act 16-ctx); acodec 1: gbuf Golomb (C-gol 4-ctx).
 * cmap[729] (Q) or grid_thr[5] (GRID); predid[ng] 0..18; wuse/ww G32; nbw.
 * Returns 0 ok; -1 overrun; -2 tree n/a (reserved); -3 alloc n/a; -4 range; -5 bad meta; -6 prefix runaway.
 */
int crown5_decode_ch(const uint8_t *arith, size_t alen,
  const uint8_t *raw, size_t rlen,
  const uint8_t *gbuf, size_t glen,
  const uint8_t *cmap, const uint8_t *predid,
  const uint8_t *wuse, const int8_t *ww, int nbw,
  int family, const int *grid_thr, int acodec,
  int16_t *plane, int H, int W, int nX){
  if(nX<=0||nX>128) return -5;
  for(int t=0;t<nX;t++) if(predid[t]>18) return -5;
  if(acodec!=0&&acodec!=1) return -5;
  if(family!=0&&family!=1) return -5;
  int w0[5]={3,3,3,3,4}, w3[5]={3,3,3,3,4};
  if(acodec==0){
    C5BinDec bd; bd.buf=arith; bd.len=alen; bd.bitpos=0; bd.low=0; bd.high=0xFFFF; bd.code=0;
    for(int k=0;k<16;k++){ bd.c0[k]=1; bd.c1[k]=1; }
    for(int k=0;k<16;k++) bd.code=(bd.code<<1)|(unsigned)c5_bd_read(&bd);
    size_t rawpos=0;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
      int a,b,c,d,Ww,NNe,NE; c5_nbhd(plane,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
      int g, sg=1;
      if(family==0){ int q1=c5_loco_q1(b-c),q2=c5_loco_q1(c-a),q3=c5_loco_q1(d-b);
        if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){sg=-1;q1=-q1;q2=-q2;q3=-q3;}
        g=cmap[q1*81+q2*9+q3];
      } else { int e=abs(a-c)+abs(b-c); g=0; for(int t=0;t<5;t++) if(grid_thr[t]<e) g++; }
      if(g<0||g>=nX) return -5;
      /* activity from causal recon (B17-style lv/tv/tl) */
      int lv = (j>0)?plane[i*W+j-1]:((i>0)?plane[(i-1)*W]:0);
      int tv = (i>0)?plane[(i-1)*W+j]:((j>0)?plane[i*W+j-1]:0);
      int tl = (i>0&&j>0)?plane[(i-1)*W+j-1]:0;
      int ee = abs(lv-tl)+abs(tv-tl);
      int ab = c5_act_bin(ee);
      int base = ab*4;
      int L=1;
      for(;;){
        int p = (L-1)<4?(L-1):3;
        int bb2 = c5_bd_dec(&bd, base+p);
        if(bb2==1) break;
        L++; if(L>16) return -6;
      }
      unsigned rem=0;
      for(int k=0;k<L-1;k++){
        int bit;
        if((rawpos>>3)<rlen){ int byte=raw[rawpos>>3]; bit=(byte>>(7-(rawpos&7)))&1; rawpos++; }
        else return -1;
        rem=(rem<<1)|(unsigned)bit;
      }
      unsigned cn = (1u<<(L-1))|rem;
      int M = (int)(cn-1u);
      int sym = c5_M_to_r(M);
      if(sym<-1024||sym>1024) return -4;
      int TL0=(i>0&&j>0)?plane[(i-1)*W+j-1]:0;
      int med5=c5_med_b17(a,b,TL0);
      int pid=predid[g], p;
      if(pid==16){
        int blk=(i>>5)*nbw+(j>>5);
        if(wuse[blk]){
          int Lw=(j>0)?plane[i*W+j-1]:((i>0)?plane[(i-1)*W]:0);
          int Tw=(i>0)?plane[(i-1)*W+j]:((j>0)?plane[i*W+j-1]:0);
          int TLw=(i>0&&j>0)?plane[(i-1)*W+j-1]:0;
          int TRw=(i>0&&j+1<W)?plane[(i-1)*W+j+1]:0;
          p=c5_wpred(ww+4*blk,Lw,Tw,TLw,TRw);
        } else p=pred_e16_c5(0,a,b,c,d,Ww,NNe,NE);
      } else if(pid==17){ p=c5_lms_pred(w0,a,b,c,d,med5);
      } else if(pid==18){ p=c5_lms_pred(w3,a,b,c,d,med5);
      } else p=pred_e16_c5(pid,a,b,c,d,Ww,NNe,NE);
      int yv = p+sg*sym;
      plane[i*W+j]=(int16_t)yv;
      c5_lms_update(w0,a,b,c,d,med5,yv,0);
      c5_lms_update(w3,a,b,c,d,med5,yv,3);
    }
    return 0;
  } else {
    int NN[4]={1,1,1,1}, AA[4]={4,4,4,4};
    size_t bitpos=0;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
      int a,b,c,d,Ww,NNe,NE; c5_nbhd(plane,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
      int g, sg=1;
      if(family==0){ int q1=c5_loco_q1(b-c),q2=c5_loco_q1(c-a),q3=c5_loco_q1(d-b);
        if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){sg=-1;q1=-q1;q2=-q2;q3=-q3;}
        g=cmap[q1*81+q2*9+q3];
      } else { int e=abs(a-c)+abs(b-c); g=0; for(int t=0;t<5;t++) if(grid_thr[t]<e) g++; }
      if(g<0||g>=nX) return -5;
      int lv = (j>0)?plane[i*W+j-1]:((i>0)?plane[(i-1)*W]:0);
      int tv = (i>0)?plane[(i-1)*W+j]:((j>0)?plane[i*W+j-1]:0);
      int tl = (i>0&&j>0)?plane[(i-1)*W+j-1]:0;
      int ee = abs(lv-tl)+abs(tv-tl);
      int ctx = c5_act_bin(ee);
      int Nv=NN[ctx], Av=AA[ctx], k=0;
      while((Nv<<k)<Av){ k++; if(k>30) return -5; }
      unsigned q=0;
      for(;;){
        if((bitpos>>3)>=glen) return -1;
        int byte=gbuf[bitpos>>3]; int bit=(byte>>(7-(bitpos&7)))&1; bitpos++;
        if(bit) break; q++; if(q>5000) return -1;
      }
      unsigned rem=0;
      for(int t=0;t<k;t++){
        if((bitpos>>3)>=glen) return -1;
        int byte=gbuf[bitpos>>3]; int bit=(byte>>(7-(bitpos&7)))&1; bitpos++;
        rem=(rem<<1)|(unsigned)bit;
      }
      unsigned M=(q<<k)|rem;
      int sym = c5_M_to_r((int)M);
      if(sym<-1024||sym>1024) return -4;
      AA[ctx]+= (sym<0?-sym:sym); NN[ctx]+=1;
      if(NN[ctx]>=64){ NN[ctx]>>=1; AA[ctx]>>=1; if(NN[ctx]<1) NN[ctx]=1; }
      int TL0=(i>0&&j>0)?plane[(i-1)*W+j-1]:0;
      int med5=c5_med_b17(a,b,TL0);
      int pid=predid[g], p;
      if(pid==16){
        int blk=(i>>5)*nbw+(j>>5);
        if(wuse[blk]){
          int Lw=(j>0)?plane[i*W+j-1]:((i>0)?plane[(i-1)*W]:0);
          int Tw=(i>0)?plane[(i-1)*W+j]:((j>0)?plane[i*W+j-1]:0);
          int TLw=(i>0&&j>0)?plane[(i-1)*W+j-1]:0;
          int TRw=(i>0&&j+1<W)?plane[(i-1)*W+j+1]:0;
          p=c5_wpred(ww+4*blk,Lw,Tw,TLw,TRw);
        } else p=pred_e16_c5(0,a,b,c,d,Ww,NNe,NE);
      } else if(pid==17){ p=c5_lms_pred(w0,a,b,c,d,med5);
      } else if(pid==18){ p=c5_lms_pred(w3,a,b,c,d,med5);
      } else p=pred_e16_c5(pid,a,b,c,d,Ww,NNe,NE);
      int yv = p+sg*sym;
      plane[i*W+j]=(int16_t)yv;
      c5_lms_update(w0,a,b,c,d,med5,yv,0);
      c5_lms_update(w3,a,b,c,d,med5,yv,3);
    }
    return 0;
  }
}
