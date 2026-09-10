#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
// HAPRE-C: YCoCg-R + causal MED residuals + canonical-Huffman pack/unpack + reconstruct.
static inline int med(int a,int b,int c){ if(c>=a&&c>=b) return a<b?a:b; if(c<=a&&c<=b) return a>b?a:b; return a+b-c; }

void ycocg_fwd(const uint8_t *rgb, int H, int W, int16_t *yc) {
  for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int R=rgb[(i*W+j)*3+0], G=rgb[(i*W+j)*3+1], B=rgb[(i*W+j)*3+2];
    int Co=R-B, t=B+(Co>>1), Cg=G-t, Y=t+(Cg>>1);
    yc[(0*H+i)*W+j]=(int16_t)Y; yc[(1*H+i)*W+j]=(int16_t)Co; yc[(2*H+i)*W+j]=(int16_t)Cg;
  }
}
void ycocg_inv(const int16_t *yc, int H, int W, uint8_t *rgb) {
  for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int Y=yc[(0*H+i)*W+j], Co=yc[(1*H+i)*W+j], Cg=yc[(2*H+i)*W+j];
    int t=Y-(Cg>>1), G=Cg+t, B=t-(Co>>1), R=Co+B;
    if(R<0)R=0; if(R>255)R=255; if(G<0)G=0; if(G>255)G=255; if(B<0)B=0; if(B>255)B=255;
    rgb[(i*W+j)*3+0]=(uint8_t)R; rgb[(i*W+j)*3+1]=(uint8_t)G; rgb[(i*W+j)*3+2]=(uint8_t)B;
  }
}
// Exact causal MED residuals using reconstructed (== originals, lossless) neighbors + histograms over idx=r+256 clipped 0..512.
void med_residuals_exact(const int16_t *img, int H, int W, int16_t *res, int64_t hist[3][2049]) {
  for(int k=0;k<3;k++) for(int i=0;i<2049;i++) hist[k][i]=0;
  for(int k=0;k<3;k++){
    const int16_t *P=img+k*H*W; int16_t *R=res+k*H*W;
    for(int i=0;i<H;i++){
      for(int j=0;j<W;j++){
        int a,b,c;
        if(i==0&&j==0){a=b=c=0;}
        else if(i==0){a=P[j-1];b=c=a;}
        else if(j==0){b=P[(i-1)*W];a=c=b;}
        else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];}
        int p=med(a,b,c);
        int r=(int)P[i*W+j]-p;
        R[i*W+j]=(int16_t)r;
        int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
        hist[k][idx]++;
      }
    }
  }
}
size_t huff_pack(const int16_t *res, int N, const uint32_t *codes, const uint8_t *lens, uint8_t *out, size_t outcap) {
  size_t bytepos=0; uint64_t buf=0; int nbits=0;
  for(int n=0;n<N;n++){
    int r=res[n]; int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
    buf=(buf<<lens[idx])|codes[idx]; nbits+=lens[idx];
    while(nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0); }
  }
  if(nbits){ if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf<<(8-nbits)); }
  return bytepos;
}
int huff_unpack(const uint8_t *in_, size_t inlen, int N, const uint32_t *codes, const uint8_t *lens, int16_t *res) {
  // Fast path: 8-bit LUT for codes with len<=8; tree fallback otherwise.
  int16_t lut_sym[256]; uint8_t lut_len[256];
  for(int i=0;i<256;i++){lut_sym[i]=-9999;lut_len[i]=0;}
  int maxnodes=1400; int (*ch)[2]=(int(*)[2])malloc(sizeof(int)*2*maxnodes);
  int *sval=(int*)malloc(sizeof(int)*maxnodes);
  if(!ch||!sval){free(ch);free(sval);return -3;}
  for(int i=0;i<maxnodes;i++){ch[i][0]=ch[i][1]=-1;sval[i]=-9999;}
  int nn=1;
  for(int s=0;s<2049;s++){ uint8_t ln=lens[s]; if(!ln) continue; uint32_t cd=codes[s];
    if(ln<=8){ uint32_t base=cd<<(8-ln); for(uint32_t t=0;t<(1u<<(8-ln));t++){ lut_sym[base|t]=(int16_t)(s-1024); lut_len[base|t]=ln; } }
    int node=0;
    for(int b=ln-1;b>=0;b--){ int bit=(cd>>b)&1;
      if(ch[node][bit]==-1){ if(nn>=maxnodes){free(ch);free(sval);return -2;} ch[node][bit]=nn; ch[nn][0]=ch[nn][1]=-1; sval[nn]=-9999; nn++; }
      node=ch[node][bit]; }
    sval[node]=s-1024; }
  size_t bitpos=0; size_t nbytes=inlen;
  // bit reader with 16-bit window
  for(int n=0;n<N;n++){
    size_t byte=bitpos>>3; int off=bitpos&7;
    uint16_t win;
    if(byte+1<nbytes) win=((uint16_t)in_[byte]<<8)|in_[byte+1];
    else if(byte<nbytes) win=((uint16_t)in_[byte]<<8);
    else {free(ch);free(sval);return -1;}
    uint8_t key=(win>>(8-off))&0xFF;
    if(lut_len[key]){ res[n]=lut_sym[key]; bitpos+=lut_len[key]; continue; }
    int node=0;
    for(;;){ if((bitpos>>3)>=nbytes){free(ch);free(sval);return -1;}
      int bt=in_[bitpos>>3]; int bit=(bt>>(7-(bitpos&7)))&1; bitpos++; node=ch[node][bit];
      if(node<0){free(ch);free(sval);return -1;}
      if(sval[node]!=-9999){ res[n]=(int16_t)sval[node]; break; } } }
  free(ch);free(sval); return 0;
}
void med_reconstruct(const int16_t *res, int H, int W, int16_t *img) {
  for(int k=0;k<3;k++){
    const int16_t *R=res+k*H*W; int16_t *P=img+k*H*W;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
      int a,b,c;
      if(i==0&&j==0){a=b=c=0;}
      else if(i==0){a=P[j-1];b=c=a;}
      else if(j==0){b=P[(i-1)*W];a=c=b;}
      else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];}
      // NOTE: identical convention to med_residuals_exact (required for lossless round-trip).
      P[i*W+j]=(int16_t)(med(a,b,c)+R[i*W+j]);
    }
  }
}
// ---- CTX9 extension: energy-class contexts (0..8) per channel, shared MED core ----
void med_res_ctx9(const int16_t *img, int H, int W, int16_t *res, uint8_t *ctx, int64_t hist[3][9][2049]) {
  for(int k=0;k<3;k++) for(int e=0;e<9;e++) for(int i=0;i<2049;i++) hist[k][e][i]=0;
  for(int k=0;k<3;k++){
    const int16_t *P=img+k*H*W; int16_t *R=res+k*H*W; uint8_t *C=ctx+k*H*W;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
      int a,b,c;
      if(i==0&&j==0){a=b=c=0;} else if(i==0){a=P[j-1];b=c=a;} else if(j==0){b=P[(i-1)*W];a=c=b;}
      else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];}
      int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
      int r=(int)P[i*W+j]-med(a,b,c);
      R[i*W+j]=(int16_t)r; C[i*W+j]=(uint8_t)e;
      int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
      hist[k][e][idx]++;
    }
  }
}
// pack with per-(ch,ctx) tables: codes[k][e][2049], lens[k][e][2049]
size_t huff_pack_ctx9(const int16_t *res, const uint8_t *ctx, int N, const uint32_t *codes, const uint8_t *lens, uint8_t *out, size_t outcap) {
  size_t bytepos=0; uint64_t buf=0; int nbits=0;
  for(int n=0;n<N*3;n++){
    int k=n/N, m=n%N; int e=ctx[n];
    int r=res[n]; int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
    size_t t=((size_t)k*9+e)*2049+idx;
    buf=(buf<<lens[t])|codes[t]; nbits+=lens[t];
    while(nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0); }
  }
  if(nbits){ if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf<<(8-nbits)); }
  return bytepos;
}
int huff_unpack_ctx9(const uint8_t *in_, size_t inlen, int N, const uint32_t *codes, const uint8_t *lens, int16_t *res, uint8_t *ctx, int H, int W) {
  // decode symbol-by-symbol with tree per (k,e); rebuild ctx on the fly from recon (streaming, O(W) rows kept via full img buffer here)
  // Build per-(k,e) decode trees
  int T=27, maxnodes=1100;
  static int ch[27][1100][2]; static int sv[27][1100];
  for(int t=0;t<T;t++) for(int i=0;i<maxnodes;i++){ch[t][i][0]=ch[t][i][1]=-1;sv[t][i]=-9999;}
  for(int k=0;k<3;k++) for(int e=0;e<9;e++){ int t=k*9+e, nn=1;
    for(int s=0;s<2049;s++){ size_t u=t*2049+s; uint8_t ln=lens[u]; if(!ln) continue; uint32_t cd=codes[u];
      int node=0;
      for(int b=ln-1;b>=0;b--){ int bit=(cd>>b)&1;
        if(ch[t][node][bit]==-1){ if(nn>=maxnodes) return -2; ch[t][node][bit]=nn; ch[t][nn][0]=ch[t][nn][1]=-1; sv[t][nn]=-9999; nn++; }
        node=ch[t][node][bit]; }
      sv[t][node]=s-1024; } }
  // recon image buffer for ctx computation
  int16_t *img=(int16_t*)calloc((size_t)3*N,sizeof(int16_t));
  if(!img) return -3;
  size_t bitpos=0;
  for(int k=0;k<3;k++) for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int16_t *P=img+k*N;
    int a,b,c;
    if(i==0&&j==0){a=b=c=0;} else if(i==0){a=P[j-1];b=c=a;} else if(j==0){b=P[(i-1)*W];a=c=b;}
    else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];}
    int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
    int t=k*9+e, node=0, sym=0;
    for(;;){ if((bitpos>>3)>=inlen){free(img);return -1;}
      int byte=in_[bitpos>>3]; int bit=(byte>>(7-(bitpos&7)))&1; bitpos++; node=ch[t][node][bit];
      if(node<0){free(img);return -1;}
      if(sv[t][node]!=-9999){ sym=sv[t][node]; break; } }
    int p=med(a,b,c);
    P[i*W+j]=(int16_t)(p+sym);
    res[(k*N+i*W+j)]=(int16_t)sym; ctx[k*N+i*W+j]=(uint8_t)e;
  }
  free(img); return 0;
}
// ---- NL-LS extension: per-(ch,ctx) 6-tap fixed-point predictor, coeffs int16 Q8 ----
void nlls_res_ctx9(const int16_t *img, int H, int W, const int16_t *coef, const uint8_t *medflag, int16_t *res, uint8_t *ctx, int64_t hist[3][9][2049]) {
  for(int k=0;k<3;k++) for(int e=0;e<9;e++) for(int i=0;i<2049;i++) hist[k][e][i]=0;
  for(int k=0;k<3;k++){
    const int16_t *P=img+k*H*W; int16_t *R=res+k*H*W; uint8_t *C=ctx+k*H*W;
    const int16_t *Wc=coef+k*9*6;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
      int a,b,c;
      if(i==0&&j==0){a=b=c=0;} else if(i==0){a=P[j-1];b=c=a;} else if(j==0){b=P[(i-1)*W];a=c=b;}
      else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];}
      int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
      int t=k*9+e, p;
      if(medflag[t]) p=med(a,b,c);
      else { const int16_t *w=Wc+e*6;
        p=(w[0]+w[1]*a+w[2]*b+w[3]*c+w[4]*abs(a-b)+w[5]*(a+b-2*c)+128)>>8; }
      int r=(int)P[i*W+j]-p;
      R[i*W+j]=(int16_t)r; C[i*W+j]=(uint8_t)e;
      int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
      hist[k][e][idx]++;
    }
  }
}
// NL-LS streaming decode: recompute ctx+pred from recon, decode symbol, reconstruct.
int nlls_unpack(const uint8_t *in_, size_t inlen, int N, const int16_t *coef, const uint8_t *medflag, const uint32_t *codes, const uint8_t *lens, int16_t *res, int16_t *img_out, int H, int W) {
  int T=27, maxnodes=1100;
  static int ch[27][1100][2]; static int sv[27][1100];
  for(int t=0;t<T;t++) for(int i=0;i<maxnodes;i++){ch[t][i][0]=ch[t][i][1]=-1;sv[t][i]=-9999;}
  for(int k=0;k<3;k++) for(int e=0;e<9;e++){ int t=k*9+e, nn=1;
    for(int s=0;s<2049;s++){ size_t u=t*2049+s; uint8_t ln=lens[u]; if(!ln) continue; uint32_t cd=codes[u];
      int node=0;
      for(int b=ln-1;b>=0;b--){ int bit=(cd>>b)&1;
        if(ch[t][node][bit]==-1){ if(nn>=maxnodes) return -2; ch[t][node][bit]=nn; ch[t][nn][0]=ch[t][nn][1]=-1; sv[t][nn]=-9999; nn++; }
        node=ch[t][node][bit]; }
      sv[t][node]=s-1024; } }
  int16_t *img=(int16_t*)calloc((size_t)3*N,sizeof(int16_t));
  if(!img) return -3;
  size_t bitpos=0;
  for(int k=0;k<3;k++){ int16_t *P=img+k*N; const int16_t *Wc=coef+k*9*6;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
      int a,b,c;
      if(i==0&&j==0){a=b=c=0;} else if(i==0){a=P[j-1];b=c=a;} else if(j==0){b=P[(i-1)*W];a=c=b;}
      else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];}
      int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
      int t=k*9+e, node=0, sym=0;
      for(;;){ if((bitpos>>3)>=inlen){free(img);return -1;}
        int byte=in_[bitpos>>3]; int bit=(byte>>(7-(bitpos&7)))&1; bitpos++; node=ch[t][node][bit];
        if(node<0){free(img);return -1;}
        if(sv[t][node]!=-9999){ sym=sv[t][node]; break; } }
      int p;
      if(medflag[t]) p=med(a,b,c);
      else { const int16_t *w=Wc+e*6;
        p=(w[0]+w[1]*a+w[2]*b+w[3]*c+w[4]*abs(a-b)+w[5]*(a+b-2*c)+128)>>8; }
      P[i*W+j]=(int16_t)(p+sym);
      res[k*N+i*W+j]=(int16_t)sym;
    } }
  if(img_out) memcpy(img_out,img,(size_t)3*N*sizeof(int16_t));
  free(img); return 0;
}
// ---- rANS order-0 (M=10 precision, L=2^20 state, 8-bit renorm) per plane ----
#define RANS_L (1u<<20)
#define RANS_M 14
#define RANS_N (1u<<RANS_M)
void rans_norm(const int64_t *hist, int A, const int *syms, uint16_t *freq, int *cum) {
  int64_t tot=0; for(int i=0;i<A;i++) tot+=hist[syms[i]];
  int rem=RANS_N;
  for(int i=0;i<A;i++){ int f=(int)((hist[syms[i]]*RANS_N+tot/2)/tot); if(f<1)f=1; freq[i]=(uint16_t)f; rem-=f; }
  int bi=0; for(int i=1;i<A;i++) if(hist[syms[i]]>hist[syms[bi]]) bi=i;
  freq[bi]=(uint16_t)((int)freq[bi]+rem);
  int c=0; for(int i=0;i<A;i++){ cum[i]=c; c+=freq[i]; }
}
// out layout: [bytes...][4B final state LE]; encode reads symbols last->first.
size_t rans_encode(const int16_t *syms_in, int N, const int *lut, const uint16_t *freq, const int *cum, uint8_t *out) {
  uint32_t x=RANS_L; uint8_t *p=out;
  // need cum: rebuild from freq
  for(int n=N-1;n>=0;n--){
    int r=syms_in[n]; int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
    int s=lut[idx]; uint32_t f=freq[s]; uint32_t cdf=(uint32_t)cum[s];
    while(x>=(((RANS_L>>RANS_M)<<8))*f){ *p++=(uint8_t)(x&0xFF); x>>=8; }
    x=((x/f)<<RANS_M)+(x%f)+cdf;
  }
  *p++=(uint8_t)(x&0xFF); *p++=(uint8_t)((x>>8)&0xFF); *p++=(uint8_t)((x>>16)&0xFF); *p++=(uint8_t)((x>>24)&0xFF);
  return (size_t)(p-out);
}
void rans_slots(const uint16_t *freq, const int *cum, int A, int *slot2sym) {
  for(int s=0;s<A;s++) for(uint32_t i=0;i<freq[s];i++) slot2sym[cum[s]+i]=s;
}
int rans_decode(const uint8_t *in_, size_t inlen, int N, const int *syms, const uint16_t *freq, const int *cum, int A, const int *slot2sym, int16_t *out) {
  (void)syms; (void)A;
  if(inlen<4) return -1;
  const uint8_t *p=in_+inlen-4;
  uint32_t x=(uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
  const uint8_t *bot=in_;
  for(int n=0;n<N;n++){
    uint32_t slot=x&(RANS_N-1u);
    int s=slot2sym[slot];
    if(s<0) return -1;
    out[n]=(int16_t)(syms[s]-1024);
    x=freq[s]*(x>>RANS_M)+slot-(uint32_t)cum[s];
    while(x<RANS_L){ if(p<=bot){ if(n==N-1) break; return -1; } p--; x=(x<<8)|*p; }
  }
  return 0;
}
// ---- RUN extension: flat-run two-stream coding on MED+CTX9 path ----
// Encode: single causal pass. When (j+1<W && a==b && b==c && c==d): count run L of pixels equal to a
// (from originals), append L to runs[], emit run pixels' recon (no main symbols). Else regular MED symbol.
// Interruption pixel (first non-matching or row end handling): regular path symbol.
// Row end: runs stop at row end (no interruption symbol consumed).
// out_main_sym/out_main_ctx: parallel arrays; out_runs: lengths; returns counts via n_main/n_runs.
void med_run_encode(const int16_t *img, int H, int W, int16_t *msym, uint8_t *mctx, int *runs, int *n_main, int *n_runs, int64_t hist[3][9][2049]) {
  for(int k=0;k<3;k++) for(int e=0;e<9;e++) for(int i=0;i<2049;i++) hist[k][e][i]=0;
  *n_main=0; *n_runs=0;
  for(int k=0;k<3;k++){
    const int16_t *P=img+k*H*W;
    for(int i=0;i<H;i++){ int j=0;
      while(j<W){
        int a,b,c,d;
#define NBHD \
        if(i==0&&j==0){a=b=c=d=0;} \
        else if(i==0){a=P[j-1];b=c=a;d=a;} \
        else if(j==0){b=P[(i-1)*W];a=c=b;d=P[(i-1)*W+1];} \
        else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];d=(j+1<W)?P[(i-1)*W+j+1]:b;}
        NBHD
        if(j+1<W && a==b && b==c && c==d && !(i==0&&j==0)){
          int L=0;
          while(j+L<W && P[i*W+j+L]==a) L++;
          runs[(*n_runs)++]=L; j+=L;
          if(getenv("HAPRE_TRACE")) fprintf(stderr,"ENC k=%d i=%d j=%d a=%d L=%d\n",k,i,j-L,a,L);
          if(j>=W) break;
          NBHD /* fresh neighbors for forced interruption symbol (no retest) */
        }
        int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
        int r=(int)P[i*W+j]-med(a,b,c);
        int n=*n_main;
        msym[n]=(int16_t)r; mctx[n]=(uint8_t)(k*9+e);
        int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
        hist[k][e][idx]++;
        (*n_main)++; j++;
      }
    }
  }
}
// gamma writer/reader for run lengths (L>=1)
static void gamma_put(uint8_t *o, size_t oc, size_t *bp, unsigned L){
  int b=0; unsigned t=L; while(t>>=1) b++;
  for(int i=0;i<b;i++){ /*0 bit*/ *bp+=1; }
  // write bits MSB-first of L (b+1 bits)
  for(int i=b;i>=0;i--){ size_t p=*bp; if((p>>3)>=oc) return; if((L>>i)&1) o[p>>3]|=(uint8_t)(1<<(7-(p&7))); (*bp)++; }
  // NOTE: zero bits need no set (buffer must be zeroed by caller)
}
static unsigned gamma_get(const uint8_t *in_, size_t inlen, size_t *bp){
  int b=0;
  for(;;){ size_t p=*bp; unsigned bit=(p>>3)<inlen?((in_[p>>3]>>(7-(p&7)))&1):0; (*bp)++; if(bit) break; b++; }
  unsigned v=1;
  for(int i=0;i<b;i++){ size_t p=*bp; unsigned bit=(p>>3)<inlen?((in_[p>>3]>>(7-(p&7)))&1):0; (*bp)++; v=(v<<1)|bit; }
  return v;
}
size_t runs_pack(const int *runs, int n_runs, uint8_t *out, size_t outcap){
  size_t bp=0; memset(out,0,outcap);
  for(int i=0;i<n_runs;i++) gamma_put(out,outcap,&bp,(unsigned)runs[i]+1u);
  return (bp+7)>>3;
}
// Streaming run decode with recon: mirrors med_run_encode state machine.
// main stream decoded via per-(k,e) trees (same as huff_unpack_ctx9 but symbols pulled on demand);
// run stream consumed at run conditions. Produces recon img planes.
int med_run_unpack(const uint8_t *main_, size_t mainlen, const uint8_t *run_, size_t runlen,
                   const uint32_t *codes, const uint8_t *lens, int16_t *img, int H, int W){
  int T=27, maxnodes=1100;
  static int ch[27][1100][2]; static int sv[27][1100];
  for(int t=0;t<T;t++) for(int i=0;i<maxnodes;i++){ch[t][i][0]=ch[t][i][1]=-1;sv[t][i]=-9999;}
  for(int k=0;k<3;k++) for(int e=0;e<9;e++){ int t=k*9+e, nn=1;
    for(int s2=0;s2<2049;s2++){ size_t u=t*2049+s2; uint8_t ln=lens[u]; if(!ln) continue; uint32_t cd=codes[u];
      int node=0;
      for(int b=ln-1;b>=0;b--){ int bit=(cd>>b)&1;
        if(ch[t][node][bit]==-1){ if(nn>=maxnodes) return -2; ch[t][node][bit]=nn; ch[t][nn][0]=ch[t][nn][1]=-1; sv[t][nn]=-9999; nn++; }
        node=ch[t][node][bit]; }
      sv[t][node]=s2-1024; } }
  size_t mbp=0, rbp=0;
  for(int k=0;k<3;k++){ int16_t *P=img+k*H*W;
    for(int i=0;i<H;i++){ int j=0;
      while(j<W){
        int a,b,c,d;
        if(i==0&&j==0){a=b=c=d=0;}
        else if(i==0){a=P[j-1];b=c=a;d=a;}
        else if(j==0){b=P[(i-1)*W];a=c=b;d=P[(i-1)*W+1];}
        else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];d=(j+1<W)?P[(i-1)*W+j+1]:b;}
        if(j+1<W && a==b && b==c && c==d && !(i==0&&j==0)){
          unsigned L=gamma_get(run_,runlen,&rbp);
          if(L<1){ return -3; }
          L-=1;
          if(getenv("HAPRE_TRACE")) fprintf(stderr,"RUN k=%d i=%d j=%d a=%d L=%u\n",k,i,j,a,L);
          for(unsigned t2=0;t2<L && j<W;t2++,j++) P[i*W+j]=(int16_t)a;
          if(j>=W) break;
          if(i==0&&j==0){a=b=c=d=0;}
          else if(i==0){a=P[j-1];b=c=a;d=a;}
          else if(j==0){b=P[(i-1)*W];a=c=b;d=P[(i-1)*W+1];}
          else{a=P[i*W+j-1];b=P[(i-1)*W+j];c=P[(i-1)*W+j-1];d=(j+1<W)?P[(i-1)*W+j+1]:b;}
        }
        int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
        int t=k*9+e, node=0, sym=0;
        for(;;){ if((mbp>>3)>=mainlen) return -1;
          int byte=main_[mbp>>3]; int bit=(byte>>(7-(mbp&7)))&1; mbp++; node=ch[t][node][bit];
          if(node<0) return -1;
          if(sv[t][node]!=-9999){ sym=sv[t][node]; break; } }
        P[i*W+j]=(int16_t)(med(a,b,c)+sym);
        j++;
      }
    } }
  return 0;
}
// generic ordered packer for (symbol, tablekey) streams
size_t pack_syms(const int16_t *syms, const uint8_t *keys, int N, const uint32_t *codes, const uint8_t *lens, uint8_t *out, size_t outcap){
  size_t bytepos=0; uint64_t buf=0; int nbits=0;
  for(int n=0;n<N;n++){
    int r=syms[n]; int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
    size_t t=(size_t)keys[n]*2049+idx;
    buf=(buf<<lens[t])|codes[t]; nbits+=lens[t];
    while(nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0); }
  }
  if(nbits){ if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf<<(8-nbits)); }
  return bytepos;
}
// ---- µMoE experts: all integer, floor-division semantics (must match Python //) ----
static inline int fdiv(int a, int b){ int q=a/b, r=a%b; if(r!=0 && ((r<0)!=(b<0))) q--; return q; } // floor div
// experts: 0 MED, 1 TOP(b), 2 PAETH, 3 GRAD((a+b)/2+(b-c)/4 floor), 4 GAP, 5 DG((c+d)/2 needs d!)
static inline int pred_expert(int x, int a,int b,int c,int d,int Ww,int NNe,int NE){
  switch(x){
    case 0: { if(c>=a&&c>=b) return a<b?a:b; if(c<=a&&c<=b) return a>b?a:b; return a+b-c; }
    case 1: return b;
    case 2: { int p=a+b-c,pa=abs(p-a),pb=abs(p-b),pc=abs(p-c);
      if(pa<=pb&&pa<=pc) return a; else if(pb<=pc) return b; else return c; }
    case 3: return fdiv(a+b,2)+fdiv(b-c,4);
    case 4: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>80) return a; else if(gv-gh<-80) return b; else return fdiv(a+b,2)+fdiv(NE-c,4); }
    case 5: return fdiv(c+d,2);
    case 6: return a; /* LEFT */
    case 7: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>32) return a; else if(gv-gh<-32) return b; else return fdiv(a+b,2)+fdiv(NE-c,4); }
    case 8: { int gh=abs(a-Ww)+abs(b-c)+abs(b-NE), gv=abs(a-c)+abs(b-NNe)+abs(NE-b);
      if(gv-gh>16) return a; else if(gv-gh<-16) return b; else return fdiv(a+b,2)+fdiv(NE-c,4); }
    case 9: return fdiv(a+b,2); /* AVG_AB */
    case 10: return a+b-c; /* PLANE */
    case 11: return fdiv(a+c,2); /* AC */
    case 12: return c; /* C */
    case 13: return fdiv(b+c,2); /* BC */
    case 14: return d; /* D */
    default: return fdiv(a+b+c,3); /* AVG3 */
  }
}
// ---- µMoE+RUN integrated codec: per-8x8-block expert ids (0..5), run machine, CTX9 tables ----
#define MOE_B 8
static inline void moe_nbhd(const int16_t *P,int H,int W,int i,int j,int *a,int *b,int *c,int *d,int *Ww,int *NNe,int *NE){
  if(i==0&&j==0){*a=*b=*c=*d=0;*Ww=0;*NNe=0;*NE=0;}
  else if(i==0){*a=P[j-1];*b=*c=*a;*d=*a;*Ww=(j>=2)?P[j-2]:*a;*NNe=*a;*NE=*a;}
  else if(j==0){*b=P[(i-1)*W];*a=*c=*b;*d=P[(i-1)*W+1];*Ww=*a;*NNe=(i>=2)?P[(i-2)*W]:*b;*NE=*d;}
  else{*a=P[i*W+j-1];*b=P[(i-1)*W+j];*c=P[(i-1)*W+j-1];*d=(j+1<W)?P[(i-1)*W+j+1]:*b;
       *Ww=(j>=2)?P[i*W+j-2]:*a;*NNe=(i>=2)?P[(i-2)*W+j]:*b;*NE=(j+1<W)?P[(i-1)*W+j+1]:*b;}
}
void moe_run_encode(const int16_t *img,int H,int W,uint8_t *ids,int16_t *msym,uint8_t *mctx,int *runs,int *n_main,int *n_runs,int64_t hist[3][9][2049]){
  for(int k=0;k<3;k++) for(int e=0;e<9;e++) for(int i=0;i<2049;i++) hist[k][e][i]=0;
  *n_main=0;*n_runs=0;
  int nbw=(W+MOE_B-1)/MOE_B, nbh=(H+MOE_B-1)/MOE_B;
  for(int k=0;k<3;k++){
    const int16_t *P=img+k*H*W;
    // phase 1: expert choice per block (L1 proxy on regular-path residuals)
    for(int by=0;by<nbh;by++) for(int bx=0;bx<nbw;bx++){
      long best=2147483647L; int be=0;
      for(int x=0;x<6;x++){
        long s=0;
        for(int i=by*MOE_B;i<(by+1)*MOE_B&&i<H;i++) for(int j=bx*MOE_B;j<(bx+1)*MOE_B&&j<W;j++){
          int a,b,c,d,Ww,NNe,NE; moe_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
          int p=pred_expert(x,a,b,c,d,Ww,NNe,NE);
          int r=(int)P[i*W+j]-p; s+=(r<0?-r:r);
          if(s>=best) break;
        }
        if(s<best){best=s;be=x;}
      }
      ids[(k*nbh+by)*nbw+bx]=(uint8_t)be;
    }
    // phase 2: run machine with expert preds
    for(int i=0;i<H;i++){ int j=0;
      while(j<W){
        int a,b,c,d,Ww,NNe,NE; moe_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
        if(j+1<W && a==b && b==c && c==d && !(i==0&&j==0)){
          int L=0;
          while(j+L<W && P[i*W+j+L]==a) L++;
          runs[(*n_runs)++]=L; j+=L;
          if(j>=W) break;
          moe_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
        }
        int x=ids[(k*nbh+i/MOE_B)*nbw+j/MOE_B];
        int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
        int r=(int)P[i*W+j]-pred_expert(x,a,b,c,d,Ww,NNe,NE);
        int n=*n_main;
        msym[n]=(int16_t)r; mctx[n]=(uint8_t)(k*9+e);
        int idx=r+1024; assert(idx>=0&&idx<=2048); if(idx<0)idx=0; if(idx>2048)idx=2048;
        hist[k][e][idx]++;
        (*n_main)++; j++;
      }
    }
  }
}
int moe_run_unpack(const uint8_t *main_,size_t mainlen,const uint8_t *run_,size_t runlen,
  const uint32_t *codes,const uint8_t *lens,const uint8_t *ids,int16_t *img,int H,int W){
  int T=27, maxnodes=1400;
  static int ch[27][1400][2]; static int sv[27][1400];
  for(int t=0;t<T;t++) for(int i=0;i<maxnodes;i++){ch[t][i][0]=ch[t][i][1]=-1;sv[t][i]=-9999;}
  for(int k=0;k<3;k++) for(int e=0;e<9;e++){ int t=k*9+e, nn=1;
    for(int s2=0;s2<2049;s2++){ size_t u=t*2049+s2; uint8_t ln=lens[u]; if(!ln) continue; uint32_t cd=codes[u];
      int node=0;
      for(int b=ln-1;b>=0;b--){ int bit=(cd>>b)&1;
        if(ch[t][node][bit]==-1){ if(nn>=maxnodes) return -2; ch[t][node][bit]=nn; ch[t][nn][0]=ch[t][nn][1]=-1; sv[t][nn]=-9999; nn++; }
        node=ch[t][node][bit]; }
      sv[t][node]=s2-1024; } }
  int nbw=(W+MOE_B-1)/MOE_B, nbh=(H+MOE_B-1)/MOE_B;
  size_t mbp=0, rbp=0;
  for(int k=0;k<3;k++){ int16_t *P=img+k*H*W;
    for(int i=0;i<H;i++){ int j=0;
      while(j<W){
        int a,b,c,d,Ww,NNe,NE; moe_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
        if(j+1<W && a==b && b==c && c==d && !(i==0&&j==0)){
          unsigned L=gamma_get(run_,runlen,&rbp);
          if(L<1) return -3; L-=1;
          for(unsigned t2=0;t2<L && j<W;t2++,j++) P[i*W+j]=(int16_t)a;
          if(j>=W) break;
          moe_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
        }
        int x=ids[(k*nbh+i/MOE_B)*nbw+j/MOE_B];
        int e=(abs(a-c)+abs(b-c))>>4; if(e>8)e=8;
        int t=k*9+e, node=0, sym=0;
        for(;;){ if((mbp>>3)>=mainlen) return -1;
          int byte=main_[mbp>>3]; int bit=(byte>>(7-(mbp&7)))&1; mbp++; node=ch[t][node][bit];
          if(node<0) return -1;
          if(sv[t][node]!=-9999){ sym=sv[t][node]; break; } }
        P[i*W+j]=(int16_t)(pred_expert(x,a,b,c,d,Ww,NNe,NE)+sym);
        j++;
      }
    } }
  return 0;
}
// ---- LZ77 core (byte stream, 4KB window, min-match 4, greedy, hash chains) ----
// tokens: caller consumes via callbacks? Simple: produce (is_match,len,dist|lit) arrays.
#define LZ_WIN 4096
#define LZ_MIN 4
#define LZ_MAX 264
#define LZ_HASHN 4096
static unsigned lz_h(const uint8_t *p){ return ((p[0]*31u+p[1])*31u+p[2])&(LZ_HASHN-1); }
// out_lit: literals+matchlens interleave? Use two arrays: toks (u16: <256 lit, else 256+len-4), dists (u16, valid when match).
// returns n_toks. head[]/prev[] are scratch (LZ_HASHN ints + N ints).
int lz_encode(const uint8_t *in_, int N, uint16_t *toks, uint16_t *dists, int *head, int *prev){
  for(int i=0;i<LZ_HASHN;i++) head[i]=-1;
  int nt=0;
  for(int i=0;i<N;){
    int bestl=0,bestd=0;
    if(i+ LZ_MIN<=N){
      unsigned h=lz_h(in_+i);
      int cand=head[h], tries=0;
      while(cand>=0 && tries<32){
        int d=i-cand;
        if(d>0&&d<=LZ_WIN){
          int l=0;
          while(l<LZ_MAX && i+l<N && in_[cand+l]==in_[i+l]) l++;
          if(l>bestl){bestl=l;bestd=d; if(l>=32) break;}
        }
        cand=prev[cand]; tries++;
      }
      // insert current (and intervening positions lazily: only current for speed)
      prev[i]=head[h]; head[h]=i;
    }
    if(bestl>=LZ_MIN){ toks[nt]=(uint16_t)(256+bestl-LZ_MIN); dists[nt]=(uint16_t)bestd; nt++;
      // insert skipped positions
      for(int k=1;k<bestl && i+k+LZ_MIN<=N;k++){ unsigned h=lz_h(in_+i+k); prev[i+k]=head[h]; head[h]=i+k; }
      i+=bestl;
    } else { toks[nt]=in_[i]; dists[nt]=0; nt++; i++; }
  }
  return nt;
}
int lz_decode(const uint16_t *toks, const uint16_t *dists, int nt, uint8_t *out, int cap){
  int o=0;
  for(int t=0;t<nt;t++){
    if(toks[t]<256){ if(o>=cap) return -1; out[o++]=(uint8_t)toks[t]; }
    else { int l=toks[t]-256+LZ_MIN, d=dists[t];
      if(d<=0||d>o||o+l>cap) return -1;
      for(int k=0;k<l;k++) out[o]=out[o-d],o++; }
  }
  return o;
}
// ---- CROWN path: zero-border nbhd (probe_b4_b mirror) + LOCO-365 keys/signs ----
static inline void crown_nbhd(const int16_t *P,int H,int W,int i,int j,
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
static inline int loco_q1(int g){ int a=g<0?-g:g; if(g==0) return 0; if(a<=2) return g<0?-1:1; if(a<=7) return g<0?-2:2; if(a<=21) return g<0?-3:3; return g<0?-4:4; }
// key 0..728, sgn +-1
void crown_keys(const int16_t *img,int H,int W,int16_t *key,int8_t *sgn){
  for(int k=0;k<3;k++){
    const int16_t *P=img+k*H*W; int16_t *K=key+k*H*W; int8_t *S=sgn+k*H*W;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++){
      int a,b,c,d,Ww,NNe,NE; crown_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
      int q1=loco_q1(b-c),q2=loco_q1(c-a),q3=loco_q1(d-b);
      int s=1;
      if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){s=-1;q1=-q1;q2=-q2;q3=-q3;}
      K[i*W+j]=(int16_t)(q1*81+q2*9+q3); S[i*W+j]=(int8_t)s;
    }
  }
}
// ---- CROWN per-channel streaming decode ----
int crown_unpack_ch(const uint8_t *in_, size_t inlen, int N,
  const uint32_t *codes, const uint8_t *lens, const uint8_t *predid,
  const uint16_t *cmap, int16_t *P, int H, int W) {
  int T=36, maxnodes=1400;
  int (*ch)[1400][2]=0; int (*sv)[1400]=0;
  ch=(int(*)[1400][2])calloc((size_t)T*1400*2,sizeof(int));
  sv=(int(*)[1400])calloc((size_t)T*1400,sizeof(int));
  if(!ch||!sv){free(ch);free(sv);return -3;}
  for(int t=0;t<T;t++){ for(int i=0;i<1400;i++){ch[t][i][0]=ch[t][i][1]=-1;sv[t][i]=-9999;}
    int nn=1;
    for(int s2=0;s2<2049;s2++){ size_t u=(size_t)t*2049+s2; uint8_t ln=lens[u]; if(!ln) continue; uint32_t cd=codes[u];
      int node=0;
      for(int b=ln-1;b>=0;b--){ int bit=(cd>>b)&1;
        if(ch[t][node][bit]==-1){ if(nn>=1400){free(ch);free(sv);return -2;} ch[t][node][bit]=nn; ch[t][nn][0]=ch[t][nn][1]=-1; sv[t][nn]=-9999; nn++; }
        node=ch[t][node][bit]; }
      sv[t][node]=s2-1024; } }
  size_t bitpos=0;
  for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int a,b,c,d,Ww,NNe,NE; crown_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
    int q1=loco_q1(b-c),q2=loco_q1(c-a),q3=loco_q1(d-b), sg=1;
    if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){sg=-1;q1=-q1;q2=-q2;q3=-q3;}
    int key=q1*81+q2*9+q3;
    int g=cmap[key];
    int x=predid[g];
    int t=g, node=0, sym=0;
    for(;;){ if((bitpos>>3)>=inlen){free(ch);free(sv);return -1;}
      int byte=in_[bitpos>>3]; int bit=(byte>>(7-(bitpos&7)))&1; bitpos++; node=ch[t][node][bit];
      if(node<0){free(ch);free(sv);return -1;}
      if(sv[t][node]!=-9999){ sym=sv[t][node]; break; } }
    P[i*W+j]=(int16_t)(pred_expert(x,a,b,c,d,Ww,NNe,NE)+sg*sym);
  }
  free(ch);free(sv); return 0;
}
// ---- CROWN rANS assembly decode: pre-decoded per-group symbol arrays + scan assembly ----
int crown_assemble(const int16_t *syms, const int *goff, int ngroups,
  const uint16_t *cmap, const uint8_t *predid, int16_t *img, int H, int W, int ch) {
  // syms: concatenated group streams; goff[g..g+1] span; cmap: [729]; predid: [36]
  int cur[128]; if(ngroups>128) return -2;
  for(int g=0;g<ngroups;g++) cur[g]=goff[g];
  int16_t *P=img+ch*H*W;
  (void)ch;
  for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int a,b,c,d,Ww,NNe,NE; crown_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
    int q1=loco_q1(b-c),q2=loco_q1(c-a),q3=loco_q1(d-b), sg=1;
    if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){sg=-1;q1=-q1;q2=-q2;q3=-q3;}
    int g=cmap[q1*81+q2*9+q3];
    if(g<0||g>=ngroups) return -1;
    if(cur[g]>=goff[g+1]) return -1;
    int sym=syms[cur[g]++];
    P[i*W+j]=(int16_t)(pred_expert(predid[g],a,b,c,d,Ww,NNe,NE)+sg*sym);
  }
  return 0;
}
// ---- Golomb-Rice ordered packer/unpacker (zigzag MErrval, per-group k) ----
static inline uint32_t zzenc(int r){ return r>=0?(uint32_t)(2*r):(uint32_t)(-2*r-1); }
size_t golomb_pack(const int16_t *syms, const uint8_t *keys, const uint8_t *ks, int N, uint8_t *out, size_t outcap){
  size_t bytepos=0; uint64_t buf=0; int nbits=0;
  for(int n=0;n<N;n++){
    uint32_t u=zzenc(syms[n]); int k=ks[keys[n]];
    uint32_t q=u>>k;
    // unary q ones + zero: emit via loop (q can be large; chunk it)
    while(q>=64){ for(int i=0;i<64;i++){ buf=(buf<<1)|1; if(++nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0);} } q-=64; }
    for(uint32_t i=0;i<q;i++){ buf=(buf<<1)|1; if(++nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0);} }
    buf=(buf<<1); if(++nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0);}
    if(k){ uint32_t rem=u&((k>=32)?0xFFFFFFFFu:((1u<<k)-1)); buf=(buf<<k)|rem; nbits+=k;
      while(nbits>=8){ nbits-=8; if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf>>nbits); buf&=((nbits)?((1ULL<<nbits)-1):0);} }
  }
  if(nbits){ if(bytepos>=outcap) return 0; out[bytepos++]=(uint8_t)(buf<<(8-nbits)); }
  return bytepos;
}
// test hook: evaluate expert x on given neighbors (for cross-checks)
int pred_probe(int x,int a,int b,int c,int d,int Ww,int NNe,int NE){ return pred_expert(x,a,b,c,d,Ww,NNe,NE); }
// Golomb streaming unpack for ordered (sym,key) stream with per-group k
int golomb_unpack(const uint8_t *in_, size_t inlen, const uint8_t *keys, const uint8_t *ks, int N, int16_t *res){
  size_t bitpos=0; size_t nbits=inlen*8;
  for(int n=0;n<N;n++){
    int k=ks[keys[n]];
    // unary quotient
    uint32_t q=0;
    for(;;){ if(bitpos>=nbits) return -1;
      int bit=(in_[bitpos>>3]>>(7-(bitpos&7)))&1; bitpos++;
      if(!bit) break; q++; if(q>(1u<<20)) return -1; }
    uint32_t rem=0;
    for(int i=0;i<k;i++){ if(bitpos>=nbits) return -1;
      rem=(rem<<1)|((in_[bitpos>>3]>>(7-(bitpos&7)))&1); bitpos++; }
    uint32_t u=(q<<k)|rem;
    res[n]=(int16_t)((u&1)?-((int)(u>>1))-1:(int)(u>>1));
  }
  return 0;
}
// ---- CROWN2 assembly: per-group pre-decoded symbols + E16 preds + signs (b4 nbhd rules) ----
static inline void b4_nbhd(const int16_t *P,int H,int W,int i,int j,
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
  if(i==0){ D=A; /* NE=D below */ }
  *a=A;*b=B;*c=C;*d=D;*Ww=WW;*NNe=NNE;*NE=D;
}
int crown2_assemble(const int16_t *syms, const int *goff, int ngroups,
  const uint16_t *cmap, const uint8_t *predid, int16_t *img, int H, int W, int ch) {
  int cur[128]; if(ngroups>128) return -2;
  for(int g=0;g<ngroups;g++) cur[g]=goff[g];
  int16_t *P=img+ch*H*W;
  for(int i=0;i<H;i++) for(int j=0;j<W;j++){
    int a,b,c,d,Ww,NNe,NE; b4_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
    int q1=loco_q1(b-c),q2=loco_q1(c-a),q3=loco_q1(d-b), sg=1;
    if(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&q3<0)){sg=-1;q1=-q1;q2=-q2;q3=-q3;}
    int g=cmap[q1*81+q2*9+q3];
    if(g<0||g>=ngroups) return -1;
    if(cur[g]>=goff[g+1]) return -1;
    int sym=syms[cur[g]++];
    P[i*W+j]=(int16_t)(pred_expert(predid[g],a,b,c,d,Ww,NNe,NE)+sg*sym);
  }
  return 0;
}
// E16 residuals with b4 rules (for encoder-side, matches probe_b4_b nbhd)
void e16_residuals(const int16_t *img,int H,int W,int16_t *res16){
  for(int k=0;k<3;k++){
    const int16_t *P=img+k*H*W;
    for(int x=0;x<16;x++){
      int16_t *R=res16+(k*16+x)*H*W;
      for(int i=0;i<H;i++) for(int j=0;j<W;j++){
        int a,b,c,d,Ww,NNe,NE; b4_nbhd(P,H,W,i,j,&a,&b,&c,&d,&Ww,&NNe,&NE);
        R[i*W+j]=(int16_t)((int)P[i*W+j]-pred_expert(x,a,b,c,d,Ww,NNe,NE));
      }
    }
  }
}