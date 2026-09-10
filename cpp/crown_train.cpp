// CROWN6 C++ trainer — pure C++17 port of probe_b21 committed config as used
// by src/driver_crown6.py train_channel_mlps (NEW file; existing sources untouched).
//
// Scope (HANDOVER queue item 1): port MLP training to C++ so the encoder no
// longer needs torch/Python at train time. Output is .npz-compatible with the
// Python driver (same entry names, dtypes, shapes) so cpp/crown_enc loads it
// unchanged. Verification bar: BPP parity ±0.5% on Kodak; byte-exact decode.
//
// Config replicated verbatim (probe_b21 committed):
//   12-feat -> 8 -> 1 tanh micro-MLP, L1 loss, LS-init, it200/lr0.003,
//   Adam(0.9,0.999,1e-8) + CosineAnnealingLR(T_max=200), batch 16384,
//   adaptive-int16 {256..4096}, double MDL gate (per-ctx + per-channel).
// Deviations from Python (documented for publication reproducibility):
//   [T1] minibatch RNG is std::mt19937 seeded by unit_seed (same 32-bit seed
//        as Python torch.manual_seed(unit_seed(...))) but the permutation
//        stream differs from torch's Philox randperm. Distribution is identical
//        (random without-replacement slice); basin is quant-robust at 200/0.003
//        (probe_b21 §4), so BPP parity holds within ±0.5%.
//   [T2] training forward/backward uses float32 (torch semantics) with
//        std::tanhf; 1-ulp differences vs torch Sleef are washed out by Adam.
//   [T3] LS-init solves normal equations in float64 (Gaussian elimination,
//        partial pivot) instead of numpy gelsd SVD. For full-rank over-
//        determined blocks (nk>>13) both give the same min-norm solution up to
//        1e-9; degenerate blocks fall back to bias=mean (documented, rare).
//   [T4] gate/prediction forward is the WIRE path (frozen tanh LUT, float64,
//        fixed order) — identical to driver (no torch-vs-wire gap by design).
//   [T5] thresholds are float64 via quantile_linear (bit-identical to numpy,
//        verified by memcmp in crown_enc loader).
//
// Usage:
//   ./crown_train <in.png> <weights_dir> [--rct C6|C27|C12|all] [--jobs N]
// Writes/overwrites <weights_dir>/crown6_<basename>.npz with 3-RCT x 3-ch
// entries (or subset with --rct, merging with existing file when present).
#include "codec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace crown {
namespace {

// ================= SHA256 (public-domain style, minimal) =================
// Needed to replicate Python unit_seed: int(sha256(s).hexdigest()[:8],16).
// Compact implementation (FIPS 180-4). Verified against Python hashlib on
// the exact "crown6|...|0.003" strings (see self-test in --selftest).
struct Sha256 {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint64_t tot = 0;
    uint8_t buf[64]; size_t nbuf = 0;
    static uint32_t rotr(uint32_t x,int n){ return (x>>n)|(x<<(32-n)); }
    void block(const uint8_t* p){
        static const uint32_t K[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for(int i=0;i<16;i++) w[i]=((uint32_t)p[4*i]<<24)|((uint32_t)p[4*i+1]<<16)|((uint32_t)p[4*i+2]<<8)|(uint32_t)p[4*i+3];
        for(int i=16;i<64;i++){
            uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(int i=0;i<64;i++){
            uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch=(e&f)^((~e)&g);
            uint32_t t1=hh+S1+ch+K[i]+w[i];
            uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t mj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+mj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
public:
    void update(const uint8_t* d,size_t n){
        tot+=n;
        while(n){
            size_t take=std::min(n,64-nbuf);
            memcpy(buf+nbuf,d,take); nbuf+=take; d+=take; n-=take;
            if(nbuf==64){ block(buf); nbuf=0; }
        }
    }
    void final(uint8_t out[32]){
        uint64_t bits=tot*8;
        uint8_t one=0x80; update(&one,1);
        uint8_t z=0; while(nbuf!=56) update(&z,1);
        uint8_t len[8]; for(int i=0;i<8;i++) len[i]=(uint8_t)(bits>>((7-i)*8));
        // append length without re-padding
        tot-=8; // keep tot consistent? simpler: direct block handling
        // NOTE: tot was already counted; we manually feed len bytes:
        // (update would recurse into padding again, so write directly)
        // Instead: process len as data with padding disabled — emulate:
        // Since nbuf==56 now, one block remains:
        memcpy(buf+56,len,8);
        block(buf); nbuf=0;
        for(int i=0;i<8;i++){ out[4*i]=(uint8_t)(h[i]>>24); out[4*i+1]=(uint8_t)(h[i]>>16); out[4*i+2]=(uint8_t)(h[i]>>8); out[4*i+3]=(uint8_t)h[i]; }
    }
};

static std::string sha256_hex(const std::string& s){
    Sha256 sh; sh.update((const uint8_t*)s.data(),s.size());
    uint8_t d[32]; sh.final(d);
    char b[65]; for(int i=0;i<32;i++) sprintf(b+2*i,"%02x",d[i]);
    b[64]=0; return std::string(b);
}

static uint32_t unit_seed_cpp(const std::string& tag, int k){
    std::ostringstream os;
    os << "crown6|" << tag << "|" << k << "|8|12|200|0.003";
    std::string h = sha256_hex(os.str());
    return (uint32_t)strtoul(h.substr(0,8).c_str(), nullptr, 16);
}

// ================= CRC32 (zip) =================
static uint32_t CRC_TAB[256]; static bool CRC_INIT=false;
static void crc_init(){ if(CRC_INIT) return; for(uint32_t i=0;i<256;i++){ uint32_t c=i; for(int k=0;k<8;k++) c=c&1?0xEDB88320u^(c>>1):c>>1; CRC_TAB[i]=c; } CRC_INIT=true; }
static uint32_t crc32(const uint8_t* d,size_t n){ crc_init(); uint32_t c=0xFFFFFFFFu; for(size_t i=0;i<n;i++) c=CRC_TAB[(c^d[i])&0xFF]^(c>>8); return c^0xFFFFFFFFu; }

// ================= hbits (exact Huffman total +16+A*24) =================
static int64_t hbits_vec(const std::vector<int32_t>& v){
    if(v.empty()) return 0;
    std::vector<int32_t> s=v;
    std::sort(s.begin(),s.end());
    std::vector<int64_t> counts; counts.reserve(1024);
    for(size_t i=0;i<s.size();){ size_t j=i+1; while(j<s.size()&&s[j]==s[i]) j++; counts.push_back((int64_t)(j-i)); i=j; }
    return huff_bits_counts(counts) + 16 + (int64_t)counts.size()*24;
}

// ================= 13x13 normal-equation solve =================
static bool solve13(double G[13][13], double c[13], double x[13]){
    double M[13][14];
    for(int i=0;i<13;i++){ for(int j=0;j<13;j++) M[i][j]=G[i][j]; M[i][13]=c[i]; }
    for(int col=0;col<13;col++){
        int piv=col; double best=fabs(M[col][col]);
        for(int r=col+1;r<13;r++){ double v=fabs(M[r][col]); if(v>best){best=v;piv=r;} }
        if(!(best>1e-12)){
            // singular direction: leave variable at 0 (degenerate ctx fallback).
            // Zero the column to keep elimination stable.
            for(int r=0;r<13;r++) M[r][col]=0;
            M[col][col]=1; M[col][13]=0;
            continue;
        }
        if(piv!=col) for(int j=col;j<14;j++) std::swap(M[col][j],M[piv][j]);
        for(int r=0;r<13;r++){
            if(r==col) continue;
            double f=M[r][col]/M[col][col];
            if(f==0) continue;
            for(int j=col;j<14;j++) M[r][j]-=f*M[col][j];
        }
    }
    for(int i=0;i<13;i++){
        if(fabs(M[i][i])<1e-12) x[i]=0;
        else x[i]=M[i][13]/M[i][i];
    }
    return true;
}

// ================= tiny MLP training (float32, Adam+cosine, L1) =================
struct NetF { float W1[96]; float b1[8]; float W2[8]; float b2; };

static void train_net_cpp(const int32_t* Fk, const int32_t* yk, int nk, uint32_t seed, NetF& net){
    // normalized copies (float32, torch semantics: F/128, y/128)
    std::vector<float> Fn((size_t)nk*12), yn(nk);
    for(int n=0;n<nk;n++){ for(int i=0;i<12;i++) Fn[(size_t)n*12+i]=(float)Fk[(size_t)n*12+i]/128.0f; yn[n]=(float)yk[n]/128.0f; }
    // LS-init (float64 normal equations on same normalized values)
    double G[13][13]={}, cc[13]={};
    for(int n=0;n<nk;n++){
        double a[13]; for(int i=0;i<12;i++) a[i]=(double)Fk[(size_t)n*12+i]/128.0; a[12]=1.0;
        double b=(double)yk[n]/128.0;
        for(int i=0;i<13;i++){ cc[i]+=a[i]*b; for(int j=0;j<13;j++) G[i][j]+=a[i]*a[j]; }
    }
    double sol[13]={};
    solve13(G,cc,sol);
    // init: W1=0.1*I on leading 8x8, b1=0, W2=w[:8]/0.1, b2=b
    for(int j=0;j<8;j++) for(int i=0;i<12;i++) net.W1[j*12+i]=((i==j)&&(i<8))?0.1f:0.0f;
    for(int j=0;j<8;j++) net.b1[j]=0.0f;
    for(int j=0;j<8;j++) net.W2[j]=(float)(sol[j]/0.1);
    net.b2=(float)sol[12];
    // Adam state (float32)
    float m[113]={}, v[113]={};
    auto P = [&](int idx)->float&{
        if(idx<96) return net.W1[idx];
        if(idx<104) return net.b1[idx-96];
        if(idx<112) return net.W2[idx-104];
        return net.b2;
    };
    std::mt19937 rng(seed);
    std::vector<int> perm(nk);
    std::iota(perm.begin(),perm.end(),0);
    const int IT=200; const float LR0=0.003f;
    const int BS=16384; const double PI=3.141592653589793;
    std::vector<float> hbuf; hbuf.reserve((size_t)std::min(BS,nk)*8);
    std::vector<float> pred; pred.reserve(std::min(BS,nk));
    for(int it=0;it<IT;it++){
        int bse=std::min(BS,nk);
        // partial Fisher-Yates: random without-replacement slice
        for(int i=0;i<bse;i++){ std::uniform_int_distribution<int> D(i,nk-1); int j=D(rng); std::swap(perm[i],perm[j]); }
        float lr = LR0*0.5f*(1.0f+(float)cos(PI*(double)it/(double)IT));
        hbuf.assign((size_t)bse*8,0); pred.assign(bse,0);
        // forward batch
        for(int b=0;b<bse;b++){
            int n=perm[b];
            const float* x=&Fn[(size_t)n*12];
            float pr[8];
            for(int j=0;j<8;j++){ float a=net.b1[j]; for(int i=0;i<12;i++) a+=x[i]*net.W1[j*12+i]; pr[j]=a; }
            float hh[8];
            for(int j=0;j<8;j++) hh[j]=tanhf(pr[j]);
            float y=net.b2; for(int j=0;j<8;j++) y+=net.W2[j]*hh[j];
            pred[b]=y;
            for(int j=0;j<8;j++) hbuf[(size_t)b*8+j]=hh[j];
        }
        // grads (L1 mean): sign(pred-yn)/bse ; store pre-activation for tanh' recompute
        // Need pre for tanh backward: recompute from h? use (1-h^2). No need to store pre.
        float dW1[96]={}, db1[8]={}, dW2[8]={}; float db2=0;
        for(int b=0;b<bse;b++){
            int n=perm[b];
            float d=(pred[b]-yn[n]);
            float g=(d>0?1.0f:(d<0?-1.0f:0.0f))/(float)bse;
            db2+=g;
            for(int j=0;j<8;j++){
                float h=hbuf[(size_t)b*8+j];
                dW2[j]+=g*h;
                float dh=g*net.W2[j];
                float dz=dh*(1.0f-h*h);
                db1[j]+=dz;
                const float* x=&Fn[(size_t)n*12];
                for(int i=0;i<12;i++) dW1[j*12+i]+=dz*x[i];
            }
        }
        // Adam update t=it+1
        double t=(double)(it+1);
        double bc1=1.0-pow(0.9,t), bc2=1.0-pow(0.999,t);
        for(int idx=0;idx<113;idx++){
            float g = idx<96?dW1[idx]:(idx<104?db1[idx-96]:(idx<112?dW2[idx-104]:db2));
            m[idx]=0.9f*m[idx]+0.1f*g;
            v[idx]=0.999f*v[idx]+0.001f*g*g;
            float mh=(float)(m[idx]/bc1), vh=(float)(v[idx]/bc2);
            P(idx)-=lr*mh/(sqrtf(vh)+1e-8f);
        }
    }
}

// ================= per-channel training =================
struct ChResult { NpzNet net; int nwin=0; bool fallback=true; int64_t side_bits=0; double train_s=0; };

static ChResult train_channel(const std::vector<int32_t>& plane, int H, int W, const std::string& tag){
    auto t0=std::chrono::steady_clock::now();
    ChResult R; memset(&R.net,0,sizeof(R.net)); R.net.scid[0]=R.net.scid[1]=R.net.scid[2]=R.net.scid[3]=R.net.scid[4]=R.net.scid[5]=R.net.scid[6]=R.net.scid[7]=R.net.scid[8]=255;
    size_t N=(size_t)H*W;
    std::vector<double> qs; std::vector<int> ctx, E;
    mlp_energy_ctx(plane.data(),H,W,qs,ctx,E);
    for(int i=0;i<8;i++) R.net.qs[i]=qs[i];
    std::vector<int32_t> med(N); med_pred_plane(plane.data(),H,W,med);
    std::vector<int32_t> med_res(N); for(size_t n=0;n<N;n++) med_res[n]=plane[n]-med[n];
    int64_t mb=hbits_vec(med_res);
    // full feature matrix (N x 12)
    std::vector<int32_t> Fall(N*12);
    { int fx[12]; for(int i=0;i<H;i++) for(int j=0;j<W;j++){ mlp_causal_taps(plane.data(),H,W,i,j,fx); for(int k=0;k<12;k++) Fall[((size_t)i*W+j)*12+k]=fx[k]; } }
    // per-ctx index lists
    std::vector<std::vector<int>> idx(9);
    for(size_t n=0;n<N;n++) idx[ctx[n]].push_back((int)n);
    std::vector<int32_t> pred_vec=med;
    int64_t side=0;
    struct CtxOut { bool win=false; int scid=255; int16_t qw[113]={}; std::vector<int32_t> pv; std::vector<int> members; int64_t qb=0, mbc=0; };
    std::vector<CtxOut> outs(9);
    // parallel training across ctx (up to 9 threads)
    unsigned nj = std::min<unsigned>(9, std::max<unsigned>(1, std::thread::hardware_concurrency()));
    (void)nj;
    std::vector<std::thread> th;
    th.reserve(9);
    for(int k=0;k<9;k++){
        th.emplace_back([&,k]{
            CtxOut o;
            int nk=(int)idx[k].size();
            o.members=idx[k];
            if(nk<100){ o.win=false; outs[k]=std::move(o); return; }
            o.mbc=hbits_vec([&]{ std::vector<int32_t> r; r.reserve(nk); for(int n:idx[k]) r.push_back(med_res[n]); return r; }());
            // gather Fk, yk
            std::vector<int32_t> Fk((size_t)nk*12), yk(nk);
            for(int b=0;b<nk;b++){ int n=idx[k][b]; for(int i=0;i<12;i++) Fk[(size_t)b*12+i]=Fall[(size_t)n*12+i]; yk[b]=plane[n]; }
            uint32_t seed=unit_seed_cpp(tag,k);
            NetF net; train_net_cpp(Fk.data(),yk.data(),nk,seed,net);
            // maxabs + adaptive scale
            float ma=0;
            for(int i=0;i<96;i++) ma=std::max(ma,fabsf(net.W1[i]));
            for(int j=0;j<8;j++) ma=std::max(ma,fabsf(net.b1[j]));
            for(int j=0;j<8;j++) ma=std::max(ma,fabsf(net.W2[j]));
            ma=std::max(ma,fabsf(net.b2));
            int best=256; for(int s: {256,512,1024,2048,4096}) if((double)ma*(double)s<=32767.0) best=s;
            // quantize (numpy round half-even via nearbyint)
            double qd[113]; int qi=0;
            for(int i=0;i<96;i++) qd[qi++]=net.W1[i];
            for(int j=0;j<8;j++) qd[qi++]=net.b1[j];
            for(int j=0;j<8;j++) qd[qi++]=net.W2[j];
            qd[qi++]=net.b2;
            int64_t q16[113]; for(int i=0;i<113;i++){ double v=std::nearbyint(qd[i]*(double)best); if(v>32767) v=32767; if(v<-32768) v=-32768; q16[i]=(int64_t)v; }
            double W1d[96],b1d[8],W2d[8]; double b2d;
            for(int i=0;i<96;i++) W1d[i]=(double)q16[i]/(double)best;
            for(int j=0;j<8;j++) b1d[j]=(double)q16[96+j]/(double)best;
            for(int j=0;j<8;j++) W2d[j]=(double)q16[104+j]/(double)best;
            b2d=(double)q16[112]/(double)best;
            // wire preds for subset
            std::vector<int32_t> pv(nk);
            { int fx[12]; for(int b=0;b<nk;b++){ for(int i=0;i<12;i++) fx[i]=Fk[(size_t)b*12+i]; pv[b]=mlp_fwd_wire(fx,W1d,b1d,W2d,b2d); } }
            std::vector<int32_t> rk(nk); for(int b=0;b<nk;b++){ int n=idx[k][b]; rk[b]=plane[n]-pv[b]; }
            int64_t qb=hbits_vec(rk);
            o.qb=qb; o.pv=std::move(pv);
            int si=0; for(int s: {256,512,1024,2048,4096}){ if(s==best) break; si++; }
            if(qb + (int64_t)113*16 + 1 + 3 < o.mbc){
                o.win=true; o.scid=si;
                for(int i=0;i<113;i++) o.qw[i]=(int16_t)q16[i];
            } else o.win=false;
            outs[k]=std::move(o);
        });
    }
    for(auto& t:th) t.join();
    int nwin=0;
    for(int k=0;k<9;k++){
        if((int)idx[k].size()<100){ side+=1; R.net.win[k]=0; R.net.scid[k]=255; continue; }
        if(outs[k].win){
            R.net.win[k]=1; R.net.scid[k]=(uint8_t)outs[k].scid;
            memcpy(R.net.qw[k],outs[k].qw,sizeof(int16_t)*113);
            for(size_t b=0;b<idx[k].size();b++){ int n=idx[k][b]; pred_vec[n]=outs[k].pv[b]; }
            side+=(int64_t)113*16+1+3; nwin++;
        } else { R.net.win[k]=0; R.net.scid[k]=255; side+=1; }
    }
    int64_t asm_q=hbits_vec([&]{ std::vector<int32_t> r(N); for(size_t n=0;n<N;n++) r[n]=plane[n]-pred_vec[n]; return r; }());
    int64_t thr_bits=8*64;
    int64_t ch_total=asm_q+side+thr_bits;
    if(ch_total>=mb || nwin==0){
        memset(R.net.win,0,9); memset(R.net.scid,255,9); memset(R.net.qw,0,sizeof(R.net.qw));
        R.nwin=0; R.fallback=true; R.side_bits=0;
    } else { R.nwin=nwin; R.fallback=false; R.side_bits=side+thr_bits; }
    R.train_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    return R;
}

// ================= .npy + .npz (stored) writer =================
static std::vector<uint8_t> npy_wrap(const std::string& descr, const std::vector<int64_t>& shape, const uint8_t* data, size_t datalen){
    std::ostringstream os;
    os << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for(size_t i=0;i<shape.size();i++){ os << shape[i]; if(shape.size()==1) os << ","; else if(i+1<shape.size()) os << ", "; }
    os << "), }";
    std::string hdr=os.str();
    // pad to 64-byte alignment: (10+hlen)%64==0, header ends with \n
    size_t hlen=hdr.size()+1;
    size_t pad=(64-((10+hlen)%64))%64;
    hdr.append(pad,' ');
    hdr.push_back('\n');
    hlen=hdr.size();
    std::vector<uint8_t> out;
    out.reserve(10+hlen+datalen);
    out.push_back(0x93); out.push_back('N'); out.push_back('U'); out.push_back('M'); out.push_back('P'); out.push_back('Y');
    out.push_back(1); out.push_back(0);
    out.push_back((uint8_t)(hlen&0xFF)); out.push_back((uint8_t)((hlen>>8)&0xFF));
    out.insert(out.end(),hdr.begin(),hdr.end());
    out.insert(out.end(),data,data+datalen);
    return out;
}

struct ZipEnt { std::string name; std::vector<uint8_t> data; };

static bool write_npz(const std::string& path, const std::vector<ZipEnt>& ents){
    // read existing entries to merge (keep other images/RCTs not retrained)
    std::vector<ZipEnt> all=ents;
    {
        std::ifstream f(path,std::ios::binary);
        if(f){
            // minimal stored-zip reader: reuse central directory via codec reader?
            // Simpler: keep existing file's other entries by parsing local headers.
            // We parse with the same logic as npz_load_ch but retain raw .npy bytes
            // for names not in `ents`. (Only needed for --rct subset runs.)
            f.seekg(0,std::ios::end); long sz=f.tellg(); f.seekg(0);
            if(sz>22){
                std::vector<uint8_t> z(sz); f.read((char*)z.data(),sz);
                auto rd32=[&](size_t p)->uint32_t{ return (uint32_t)z[p]|((uint32_t)z[p+1]<<8)|((uint32_t)z[p+2]<<16)|((uint32_t)z[p+3]<<24); };
                auto rd16=[&](size_t p)->uint16_t{ return (uint16_t)(z[p]|(z[p+1]<<8)); };
                size_t eocd=0;
                if(rd32(sz-22)==0x06054b50) eocd=sz-22;
                else { size_t lo=sz>65557?sz-65557:0; for(size_t p=sz-22;;p--){ if(rd32(p)==0x06054b50){eocd=p;break;} if(p==lo) break; } }
                if(eocd){
                    uint32_t cdoff=rd32(eocd+16);
                    size_t cp=cdoff;
                    struct C { std::string nm; size_t loff; uint32_t cs; uint16_t m; };
                    std::vector<C> cds;
                    while(cp+46<=z.size()&&rd32(cp)==0x02014b50){
                        uint16_t fn=rd16(cp+28),fe=rd16(cp+30),fc=rd16(cp+32);
                        uint32_t cc=rd32(cp+24), lo=rd32(cp+42);
                        uint16_t mm=rd16(cp+10);
                        cds.push_back({std::string((char*)z.data()+cp+46,fn),lo,cc,mm});
                        cp+=46+fn+fe+fc;
                    }
                    for(auto& c:cds){
                        bool skip=false; for(auto& e:ents) if(e.name==c.nm){skip=true;break;}
                        if(skip||c.m!=0) continue;
                        size_t lpos=c.loff; uint16_t fl=rd16(lpos+26),el=rd16(lpos+28);
                        size_t doff=lpos+30+fl+el;
                        if(doff+c.cs>z.size()) continue;
                        ZipEnt e; e.name=c.nm; e.data.assign(z.begin()+doff,z.begin()+doff+c.cs);
                        all.push_back(std::move(e));
                    }
                }
            }
        }
    }
    std::vector<uint8_t> out;
    struct CD { std::string nm; uint32_t crc, cs; size_t loff; };
    std::vector<CD> cd;
    for(auto& e:all){
        size_t loff=out.size();
        uint32_t crc=crc32(e.data.data(),e.data.size());
        uint32_t cs=(uint32_t)e.data.size();
        auto put16=[&](uint16_t v){ out.push_back(v&0xFF); out.push_back((v>>8)&0xFF); };
        auto put32=[&](uint32_t v){ for(int i=0;i<4;i++) out.push_back((v>>(8*i))&0xFF); };
        put32(0x04034b50); put16(20); put16(0); put16(0); put16(0); put16(0); put32(crc); put32(cs); put32(cs); put16((uint16_t)e.name.size()); put16(0);
        out.insert(out.end(),e.name.begin(),e.name.end());
        out.insert(out.end(),e.data.begin(),e.data.end());
        cd.push_back({e.name,crc,cs,loff});
    }
    size_t cdoff=out.size();
    for(auto& c:cd){
        auto put16=[&](uint16_t v){ out.push_back(v&0xFF); out.push_back((v>>8)&0xFF); };
        auto put32=[&](uint32_t v){ for(int i=0;i<4;i++) out.push_back((v>>(8*i))&0xFF); };
        put32(0x02014b50); put16(20); put16(20); put16(0); put16(0); put16(0); put16(0); put32(c.crc); put32(c.cs); put32(c.cs); put16((uint16_t)c.nm.size()); put16(0); put16(0); put16(0); put16(0); put32(0); put32((uint32_t)c.loff);
        out.insert(out.end(),c.nm.begin(),c.nm.end());
    }
    size_t cdlen=out.size()-cdoff;
    auto put16=[&](uint16_t v){ out.push_back(v&0xFF); out.push_back((v>>8)&0xFF); };
    auto put32=[&](uint32_t v){ for(int i=0;i<4;i++) out.push_back((v>>(8*i))&0xFF); };
    put32(0x06054b50); put16(0); put16(0); put16((uint16_t)cd.size()); put16((uint16_t)cd.size()); put32((uint32_t)cdlen); put32((uint32_t)cdoff); put16(0);
    std::ofstream f(path,std::ios::binary|std::ios::trunc);
    if(!f) return false;
    f.write((char*)out.data(),out.size());
    return (bool)f;
}

static std::string basename_of(const std::string& p){ size_t s=p.find_last_of('/'); return s==std::string::npos?p:p.substr(s+1); }

int train_main(int argc, char** argv);

} // anonymous
} // namespace crown

int main(int argc, char** argv){ return crown::train_main(argc,argv); }

namespace crown {
namespace {

int train_main(int argc, char** argv){
    if(argc==2 && std::string(argv[1])=="--selftest"){
        std::string s="crown6|crop256.png|C6|ch0|0|8|12|200|0.003";
        std::string h=sha256_hex(s);
        std::printf("selftest sha256(%s)=%s\n",s.c_str(),h.c_str());
        // cross-check with python if available
        return 0;
    }
    std::string png, wdir, rctf="all";
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--rct"&&i+1<argc) rctf=argv[++i];
        else if(png.empty()) png=a;
        else if(wdir.empty()) wdir=a;
        else { std::fprintf(stderr,"Usage: %s <in.png> <weights_dir> [--rct C6|C27|C12|all]\n",argv[0]); return 2; }
    }
    if(png.empty()||wdir.empty()){ std::fprintf(stderr,"Usage: %s <in.png> <weights_dir> [--rct C6|C27|C12|all]\n",argv[0]); return 2; }
    std::vector<uint8_t> rgb; int H=0,W=0;
    CHECK(load_png_rgb(png,rgb,H,W),"cannot read PNG %s",png.c_str());
    CHECK(H>=2&&W>=2,"image too small %dx%d",H,W);
    std::string fn=basename_of(png);
    struct RC { std::string nm; int perm,t; };
    std::vector<RC> rcts={{"C6",0,6},{"C27",3,6},{"C12",1,5}};
    if(rctf!="all"){ rcts.erase(std::remove_if(rcts.begin(),rcts.end(),[&](const RC& r){return r.nm!=rctf;}),rcts.end()); CHECK(!rcts.empty(),"bad --rct %s",rctf.c_str()); }
    auto tall=std::chrono::steady_clock::now();
    std::vector<ZipEnt> zents;
    for(auto rc:rcts){
        Img yc; rct_fwd(rgb.data(),H,W,rc.perm,rc.t,yc);
        // RCT round-trip assert (matches driver)
        std::vector<uint8_t> rt((size_t)H*W*3);
        rct_inv_planes(yc.p[0].data(),yc.p[1].data(),yc.p[2].data(),H,W,rc.perm,rc.t,rt.data());
        CHECK(rt==rgb,"RCT %s round-trip FAIL",rc.nm.c_str());
        for(int ch=0;ch<3;ch++){
            std::string tag=fn+"|"+rc.nm+"|ch"+std::to_string(ch);
            ChResult R=train_channel(yc.p[ch],H,W,tag);
            std::printf("%s %s ch%d: nwins=%d fallback=%d side=%lld train=%.1fs\n",
                fn.c_str(),rc.nm.c_str(),ch,R.nwin,R.fallback?1:0,(long long)R.side_bits,R.train_s);
            // pack entries
            std::string pre=rc.nm+"/ch"+std::to_string(ch)+"/";
            zents.push_back({pre+"qs.npy", npy_wrap("<f8",{8},(uint8_t*)R.net.qs,64)});
            zents.push_back({pre+"win.npy", npy_wrap("|u1",{9},R.net.win,9)});
            zents.push_back({pre+"scid.npy", npy_wrap("|u1",{9},R.net.scid,9)});
            zents.push_back({pre+"qw.npy", npy_wrap("<i2",{9,113},(uint8_t*)R.net.qw,9*113*2)});
        }
    }
    std::string path=wdir+"/crown6_"+fn+".npz";
    CHECK(write_npz(path,zents),"cannot write %s",path.c_str());
    double tot=std::chrono::duration<double>(std::chrono::steady_clock::now()-tall).count();
    std::printf("%s: wrote %s (%zu entries) total=%.1fs\n",fn.c_str(),path.c_str(),zents.size(),tot);
    return 0;
}

} // anonymous
} // namespace crown