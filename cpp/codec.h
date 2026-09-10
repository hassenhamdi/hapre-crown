// CROWN6 C++ port — shared declarations (NEW file; existing sources untouched).
//
// Bit-identity contract: every decision here mirrors src/driver_crown6.py
// (with probe modules probe_b5_c/d, probe_b17_rctw, probe_b18_stack) exactly:
//   - floor division on negatives (fdiv), never truncating /
//   - border doctrines: b4-style for E16/keys/GRID, B17-style for WAVG/LMS taps,
//     zero-border causal taps for MLP features
//   - canonical Huffman via line-for-line port of CPython heapq semantics
//   - numpy linear-quantile formula incl. the t>=0.5 mirrored lerp branch
//   - stable sorts where Python is stable; strict < where Python is strict
// Loud asserts (die()) on any alphabet/framing violation; never silent clips.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define DIE(...) do { std::fprintf(stderr, "FATAL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::abort(); } while (0)
#define CHECK(cond, ...) do { if (!(cond)) DIE(__VA_ARGS__); } while (0)

namespace crown {

// ---- constants (FORMAT + driver verbatim) ----
inline constexpr int RES_MAX = 1024;
inline constexpr int GS = 32;
inline constexpr int WIDE_ARR[11] = {2,3,4,6,9,12,18,27,36,48,64};
inline constexpr int NWIDE = 11;
inline constexpr int MLP_NCTX = 9, MLP_FEAT = 12, MLP_H = 8, MLP_NPAR = 113;
inline constexpr int MLP_SCALES[5] = {256,512,1024,2048,4096};
// E20 order (driver_crown6.py): E16 + WAVG + LMS5_T0 + LMS5_T3 + MLP
inline constexpr int EX_MED=0, EX_TOP=1, EX_LEFT=2, EX_PAETH=3, EX_GRAD=4,
    EX_GAP80=5, EX_DG=6, EX_AVG_AB=7, EX_PLANE=8, EX_GAP32=9, EX_AC=10,
    EX_C=11, EX_BC=12, EX_D=13, EX_GAP16=14, EX_AVG3=15,
    EX_WAVG=16, EX_LMS0=17, EX_LMS3=18, EX_MLP=19;
inline constexpr int GRID_TH[8][5] = {
    {4,12,28,60,120},{2,6,16,40,100},{8,24,64,160,400},{1,3,8,24,80},
    {3,9,20,48,110},{6,18,44,100,220},{1,2,5,16,48},{5,15,36,80,160}};
inline constexpr int PERMS[6][3] = {{0,1,2},{1,2,0},{2,0,1},{0,2,1},{1,0,2},{2,1,0}};

inline int fdiv(int a, int b) { int q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) { q--; } return q; }

// ---- linked C ops (READ-ONLY existing objects; declared here, defined there) ----
extern "C" {
size_t pack_syms(const int16_t *syms, const uint8_t *keys, int N,
                 const uint32_t *codes, const uint8_t *lens,
                 uint8_t *out, size_t outcap);
void rans_norm(const int64_t *hist, int A, const int *syms,
               uint16_t *freq, int *cum);
size_t rans_encode(const int16_t *syms_in, int N, const int *lut,
                   const uint16_t *freq, const int *cum, uint8_t *out);
void rans_slots(const uint16_t *freq, const int *cum, int A, int *slot2sym);
int rans_decode(const uint8_t *in_, size_t inlen, int N, const int *syms,
                const uint16_t *freq, const int *cum, int A,
                const int *slot2sym, int16_t *out);
size_t crown6_golomb_pack(const int16_t *syms, const uint8_t *keys,
                          const uint8_t *kbygrp, int N, uint8_t *out,
                          size_t outcap);
int crown6_decode_ch(const uint8_t *hbuf, size_t hlen, const uint8_t *gbuf,
                     size_t glen, const int16_t *rsyms, const int *goff, int nX,
                     const uint32_t *cd, const uint8_t *ln, const uint8_t *cmap,
                     const uint8_t *predid, const uint8_t *backend,
                     const uint8_t *kvals, const int8_t *dbias,
                     const uint8_t *hasn, const uint8_t *wuse, const int8_t *ww,
                     int nbw, int family, const int *grid_thr, int16_t *plane,
                     int H, int W, int mlp_present, const double *mlp_thr,
                     const uint8_t *mlp_win, const double *mlp_w);
int crown6_rct_inv(const int16_t *p0, const int16_t *p1, const int16_t *p2,
                   int H, int W, int rct, uint8_t *rgb);
}

// ---- image ----
struct Img { int H = 0, W = 0; std::vector<int32_t> p[3]; };
inline size_t Npx(const Img &im) { return (size_t)im.H * im.W; }

// ---- bit writer (driver BitWriter.put verbatim: MSB-first) ----
struct BitWriter {
    uint64_t acc = 0; int nb = 0; std::vector<uint8_t> out;
    void put(uint32_t v, int n) {
        acc = (acc << n) | (v & (n >= 32 ? 0xFFFFFFFFu : ((n == 0) ? 0u : ((1u << n) - 1u))));
        nb += n;
        while (nb >= 8) { nb -= 8; out.push_back((uint8_t)((acc >> nb) & 0xFF));
            acc &= nb ? ((1ULL << nb) - 1) : 0; }
    }
    void flush() { if (nb) { out.push_back((uint8_t)((acc << (8 - nb)) & 0xFF)); acc = 0; nb = 0; } }
};

// ---- bit reader (decoder MSB-first) ----
struct BitReader {
    const uint8_t *d; size_t n; size_t qp = 0; uint64_t acc = 0; int nb = 0;
    BitReader(const uint8_t *d_, size_t n_) : d(d_), n(n_) {}
    uint32_t get(int k) {
        while (nb < k) {
            CHECK(qp < n, "bitstream overrun (need %d bits)", k);
            acc = (acc << 8) | d[qp++]; nb += 8;
        }
        nb -= k; uint32_t v = (uint32_t)((acc >> nb) & (k >= 32 ? 0xFFFFFFFFu : ((1u << k) - 1u)));
        acc &= nb ? ((1ULL << nb) - 1) : 0; return v;
    }
};

void check_range_vec(const int32_t *a, size_t n, const char *where);
inline void check_range_plane(const std::vector<int32_t> &v, const char *w) {
    check_range_vec(v.data(), v.size(), w);
}

// ---- experts / neighborhoods ----
void nbhd_b4(const int32_t *P, int H, int W, int i, int j,
             int &a, int &b, int &c, int &d, int &Ww, int &NNe, int &NE);
int pred_e16(int x, int a, int b, int c, int d, int Ww, int NNe, int NE);
inline int loco_q1(int g) { int a = g < 0 ? -g : g; if (g == 0) { return 0; }
    if (a <= 2) { return g < 0 ? -1 : 1; } if (a <= 7) { return g < 0 ? -2 : 2; }
    if (a <= 21) { return g < 0 ? -3 : 3; } return g < 0 ? -4 : 4; }
// full E16 prediction planes + LOCO key/sign (prepX equivalent)
struct PrepX { std::vector<int32_t> P[16]; std::vector<int> key; std::vector<int8_t> s;
    int32_t a0,b0,c0; // unused padding guard
    std::vector<int32_t> av, bv, cv, dv; };
PrepX prep_x(const std::vector<int32_t> &ch, int H, int W);

// ---- RCT ----
void rct_fwd(const uint8_t *rgb, int H, int W, int perm, int t, Img &yc /*planes filled*/);
void rct_inv_planes(const int32_t *p0, const int32_t *p1, const int32_t *p2,
                    int H, int W, int perm, int t, uint8_t *rgb);

// ---- B17 MED / nbhd4 / weighted fit ----
void med_pred_plane(const int32_t *P, int H, int W, std::vector<int32_t> &out);
struct WFit { std::vector<int32_t> res; std::vector<uint8_t> use; // ng
    std::vector<int8_t> w; /* ng*4 */ int ng = 0, nbh = 0, nbw = 0; };
WFit weighted_fit_ch(const int32_t *plane, int H, int W);
int wpred_from_w(const int8_t *w, int L, int T, int TL, int TR);

// ---- LMS ----
void lms_pred_plane(const int32_t *plane, int H, int W, int thr,
                    std::vector<int32_t> &out);

// ---- MLP causal features / contexts / wire forward ----
void mlp_causal_taps(const int32_t *P, int H, int W, int i, int j, int fx[12]);
void mlp_energy_ctx(const int32_t *P, int H, int W,
                    std::vector<double> &qs /*8*/, std::vector<int> &ctx /*N*/,
                    std::vector<int> &E /*N*/);
int mlp_fwd_wire(const int fx[12], const double *W1 /*96*/,
                 const double *b1 /*8*/, const double *W2 /*8*/, double b2);

// ---- numpy-linear quantile (exact replica incl. mirrored lerp) ----
double quantile_linear(const std::vector<double> &sorted_vals, double q);

// ---- Huffman via exact CPython-heapq port ----
int64_t huff_bits_counts(const std::vector<int64_t> &counts); // probe_b4_a.huff_bits
// canon_tables: hist[2049] -> lens[2049] + codes[2049] (single-symbol => len 1, code 0)
void canon_tables(const int64_t hist[2049], uint8_t lens[2049], uint32_t codes[2049]);

// ---- Golomb ----
void golomb_best(const int32_t *g, size_t n, int64_t &tot, int &k); // zigzag cost
struct GolombCost { int64_t tot; int k, d; int dbits; };
GolombCost golomb_cost(const int32_t *g, size_t n);

// ---- rANS cost (links libhapre; asserts roundtrip) ----
struct RansBlob { std::vector<uint8_t> pay; std::vector<int32_t> syms;
    std::vector<uint16_t> freq; std::vector<int32_t> cum; };
int64_t rans_cost(const int16_t *g, size_t n, RansBlob &blob); // bits; blob valid iff n>0

// ---- .npz (stored zip) loader ----
struct NpzNet { double qs[8]; uint8_t win[9]; uint8_t scid[9]; int16_t qw[9][113]; };
bool npz_load_ch(const std::string &path, const std::string &rct, int ch, NpzNet &nt);

// ---- PNG via vendored stb_image ----
bool load_png_rgb(const std::string &path, std::vector<uint8_t> &rgb, int &H, int &W);

} // namespace crown