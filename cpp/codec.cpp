// CROWN6 C++ port — shared implementation (NEW file).
// Mirrors driver_crown6.py + probe_b5_c/d + probe_b17_rctw + probe_b18_stack
// decision-for-decision. See codec.h for the bit-identity contract.
#include "codec.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "../src/crown6_tanhlut.h"

namespace crown {

void check_range_vec(const int32_t *a, size_t n, const char *where) {
    int64_t m = 0;
    for (size_t i = 0; i < n; i++) { int64_t v = a[i] >= 0 ? (int64_t)a[i] : -(int64_t)a[i];
        if (v > m) m = v; }
    CHECK(m <= RES_MAX, "ALPHABET VIOLATION %s: |r|max=%lld > 1024 (abort, never clip)",
          where, (long long)m);
}

// ---- b4-style neighborhood, per-pixel mirror of c6_nbhd/probe_b4_b.nbhd ----
void nbhd_b4(const int32_t *P, [[maybe_unused]] int H, int W, int i, int j,
             int &a, int &b, int &c, int &d, int &Ww, int &NNe, int &NE) {
    int A = (j > 0) ? P[i * W + j - 1] : 0;
    int B = (i > 0) ? P[(i - 1) * W + j] : 0;
    int C = 0;
    if (i == 0 && j == 0) { A = 0; B = 0; C = 0; }
    else if (i == 0) { B = A; C = A; }
    else if (j == 0) { A = B; C = B; }
    else { C = P[(i - 1) * W + j - 1]; }
    int D = (j + 1 >= W) ? B : ((i == 0) ? A : P[(i - 1) * W + j + 1]);
    int WW = (j >= 2) ? P[i * W + j - 2] : A;
    int NNE = (i >= 2) ? P[(i - 2) * W + j] : B;
    a = A; b = B; c = C; d = D; Ww = WW; NNe = NNE; NE = D;
}

int pred_e16(int x, int a, int b, int c, int d, int Ww, int NNe, int NE) {
    switch (x) {
    case 0: { int mn = a < b ? a : b, mx = a < b ? b : a; // MED
        if (c >= mx) { return mn; } if (c <= mn) { return mx; } return a + b - c; }
    case 1: return b;                      // TOP
    case 2: return a;                      // LEFT
    case 3: { int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c); // PAETH
        if (pa <= pb && pa <= pc) return a; else if (pb <= pc) return b; else return c; }
    case 4: return fdiv(a + b, 2) + fdiv(b - c, 4); // GRAD
    case 5: case 9: case 14: { int th = x == 5 ? 80 : (x == 9 ? 32 : 16); // GAP80/32/16
        int gh = abs(a - Ww) + abs(b - c) + abs(b - NE);
        int gv = abs(a - c) + abs(b - NNe) + abs(NE - b);
        if (gv - gh > th) return a; else if (gv - gh < -th) return b;
        else return fdiv(a + b, 2) + fdiv(NE - c, 4); }
    case 6: return fdiv(c + d, 2);          // DG
    case 7: return fdiv(a + b, 2);          // AVG_AB
    case 8: return a + b - c;               // PLANE
    case 10: return fdiv(a + c, 2);         // AC
    case 11: return c;                      // C
    case 12: return fdiv(b + c, 2);         // BC
    case 13: return d;                      // D
    default: return fdiv(a + b + c, 3);     // AVG3
    }
}

PrepX prep_x(const std::vector<int32_t> &ch, int H, int W) {
    PrepX D;
    size_t N = (size_t)H * W;
    for (int k = 0; k < 16; k++) D.P[k].resize(N);
    D.key.resize(N); D.s.resize(N);
    D.av.resize(N); D.bv.resize(N); D.cv.resize(N); D.dv.resize(N);
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        int a, b, c, d, Ww, NNe, NE;
        nbhd_b4(ch.data(), H, W, i, j, a, b, c, d, Ww, NNe, NE);
        size_t n = (size_t)i * W + j;
        D.av[n] = a; D.bv[n] = b; D.cv[n] = c; D.dv[n] = d;
        for (int x = 0; x < 16; x++) D.P[x][n] = pred_e16(x, a, b, c, d, Ww, NNe, NE);
        int q1 = loco_q1(b - c), q2 = loco_q1(c - a), q3 = loco_q1(d - b);
        int sg = 1;
        if (q1 < 0 || (q1 == 0 && q2 < 0) || (q1 == 0 && q2 == 0 && q3 < 0)) {
            sg = -1; q1 = -q1; q2 = -q2; q3 = -q3; }
        D.key[n] = q1 * 81 + q2 * 9 + q3;
        D.s[n] = (int8_t)sg;
    }
    return D;
}

// ---- RCT forward (probe_b17_rctw.rct_fwd verbatim) ----
void rct_fwd(const uint8_t *rgb, int H, int W, int perm, int t, Img &yc) {
    yc.H = H; yc.W = W;
    size_t N = (size_t)H * W;
    for (int k = 0; k < 3; k++) yc.p[k].resize(N);
    const int *pm = PERMS[perm];
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        size_t n = (size_t)i * W + j;
        int A = rgb[n * 3 + pm[0]], B = rgb[n * 3 + pm[1]], C = rgb[n * 3 + pm[2]];
        int X, Y, Z;
        if (t == 5) { X = A; Y = B - fdiv(A + C, 2); Z = C - A; }
        else if (t == 6) {
            int Co = A - C, tt = C + fdiv(Co, 2), Cg = B - tt;
            int Yy = tt + fdiv(Cg, 2);
            X = Yy; Y = Co; Z = Cg;
        } else DIE("unsupported RCT t=%d", t);
        yc.p[0][n] = X; yc.p[1][n] = Y; yc.p[2][n] = Z;
    }
}

// Managed inverse (only used for the encoder-side RCT round-trip assert)
void rct_inv_planes(const int32_t *p0, const int32_t *p1, const int32_t *p2,
                    int H, int W, int perm, int t, uint8_t *rgb) {
    size_t N = (size_t)H * W;
    int unp[3] = {0, 0, 0};
    for (int k = 0; k < 3; k++) unp[PERMS[perm][k]] = k;
    for (size_t n = 0; n < N; n++) {
        int X = p0[n], Y = p1[n], Z = p2[n], A, B, C;
        if (t == 5) { A = X; B = Y + X + fdiv(Z, 2); C = Z + X; }
        else if (t == 6) {
            int tt = X - fdiv(Z, 2), G = Z + tt, Bc = tt - fdiv(Y, 2);
            A = Y + Bc; B = G; C = Bc;
        } else DIE("unsupported RCT t=%d", t);
        int q[3] = {A, B, C};
        for (int k = 0; k < 3; k++) {
            int v = q[unp[k]];
            CHECK(v >= 0 && v <= 255, "RCT inv range (%d)", v);
            rgb[n * 3 + k] = (uint8_t)v;
        }
    }
}

// ---- B17 MED + nbhd4 + weighted fit ----
void med_pred_plane(const int32_t *P, int H, int W, std::vector<int32_t> &out) {
    size_t N = (size_t)H * W; out.resize(N);
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        int L = (j > 0) ? P[i * W + j - 1] : ((i > 0) ? P[(i - 1) * W] : 0);
        int T = (i > 0) ? P[(i - 1) * W + j] : ((j > 0) ? P[i * W + j - 1] : 0);
        int TL = (i > 0 && j > 0) ? P[(i - 1) * W + j - 1] : 0;
        int mn = L < T ? L : T, mx = L < T ? T : L;
        out[(size_t)i * W + j] = TL >= mx ? mn : (TL <= mn ? mx : L + T - TL);
    }
}

int wpred_from_w(const int8_t *w, int L, int T, int TL, int TR) {
    int num = (int)w[0] * L + (int)w[1] * T + (int)w[2] * TL + (int)w[3] * TR;
    return fdiv(num + 8, 16);
}

// 4x4 least squares via normal equations + partial-pivot Gaussian elimination.
// Replicates np.linalg.lstsq (gelsd, full-rank min-norm) up to rounding of
// wq=clip(round_half_even(sol*16)); VERIFIED block-for-block against the
// reference (see cpp/README.md). Any systematic mismatch aborts loudly.
static void lstsq4(const double *Cg /*n*4 row-major*/, const double *yg, int n,
                   double sol[4]) {
    double G[4][4] = {}, c[4] = {};
    for (int r = 0; r < n; r++) {
        const double *a = Cg + (size_t)r * 4;
        for (int i = 0; i < 4; i++) { c[i] += a[i] * yg[r];
            for (int j = 0; j < 4; j++) G[i][j] += a[i] * a[j]; }
    }
    double M[4][5];
    for (int i = 0; i < 4; i++) { for (int j = 0; j < 4; j++) M[i][j] = G[i][j];
        M[i][4] = c[i]; }
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(M[r][col]) > fabs(M[piv][col])) piv = r;
        CHECK(fabs(M[piv][col]) > 0, "lstsq singular block (rank-deficient)");
        if (piv != col) for (int j = col; j < 5; j++) std::swap(M[col][j], M[piv][j]);
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            double f = M[r][col] / M[col][col];
            for (int j = col; j < 5; j++) M[r][j] -= f * M[col][j];
        }
    }
    for (int i = 0; i < 4; i++) sol[i] = M[i][4] / M[i][i];
}

WFit weighted_fit_ch(const int32_t *plane, int H, int W) {
    WFit F;
    F.nbw = (W + GS - 1) / GS; F.nbh = (H + GS - 1) / GS; F.ng = F.nbh * F.nbw;
    size_t N = (size_t)H * W;
    std::vector<int32_t> L(N), T(N), TL(N), TR(N), med(N);
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        size_t n = (size_t)i * W + j;
        L[n] = (j > 0) ? plane[n - 1] : ((i > 0) ? plane[(size_t)(i - 1) * W] : 0);
        T[n] = (i > 0) ? plane[(size_t)(i - 1) * W + j]
                       : ((j > 0) ? plane[n - 1] : 0);
        TL[n] = (i > 0 && j > 0) ? plane[(size_t)(i - 1) * W + j - 1] : 0;
        TR[n] = (i > 0 && j + 1 < W) ? plane[(size_t)(i - 1) * W + j + 1] : 0;
        int mn = L[n] < T[n] ? L[n] : T[n], mx = L[n] < T[n] ? T[n] : L[n];
        med[n] = TL[n] >= mx ? mn : (TL[n] <= mn ? mx : L[n] + T[n] - TL[n]);
    }
    F.res.resize(N); F.use.assign(F.ng, 0); F.w.assign((size_t)F.ng * 4, 0);
    std::vector<double> Cg; Cg.reserve(1024 * 4);
    std::vector<double> yg; yg.reserve(1024);
    std::vector<int> idx; idx.reserve(1024);
    for (int g = 0; g < F.ng; g++) {
        int bi = g / F.nbw, bj = g % F.nbw;
        idx.clear();
        for (int i = bi * GS; i < std::min(bi * GS + GS, H); i++)
            for (int j = bj * GS; j < std::min(bj * GS + GS, W); j++)
                idx.push_back(i * W + j);
        int64_t s_med = 0;
        for (int n : idx) { int64_t v = (int64_t)plane[n] - med[n];
            s_med += v >= 0 ? v : -v; }
        double b_med = (double)s_med / (double)idx.size() + 1e-9;
        Cg.assign(idx.size() * 4, 0.0); yg.assign(idx.size(), 0.0);
        for (size_t r = 0; r < idx.size(); r++) {
            int n = idx[r];
            Cg[r * 4 + 0] = (double)L[n]; Cg[r * 4 + 1] = (double)T[n];
            Cg[r * 4 + 2] = (double)TL[n]; Cg[r * 4 + 3] = (double)TR[n];
            yg[r] = (double)plane[n];
        }
        double sol[4] = {0, 0, 0, 0};
        lstsq4(Cg.data(), yg.data(), (int)idx.size(), sol);
        int32_t wq[4];
        for (int k = 0; k < 4; k++) {
            double v = std::nearbyint(sol[k] * 16.0); // half-even == np.round
            if (v < -16) { v = -16; } if (v > 15) { v = 15; }
            wq[k] = (int32_t)v;
        }
        bool ok = false;
        if (wq[0] || wq[1] || wq[2] || wq[3]) {
            int64_t s_w = 0;
            for (size_t r = 0; r < idx.size(); r++) {
                int n = idx[r];
                // exact integer path: products/sums are small, bit-exact
                int64_t num = (int64_t)L[n] * wq[0] + (int64_t)T[n] * wq[1]
                            + (int64_t)TL[n] * wq[2] + (int64_t)TR[n] * wq[3];
                int64_t e = (int64_t)plane[n] - fdiv((int)(num + 8), 16);
                s_w += e >= 0 ? e : -e;
            }
            double b_w = (double)s_w / (double)idx.size() + 1e-9;
            if (b_w < b_med && (double)idx.size() * std::log2(b_med / b_w) > 21.0) {
                ok = true;
                for (size_t r = 0; r < idx.size(); r++) {
                    int n = idx[r];
                    int64_t num = (int64_t)L[n] * wq[0] + (int64_t)T[n] * wq[1]
                                + (int64_t)TL[n] * wq[2] + (int64_t)TR[n] * wq[3];
                    F.res[n] = plane[n] - fdiv((int)(num + 8), 16);
                }
            }
        }
        if (!ok) for (int n : idx) F.res[n] = plane[n] - med[n];
        F.use[g] = ok ? 1 : 0;
        if (ok) for (int k = 0; k < 4; k++) F.w[(size_t)g * 4 + k] = (int8_t)wq[k];
    }
    return F;
}

// ---- LMS sign-sign 5-tap (probe_b18_stack.lms_pred_plane verbatim) ----
void lms_pred_plane(const int32_t *plane, int H, int W, int thr,
                    std::vector<int32_t> &out) {
    size_t N = (size_t)H * W; out.resize(N);
    int w[5] = {3, 3, 3, 3, 4};
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        int a, b, c, d, Ww, NNe, NE;
        nbhd_b4(plane, H, W, i, j, a, b, c, d, Ww, NNe, NE);
        int L = (j > 0) ? plane[i * W + j - 1] : ((i > 0) ? plane[(i - 1) * W] : 0);
        int T = (i > 0) ? plane[(i - 1) * W + j] : ((j > 0) ? plane[i * W + j - 1] : 0);
        int TL0 = (i > 0 && j > 0) ? plane[(i - 1) * W + j - 1] : 0;
        int mn = L < T ? L : T, mx = L < T ? T : L;
        int md = TL0 >= mx ? mn : (TL0 <= mn ? mx : L + T - TL0);
        // xs = [L,T,TL,TR(b4),MED5(B17)]; NOTE b,c here are b4 TL/TR copies
        int xs[5] = {a, b, c, d, md};
        int pr = fdiv(w[0] * xs[0] + w[1] * xs[1] + w[2] * xs[2] + w[3] * xs[3]
                      + w[4] * xs[4] + 8, 16);
        size_t n = (size_t)i * W + j;
        out[n] = pr;
        int y = plane[n], e = y - pr;
        if (abs(e) > thr) {
            int se = e > 0 ? 1 : -1;
            int m = fdiv(xs[0] + xs[1] + xs[2] + xs[3] + xs[4] + 2, 5);
            for (int k = 0; k < 5; k++) { int di = xs[k] - m;
                if (di > 0) w[k] += se; else if (di < 0) w[k] -= se; }
            for (int k = 0; k < 5; k++) { if (w[k] < -8) w[k] = -8;
                else if (w[k] > 20) w[k] = 20; }
            int s = w[0] + w[1] + w[2] + w[3] + w[4];
            while (s > 16) { int bi = 0;
                for (int k = 1; k < 5; k++) if (w[k] > w[bi]) bi = k;
                w[bi]--; s--; }
            while (s < 16) { int bi = 0;
                for (int k = 1; k < 5; k++) if (w[k] < w[bi]) bi = k;
                w[bi]++; s++; }
        }
    }
}

// ---- MLP causal taps / energy contexts / wire forward ----
static inline int c6_edge(const int32_t *P, int H, int W, int i, int j, int di, int dj) {
    int r = i + di, c = j + dj;
    if (r < 0 || r >= H || c < 0 || c >= W) return 0;
    return P[r * W + c];
}

void mlp_causal_taps(const int32_t *P, int H, int W, int i, int j, int fx[12]) {
    int L = c6_edge(P, H, W, i, j, 0, -1), T = c6_edge(P, H, W, i, j, -1, 0);
    fx[0] = L; fx[1] = T;
    fx[2] = c6_edge(P, H, W, i, j, -1, -1);
    fx[3] = c6_edge(P, H, W, i, j, -1, 1);
    fx[4] = c6_edge(P, H, W, i, j, 0, -2);
    fx[5] = c6_edge(P, H, W, i, j, -2, 0);
    fx[6] = fdiv(L + T, 2);
    fx[7] = abs(L - T);
    fx[8] = c6_edge(P, H, W, i, j, 0, -3);
    fx[9] = c6_edge(P, H, W, i, j, -3, 0);
    fx[10] = c6_edge(P, H, W, i, j, -2, -2);
    fx[11] = c6_edge(P, H, W, i, j, -2, 2);
}

double quantile_linear(const std::vector<double> &s, double q) {
    size_t n = s.size();
    CHECK(n > 0, "quantile of empty array");
    double v = (double)(n - 1) * q;
    double lo = floor(v);
    size_t ilo = (size_t)lo, ihi = ilo + 1;
    if (!(v < (double)(n - 1))) { ilo = n - 1; ihi = n - 1; lo = v; }
    if (ihi >= n) ihi = n - 1;
    double g = v - floor(v);
    if (v >= (double)(n - 1)) g = v - floor(v); // ==0 up to rounding; keep formula
    double a = s[ilo], b = s[ihi], d = b - a;
    double r = a + d * g;
    if (g >= 0.5) r = b - d * (1.0 - g); // numpy _lerp mirrored branch
    return r;
}

void mlp_energy_ctx(const int32_t *P, int H, int W,
                    std::vector<double> &qs, std::vector<int> &ctx,
                    std::vector<int> &E) {
    size_t N = (size_t)H * W;
    E.resize(N);
    std::vector<double> Es(N);
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        int L = c6_edge(P, H, W, i, j, 0, -1), T = c6_edge(P, H, W, i, j, -1, 0),
            TL = c6_edge(P, H, W, i, j, -1, -1);
        int e = abs(L - T) + abs(L - TL) + abs(T - TL);
        E[(size_t)i * W + j] = e; Es[(size_t)i * W + j] = (double)e;
    }
    std::sort(Es.begin(), Es.end());
    qs.resize(8);
    for (int k = 1; k <= 8; k++) qs[k - 1] = quantile_linear(Es, (double)k * (1.0 / 9.0));
    ctx.resize(N);
    for (size_t n = 0; n < N; n++) {
        // searchsorted side='left' == #{thr < E}
        int c = 0; for (int t = 0; t < 8; t++) if (qs[t] < (double)E[n]) c++;
        ctx[n] = c;
    }
}

int mlp_fwd_wire(const int fx[12], const double *W1, const double *b1,
                 const double *W2, double b2) {
    double pre[8], h[8];
    for (int j = 0; j < 8; j++) {
        double acc = b1[j];
        for (int i = 0; i < 12; i++) acc += ((double)fx[i] / 128.0) * W1[j * 12 + i];
        pre[j] = acc;
    }
    for (int j = 0; j < 8; j++) {
        double pr = pre[j], rr = (pr < 0.0 ? -pr : pr) * 16384.0, hh;
        if (rr > 131072.0) hh = pr < 0.0 ? -1.0 : 1.0;
        else { int idx = (int)floor(rr + 0.5);
            hh = (pr < 0.0 ? -1.0 : 1.0) * C6_TANH[idx]; }
        h[j] = hh;
    }
    double y = b2;
    for (int j = 0; j < 8; j++) y += W2[j] * h[j];
    return (int)floor(y * 128.0 + 0.5);
}

// ---- exact CPython heapq port (tuples compare lexicographically) ----
struct HIt { int64_t a, b; };
static inline bool hlt(const HIt &x, const HIt &y) {
    return x.a < y.a || (x.a == y.a && x.b < y.b); }
static void hsiftdown(std::vector<HIt> &h, size_t startpos, size_t pos) {
    HIt ni = h[pos];
    while (pos > startpos) {
        size_t pp = (pos - 1) >> 1;
        if (hlt(ni, h[pp])) { h[pos] = h[pp]; pos = pp; continue; }
        break;
    }
    h[pos] = ni;
}
static void hsiftup(std::vector<HIt> &h, size_t pos) {
    size_t end = h.size(), start = pos;
    HIt ni = h[pos];
    size_t ch = 2 * pos + 1;
    while (ch < end) {
        size_t rt = ch + 1;
        if (rt < end && !hlt(h[ch], h[rt])) ch = rt;
        h[pos] = h[ch]; pos = ch; ch = 2 * pos + 1;
    }
    h[pos] = ni;
    hsiftdown(h, start, pos);
}
static inline void hpush(std::vector<HIt> &h, HIt v) {
    h.push_back(v); hsiftdown(h, 0, h.size() - 1); }
static inline HIt hpop(std::vector<HIt> &h) {
    HIt last = h.back(); h.pop_back();
    if (h.empty()) return last;
    HIt ret = h[0]; h[0] = last; hsiftup(h, 0); return ret;
}
static inline void hheapify(std::vector<HIt> &h) {
    if (h.size() < 2) return;
    for (size_t i = h.size() / 2; i-- > 0;) hsiftup(h, i);
}

int64_t huff_bits_counts(const std::vector<int64_t> &counts) {
    size_t n = counts.size();
    if (n == 0) return 0;
    if (n == 1) return counts[0] * 1;
    std::vector<HIt> h;
    for (auto c : counts) h.push_back({c, 0});
    hheapify(h);
    // NOTE: plain-int heap (no ids): equal elements are indistinguishable,
    // so the missing tie-break is immaterial. Total is all that is used.
    int64_t t = 0;
    while (h.size() > 1) {
        int64_t x = hpop(h).a, y = hpop(h).a, s = x + y;
        t += s; hpush(h, {s, 0});
    }
    return t;
}

void canon_tables(const int64_t hist[2049], uint8_t lens[2049], uint32_t codes[2049]) {
    std::vector<int> syms;
    for (int i = 0; i < 2049; i++) if (hist[i] > 0) syms.push_back(i);
    memset(lens, 0, 2049); memset(codes, 0, 2049 * 4);
    if (syms.empty()) return;
    if (syms.size() == 1) { lens[syms[0]] = 1; codes[syms[0]] = 0; return; }
    std::vector<HIt> H;
    for (int s : syms) H.push_back({hist[s], s});
    hheapify(H);
    std::vector<HIt> H2 = H;
    int64_t nxt = (int64_t)1 << 28;
    // parent id per node id (leaves 0..2048, internals nxt++); sparse -> hash map
    std::unordered_map<int64_t, int64_t> par;
    par.reserve(8192);
    while (H2.size() > 1) {
        HIt A = hpop(H2), B = hpop(H2);
        int64_t nn = nxt++;
        par[A.b] = nn; par[B.b] = nn;
        hpush(H2, {A.a + B.a, nn});
    }
    std::vector<std::pair<int,int>> ol; // (len, sym)
    for (int s : syms) {
        int d = 0; int64_t n = s;
        auto it = par.find(n);
        while (it != par.end()) { n = it->second; d++; it = par.find(n); }
        lens[s] = (uint8_t)d;
        ol.push_back({d, s});
    }
    std::sort(ol.begin(), ol.end());
    uint32_t code = 0; int prev = 0;
    for (auto &pr : ol) {
        code <<= (pr.first - prev);
        codes[pr.second] = code;
        code++; prev = pr.first;
    }
}

// ---- Golomb best-k (probe_b5_d.golomb_best verbatim semantics) ----
void golomb_best(const int32_t *g, size_t n, int64_t &tot, int &k) {
    if (n == 0) { tot = 0; k = 0; return; }
    bool first = true;
    tot = 0; k = 0;
    for (int kk = 0; kk < 13; kk++) {
        int64_t s = 0;
        for (size_t i = 0; i < n; i++) {
            int64_t M = g[i] >= 0 ? (int64_t)g[i] * 2 : (int64_t)(-(int64_t)g[i]) * 2 - 1;
            s += M >> kk;
        }
        int64_t t = s + (int64_t)n * (1 + kk);
        if (first || t < tot) { tot = t; k = kk; first = false; }
    }
}

// Scalar fused sums for one full (base + 7 biases) evaluation.
static void golomb_sums_scalar(const int32_t *g, size_t n,
                               int64_t S0[13], int64_t Sd[8][13]) {
    for (size_t i = 0; i < n; i++) {
        int64_t v = g[i];
        int64_t M0 = v >= 0 ? v * 2 : (-v) * 2 - 1;
        for (int k = 0; k < 13; k++) S0[k] += M0 >> k;
        for (int di = 0; di < 8; di++) {
            if (di == 4) continue;  // d=0 dead
            int64_t u = v - (di - 4);
            int64_t Mu = u >= 0 ? u * 2 : (-u) * 2 - 1;
            for (int k = 0; k < 13; k++) Sd[di][k] += Mu >> k;
        }
    }
}

GolombCost golomb_cost(const int32_t *g, size_t n) {
    GolombCost r;
    if (n == 0) { r.tot = 0; r.k = 0; r.d = 0; r.dbits = 0; return r; }
    // Single-pass fusion of golomb_best(g) + golomb_best(g-d) for
    // d in {-4..-1,1..3}. d=0 in the loop is PROVEN DEAD (it compares
    // gd+4+3 against best<=gd0+4 with identical gd) and skipped.
    // Bit-exact: int sums of (zigzag>>k); integer addition is exact and
    // order-invariant, so SIMD chunking/regrouping changes nothing.
    // Selection order (k=0..12 strict <; d=-4..3) replicates verbatim.
    // Speed: 8 sweeps (was 9x13=117) + no tmp alloc. NOTE: an AVX2 i32
    // variant of this loop was tried and reverted: the autovectorizer already
    // captures most of it, and 8 extra sweeps trade ~4% CPU for dual code
    // paths + an untestable n-guard fallback. See cycle-54 memory note.
    int64_t S0[13] = {0}, Sd[8][13] = {{0}};
    golomb_sums_scalar(g, n, S0, Sd);
    int64_t nn = (int64_t)n;
    bool first = true;
    int64_t best = 0; int bk = 0, bd = 0;
    for (int k = 0; k < 13; k++) {
        int64_t t = S0[k] + nn * (1 + k);
        if (first || t < best) { best = t; bk = k; first = false; }
    }
    best += 4; bd = 0;
    // NOTE: selection over (d,k) must compare gd+4+3 against running best,
    // where gd = min_k Sd+nn(1+k) per d — replicate per-d best-k first.
    // di==4 (d=0) is dead (gd+4+3 can never beat best<=gd0+4); skipped.
    for (int di = 0; di < 8; di++) {
        if (di == 4) continue;
        int64_t gd = 0; int kk = 0; bool f2 = true;
        for (int k = 0; k < 13; k++) {
            int64_t t = Sd[di][k] + nn * (1 + k);
            if (f2 || t < gd) { gd = t; kk = k; f2 = false; }
        }
        if (gd + 4 + 3 < best) { best = gd + 4 + 3; bk = kk; bd = di - 4; }
    }
    r.tot = best; r.k = bk; r.d = bd; r.dbits = (bd != 0) ? 3 : 0;
    return r;
}

int64_t rans_cost(const int16_t *g, size_t n, RansBlob &blob) {
    if (n == 0) { blob = RansBlob(); return 0; }
    std::vector<int> vals(g, g + n);
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    int Ad = (int)vals.size();
    std::vector<int32_t> syms(Ad);
    for (int i = 0; i < Ad; i++) syms[i] = vals[i] + 1024;
    std::vector<int64_t> hist(2049, 0);
    for (size_t i = 0; i < n; i++) hist[(int)g[i] + 1024]++;
    std::vector<uint16_t> freq(Ad);
    std::vector<int32_t> cum(Ad);
    rans_norm(hist.data(), Ad, syms.data(), freq.data(), cum.data());
    std::vector<int32_t> lut(2049, -1);
    for (int i = 0; i < Ad; i++) lut[syms[i]] = i;
    std::vector<uint8_t> out(n * 3 + 16);
    size_t nn = rans_encode(g, (int)n, lut.data(), freq.data(), cum.data(), out.data());
    CHECK(nn > 0, "rANS encode failed");
    std::vector<int32_t> slot(1 << 14);
    rans_slots(freq.data(), cum.data(), Ad, slot.data());
    std::vector<int16_t> dec(n);
    int rc = rans_decode(out.data(), nn, (int)n, syms.data(), freq.data(),
                         cum.data(), Ad, slot.data(), dec.data());
    CHECK(rc == 0, "rANS roundtrip FAIL");
    for (size_t i = 0; i < n; i++) CHECK(dec[i] == g[i], "rANS roundtrip FAIL");
    blob.pay.assign(out.begin(), out.begin() + nn);
    blob.syms = syms; blob.freq = freq; blob.cum = cum;
    return (int64_t)16 + (int64_t)Ad * 32 + 64 + (int64_t)nn * 8;
}

// ---- minimal ZIP(STORED)+NPY reader for the frozen MLP weights ----
static uint16_t rd16(const uint8_t *d) { return (uint16_t)(d[0] | (d[1] << 8)); }
static uint32_t rd32(const uint8_t *d) {
    return (uint32_t)d[0] | ((uint32_t)d[1] << 8) | ((uint32_t)d[2] << 16) |
           ((uint32_t)d[3] << 24); }

struct NpyInfo { char kind; int itemsize; std::vector<int64_t> shape; size_t off; };

static bool parse_npy(const uint8_t *d, size_t n, NpyInfo &ni) {
    if (n < 10) return false;
    static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (memcmp(d, magic, 6) != 0) return false;
    uint16_t hlen = rd16(d + 8);
    if (10u + hlen > n) return false;
    std::string hdr((const char *)d + 10, hlen);
    auto get = [&](const char *k) -> std::string {
        std::string key = std::string("'") + k + "':";
        size_t p = hdr.find(key);
        CHECK(p != std::string::npos, "npy header missing %s", k);
        return hdr.substr(p + key.size());
    };
    std::string descr = get("descr");
    size_t q1 = descr.find('\'');
    CHECK(q1 != std::string::npos, "npy descr parse");
    std::string dt = descr.substr(q1 + 1, 3);
    ni.kind = dt[1]; ni.itemsize = 0;
    if (dt == "<f8") { ni.kind = 'f'; ni.itemsize = 8; }
    else if (dt == "<i2") { ni.kind = 'i'; ni.itemsize = 2; }
    else if (dt == "|u1") { ni.kind = 'u'; ni.itemsize = 1; }
    else DIE("unsupported npy dtype %s", dt.c_str());
    std::string shp = get("shape");
    ni.shape.clear();
    int64_t v = -1; bool innum = false;
    for (char ch : shp) {
        if (ch >= '0' && ch <= '9') { if (!innum) { v = 0; innum = true; } v = v * 10 + (ch - '0'); }
        else { if (innum) { ni.shape.push_back(v); innum = false; } }
    }
    if (innum) ni.shape.push_back(v);
    ni.off = 10 + hlen;
    return true;
}

bool npz_load_ch(const std::string &path, const std::string &rct, int ch, NpzNet &nt) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    CHECK(sz > 0, "empty npz %s", path.c_str());
    std::vector<uint8_t> z(sz);
    CHECK(fread(z.data(), 1, sz, f) == (size_t)sz, "npz read %s", path.c_str());
    fclose(f);
    bool got_qs = false, got_win = false, got_scid = false, got_qw = false;
    size_t pos = 0;
    while (pos + 30 <= z.size()) {
        uint32_t sig = rd32(z.data() + pos);
        if (sig == 0x02014b50 || sig == 0x06054b50) break; // central dir / EOCD
        CHECK(sig == 0x04034b50, "npz not a stored zip (%s off %zu)", path.c_str(), pos);
        uint16_t method = rd16(z.data() + pos + 8);
        uint32_t csize = rd32(z.data() + pos + 18);
        uint32_t usize = rd32(z.data() + pos + 22);
        uint16_t fnlen = rd16(z.data() + pos + 26), eflen = rd16(z.data() + pos + 28);
        // ZIP64 (np.savez writes 0xFFFFFFFF + 0x0001 extra with 8-byte sizes)
        if (csize == 0xFFFFFFFFu || usize == 0xFFFFFFFFu) {
            size_t ep = pos + 30 + fnlen, eend = ep + eflen;
            bool ok64 = false;
            while (ep + 4 <= eend) {
                uint16_t eid = z[ep] | (z[ep + 1] << 8);
                uint16_t esz = z[ep + 2] | (z[ep + 3] << 8);
                if (eid == 0x0001 && esz >= 16 && ep + 4 + 16 <= eend) {
                    uint64_t u64 = 0, c64 = 0;
                    for (int i = 0; i < 8; i++) {
                        u64 |= (uint64_t)z[ep + 4 + i] << (8 * i);
                        c64 |= (uint64_t)z[ep + 12 + i] << (8 * i);
                    }
                    CHECK(u64 == c64 && c64 < (uint64_t)sz, "zip64 sizes");
                    csize = (uint32_t)c64; usize = (uint32_t)u64; ok64 = true;
                    break;
                }
                ep += 4 + esz;
            }
            CHECK(ok64, "zip64 extra missing");
        }
        std::string name((const char *)z.data() + pos + 30, fnlen);
        size_t doff = pos + 30 + fnlen + eflen;
        CHECK(doff + csize <= z.size(), "npz entry overrun %s", name.c_str());
        CHECK(method == 0, "npz entry compressed (need stored): %s", name.c_str());
        std::string want = rct + "/ch" + (char)('0' + ch) + "/";
        if (name.compare(0, want.size(), want) == 0) {
            std::string tail = name.substr(want.size());
            // strip ".npy"
            CHECK(tail.size() > 4, "npz name %s", name.c_str());
            tail = tail.substr(0, tail.size() - 4);
            NpyInfo ni;
            CHECK(parse_npy(z.data() + doff, csize, ni), "npy parse %s", name.c_str());
            const uint8_t *dd = z.data() + doff + ni.off;
            if (tail == "qs") {
                CHECK(ni.kind == 'f' && ni.shape == std::vector<int64_t>{8}, "qs shape");
                memcpy(nt.qs, dd, 64); got_qs = true;
            } else if (tail == "win") {
                CHECK(ni.shape == std::vector<int64_t>{9}, "win shape");
                memcpy(nt.win, dd, 9); got_win = true;
            } else if (tail == "scid") {
                CHECK(ni.shape == std::vector<int64_t>{9}, "scid shape");
                memcpy(nt.scid, dd, 9); got_scid = true;
            } else if (tail == "qw") {
                CHECK(ni.kind == 'i' && ni.shape == (std::vector<int64_t>{9, 113}), "qw shape");
                memcpy(nt.qw, dd, 9 * 113 * 2); got_qw = true;
            } else DIE("unexpected npz entry %s", name.c_str());
        }
        pos = doff + csize;
    }
    return got_qs && got_win && got_scid && got_qw;
}

bool load_png_rgb(const std::string &path, std::vector<uint8_t> &rgb, int &H, int &W) {
    int w = 0, h = 0, comp = 0;
    uint8_t *px = stbi_load(path.c_str(), &w, &h, &comp, 3);
    if (!px) return false;
    H = h; W = w;
    rgb.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    return true;
}

} // namespace crown
