// CROWN6 C++ encoder (NEW file). Deterministic, self-contained C++17.
// Mirrors src/driver_crown6.py encode path with train=False (weights loaded
// from cached .npz; NO torch/Python at runtime). Links READ-ONLY C objects
// for pack_syms / rans_* / crown6_golomb_pack. Emits exact CROWN6 wire bytes.
//
// Usage: crown_enc <in.png> <weights_dir> <out.bin>
#include "codec.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <map>
#include <mutex>
#include <numeric>

namespace crown {

// ---- encode-speed timers (env-gated CROWN_TIME; behavior-preserving) ----
// Thread-local accumulation merged under one lock at task join, so worker
// threads never contend and outputs stay bit-identical at any --jobs value.
struct TSet { double rct=0,wfit=0,lms=0,mlp=0,prep=0,qcoarse=0,qrefine=0,
    gcoarse=0,grefine=0,asm_=0,huff=0,golomb=0,rans=0,gather=0; };
static TSet T_all; static std::mutex T_mu;
static thread_local TSet T_loc;
struct TAcc { std::chrono::steady_clock::time_point t0; double TSet::*m;
    TAcc(double TSet::*mm): t0(std::chrono::steady_clock::now()), m(mm) {}
    ~TAcc(){ T_loc.*m += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); } };
static void T_merge(){ std::lock_guard<std::mutex> g(T_mu);
    T_all.rct+=T_loc.rct; T_all.wfit+=T_loc.wfit; T_all.lms+=T_loc.lms;
    T_all.mlp+=T_loc.mlp; T_all.prep+=T_loc.prep; T_all.qcoarse+=T_loc.qcoarse;
    T_all.qrefine+=T_loc.qrefine; T_all.gcoarse+=T_loc.gcoarse;
    T_all.grefine+=T_loc.grefine; T_all.asm_+=T_loc.asm_;
    T_all.huff+=T_loc.huff; T_all.golomb+=T_loc.golomb; T_all.rans+=T_loc.rans;
    T_all.gather+=T_loc.gather; T_loc = TSet(); }
namespace {

// ---------- grouping ----------
struct Groups {
    int K = 0;
    std::vector<std::vector<int>> keylists; // per group: member keys
};

Groups quantile_groups(const std::vector<int> &key, const int32_t *rfM, size_t N, int K) {
    std::vector<int> uk = key;
    std::sort(uk.begin(), uk.end());
    uk.erase(std::unique(uk.begin(), uk.end()), uk.end());
    int U = (int)uk.size();
    std::vector<int> keyidx(N);
    std::vector<int> kpos(729, -1);
    for (int i = 0; i < U; i++) kpos[uk[i]] = i;
    for (size_t n = 0; n < N; n++) keyidx[n] = kpos[key[n]];
    std::vector<int64_t> sum(U, 0);
    std::vector<int64_t> cnt(U, 0);
    for (size_t n = 0; n < N; n++) {
        int64_t v = rfM[n] >= 0 ? rfM[n] : -(int64_t)rfM[n];
        sum[keyidx[n]] += v; cnt[keyidx[n]]++;
    }
    std::vector<double> ma(U);
    for (int i = 0; i < U; i++) ma[i] = (double)sum[i] / (double)cnt[i];
    std::vector<int> order(U);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int x, int y) { return ma[x] < ma[y]; });
    int64_t tot = 0;
    for (auto c : cnt) tot += c;
    double tgt = (double)tot / (double)K;
    Groups G; G.K = K;
    std::vector<int> cur; double acc = 0;
    for (int oi = 0; oi < U; oi++) {
        int u = uk[order[oi]];
        cur.push_back(u); acc += (double)cnt[order[oi]];
        if (acc >= tgt && (int)G.keylists.size() < K - 1) {
            G.keylists.push_back(cur); cur.clear(); acc = 0; }
    }
    G.keylists.push_back(cur);
    return G;
}

// ---------- per-group expert/backend selection ----------
struct Win { int64_t cost = 0; int expert = 0, backend = 0, k = 0, d = 0;
    RansBlob blob; };

int64_t huff_cost(const int32_t *g, size_t n) {
    if (n == 0) return 0;
    TAcc _t(&TSet::huff);
    // Histogram over the proven ±1024 alphabet (O(n)) instead of copy+sort
    // (O(n log n)). Bit-exact: the count MULTISET is identical; totals from
    // huff_bits_counts are tie-break-invariant (plain-int heap, see codec.cpp),
    // and counts are sorted ascending for a deterministic insertion order.
    int64_t hist[2049] = {0};
    for (size_t i = 0; i < n; i++) {
        int v = g[i];
        CHECK(v >= -1024 && v <= 1024, "huff_cost range %d", v);
        hist[v + 1024]++;
    }
    std::vector<int64_t> counts;
    for (int s = 0; s < 2049; s++) if (hist[s]) counts.push_back(hist[s]);
    std::sort(counts.begin(), counts.end());
    return huff_bits_counts(counts) + 16 + (int64_t)counts.size() * 24;
}

// group columns: cols[expert] values in raster order; empty experts allowed
struct GCols { std::vector<std::vector<int32_t>> cols; }; // indexed by position in elist
Win best_hg(const GCols &gd, const std::vector<int> &elist, bool skip_mlp) {
    Win bw; bool first = true;
    for (size_t e = 0; e < elist.size(); e++) {
        int name = elist[e];
        if (skip_mlp && name == EX_MLP) continue;
        const std::vector<int32_t> &g = gd.cols[e];
        int64_t ht = 0, gt = 0; int gk = 0, gdlt = 0;
        if (!g.empty()) {
            ht = huff_cost(g.data(), g.size());
            GolombCost gc; { TAcc _t(&TSet::golomb); gc = golomb_cost(g.data(), g.size()); }
            gt = gc.tot; gk = gc.k; gdlt = gc.d;
        }
        if (first || ht < bw.cost) { bw.cost = ht; bw.expert = name;
            bw.backend = 0; bw.k = 0; bw.d = 0; first = false; }
        if (gt < bw.cost) { bw.cost = gt; bw.expert = name;
            bw.backend = 1; bw.k = gk; bw.d = gdlt; }
    }
    return bw;
}

struct DualRes { int64_t tot_all = 0, tot_no = 0;
    std::vector<Win> wins_all, wins_no; };

DualRes solve_groups_dual(const std::vector<GCols> &gvs,
                           const std::vector<int> &elist, bool use_rans,
                           bool need_no_mlp = true) {
    DualRes R;
    for (const GCols &gd : gvs) {
        Win outs[2];
        for (int t = 0; t < 2; t++) {
            if (t == 1 && !need_no_mlp) continue; // dead track: caller uses tot_all only
            Win bw = best_hg(gd, elist, t == 1);
            RansBlob blob;
            if (use_rans) {
                size_t ei = 0;
                for (; ei < elist.size(); ei++) if (elist[ei] == bw.expert) break;
                const std::vector<int32_t> &g = gd.cols[ei];
                if (!g.empty()) {
                    std::vector<int16_t> gg(g.size());
                    for (size_t i = 0; i < g.size(); i++) gg[i] = (int16_t)g[i];
                    RansBlob bl;
                    int64_t rt; { TAcc _t(&TSet::rans); rt = rans_cost(gg.data(), gg.size(), bl); }
                    if (rt < bw.cost) { bw.cost = rt; bw.backend = 2;
                        bw.k = 0; bw.d = 0; bw.blob = std::move(bl); }
                }
            }
            outs[t] = bw;
        }
        R.tot_all += outs[0].cost;
        if (need_no_mlp) {
            R.tot_no += outs[1].cost;
            R.wins_no.push_back(outs[1]);
        }
        R.wins_all.push_back(outs[0]);
    }
    return R;
}

// ---------- channel plans ----------
struct PlanQ { int K = 0; Groups groups; std::vector<Win> wins_all, wins_no; };
struct PlanG { int grid = 0; std::vector<std::vector<int>> members; // 6 x pixel idx
    std::vector<Win> wins_all, wins_no; };
struct Plan { int family = 0; PlanQ q; PlanG g; };

// gather helper: rows[expert] full-plane residuals -> per-group GCols
GCols gather(const std::vector<std::vector<int32_t>> &rows,
             const std::vector<int> &members) {
    TAcc _t(&TSet::gather);
    GCols gd; gd.cols.resize(rows.size());
    for (size_t e = 0; e < rows.size(); e++) {
        gd.cols[e].reserve(members.size());
        for (int n : members) gd.cols[e].push_back(rows[e][n]);
    }
    return gd;
}

Plan encode_channel_q(const std::vector<int> &key, const int32_t *rfM,
                      const std::vector<std::vector<int32_t>> &rows,
                      const std::vector<int> &elist, bool has_mlp, size_t N) {
    std::vector<int> uk = key;
    std::sort(uk.begin(), uk.end());
    uk.erase(std::unique(uk.begin(), uk.end()), uk.end());
    struct Cand { int64_t cost; int K; Groups groups; };
    std::vector<Cand> cands;
    for (int ci = 0; ci < NWIDE; ci++) {
        int K = WIDE_ARR[ci];
        Groups groups = quantile_groups(key, rfM, N, K);
        int gbits = 0; { int b = K - 1; while (b > 0) { gbits++; b >>= 1; } }
        int64_t mapside = 16 + 736 + (int64_t)uk.size() * gbits;
        // key -> group LUT
        std::vector<int> k2g(729, -1);
        for (size_t gi = 0; gi < groups.keylists.size(); gi++)
            for (int u : groups.keylists[gi]) k2g[u] = (int)gi;
        std::vector<std::vector<int>> members(groups.keylists.size());
        for (size_t n = 0; n < N; n++) members[k2g[key[n]]].push_back((int)n);
        std::vector<GCols> gvs;
        for (auto &m : members) gvs.push_back(gather(rows, m));
        DualRes dr; { TAcc _t(&TSet::qcoarse); dr = solve_groups_dual(gvs, elist, false, false); }
        // NOTE: coarse shortlist uses the WITH-MLP track only (driver takes
        // solve_groups_dual(...)[0]); GRID coarse below uses min instead.
        int64_t sub = dr.tot_all;
        cands.push_back({mapside + (int64_t)groups.keylists.size() * 6 + sub,
                         K, std::move(groups)});
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand &a, const Cand &b) { return a.cost < b.cost; });
    Plan best; int64_t bestcost = -1; bool have = false;
    for (int t = 0; t < 3 && t < (int)cands.size(); t++) {
        const Cand &cd = cands[t];
        std::vector<int> k2g(729, -1);
        for (size_t gi = 0; gi < cd.groups.keylists.size(); gi++)
            for (int u : cd.groups.keylists[gi]) k2g[u] = (int)gi;
        std::vector<std::vector<int>> members(cd.groups.keylists.size());
        for (size_t n = 0; n < N; n++) members[k2g[key[n]]].push_back((int)n);
        std::vector<GCols> gvs;
        for (auto &m : members) gvs.push_back(gather(rows, m));
        DualRes dr; { TAcc _t(&TSet::qrefine); dr = solve_groups_dual(gvs, elist, true); }
        int64_t cur = std::min(dr.tot_all, dr.tot_no);
        if (!have || cur < bestcost) {
            have = true; bestcost = cur;
            best.family = 0; best.q.K = cd.K; best.q.groups = cd.groups;
            best.q.wins_all = dr.wins_all; best.q.wins_no = dr.wins_no;
        }
    }
    (void)has_mlp;
    return best;
}

Plan encode_channel_grid(const std::vector<int32_t> &av,
                         const std::vector<int32_t> &bv,
                         const std::vector<int32_t> &cv,
                         const std::vector<std::vector<int32_t>> &rows,
                         const std::vector<int> &elist, size_t N) {
    std::vector<int> e(N);
    for (size_t n = 0; n < N; n++)
        e[n] = (int)(llabs((long)av[n] - cv[n]) + llabs((long)bv[n] - cv[n]));
    Plan best; int64_t bestcost = -1; bool have = false;
    int bestgi = 0;
    std::vector<std::vector<int>> bestmembers;
    for (int gi = 0; gi < 8; gi++) {
        std::vector<std::vector<int>> members(6);
        for (size_t n = 0; n < N; n++) {
            int g = 0; for (int t = 0; t < 5; t++) if (GRID_TH[gi][t] < e[n]) g++;
            members[g].push_back((int)n);
        }
        std::vector<GCols> gvs;
        for (auto &m : members) gvs.push_back(gather(rows, m));
        DualRes dr; { TAcc _t(&TSet::gcoarse); dr = solve_groups_dual(gvs, elist, false); }
        int64_t cur = 16 + 8 + (int64_t)6 * 6 + std::min(dr.tot_all, dr.tot_no);
        if (!have || cur < bestcost) {
            have = true; bestcost = cur; bestgi = gi; bestmembers = members; }
    }
    std::vector<GCols> gvs;
    for (auto &m : bestmembers) gvs.push_back(gather(rows, m));
    DualRes dr; { TAcc _t(&TSet::grefine); dr = solve_groups_dual(gvs, elist, true); }
    best.family = 1; best.g.grid = bestgi; best.g.members = bestmembers;
    best.g.wins_all = dr.wins_all; best.g.wins_no = dr.wins_no;
    return best;
}

// ---------- assembly ----------
struct AsmOut { std::vector<uint8_t> blob; int nmlp = 0; };

AsmOut assemble_channel(const Plan &plan, const PrepX &D,
                        const std::vector<int32_t> &ch,
                        const std::vector<int32_t> &wavg_res,
                        const std::vector<int32_t> &lms0_pred,
                        const std::vector<int32_t> &lms3_pred,
                        const int32_t *mlp_res, const std::vector<uint8_t> &mlp_side,
                        const WFit &wch, bool use_mlp, int H, int W) {
    size_t N = (size_t)H * W;
    const std::vector<Win> &wins = use_mlp
        ? (plan.family == 0 ? plan.q.wins_all : plan.g.wins_all)
        : (plan.family == 0 ? plan.q.wins_no : plan.g.wins_no);
    int ng = (int)wins.size();
    if (use_mlp) CHECK(!mlp_side.empty(), "MLP variant without side bytes");
    std::vector<int> expmap(N);
    std::vector<int32_t> resplane(N), symplane(N);
    std::vector<int> uk;
    int gbits = 0;
    if (plan.family == 0) {
        std::vector<uint8_t> cmap(729, 0);
        for (size_t gi = 0; gi < plan.q.groups.keylists.size(); gi++)
            for (int u : plan.q.groups.keylists[gi]) cmap[u] = (uint8_t)gi;
        for (size_t n = 0; n < N; n++) expmap[n] = cmap[D.key[n]];
        for (int gi = 0; gi < ng; gi++) {
            int ex = wins[gi].expert;
            for (size_t n = 0; n < N; n++) {
                if (expmap[n] != gi) continue;
                int32_t r;
                if (ex == EX_WAVG) r = wavg_res[n];
                else if (ex == EX_LMS0) r = ch[n] - lms0_pred[n];
                else if (ex == EX_LMS3) r = ch[n] - lms3_pred[n];
                else if (ex == EX_MLP) r = mlp_res[n];
                else r = ch[n] - D.P[ex][n];
                resplane[n] = r;
            }
        }
        check_range_plane(resplane, "Q residuals");
        for (size_t n = 0; n < N; n++)
            symplane[n] = D.s[n] ? resplane[n] * (int32_t)D.s[n] : resplane[n];
        check_range_plane(symplane, "Q flipped");
        uk = D.key;
        std::sort(uk.begin(), uk.end());
        uk.erase(std::unique(uk.begin(), uk.end()), uk.end());
        int b = plan.q.K - 1; while (b > 0) { gbits++; b >>= 1; }
    } else {
        for (int gi = 0; gi < 6; gi++)
            for (int n : plan.g.members[gi]) expmap[n] = gi;
        for (int gi = 0; gi < ng; gi++) {
            int ex = wins[gi].expert;
            for (int n : plan.g.members[gi]) {
                int32_t r;
                if (ex == EX_WAVG) r = wavg_res[n];
                else if (ex == EX_MLP) r = mlp_res[n];
                else r = ch[n] - D.P[ex][n];
                resplane[n] = r;
            }
        }
        check_range_plane(resplane, "GRID residuals");
        symplane = resplane;
    }
    std::vector<int> back(ng);
    std::vector<int64_t> counts(ng, 0);
    for (size_t n = 0; n < N; n++) counts[expmap[n]]++;
    for (int gi = 0; gi < ng; gi++) {
        back[gi] = wins[gi].backend;
        if (counts[gi] == 0) back[gi] = 0;
    }
    int nmlp = 0;
    for (int gi = 0; gi < ng; gi++)
        if (wins[gi].expert == EX_MLP && counts[gi] > 0) nmlp++;
    if (use_mlp) CHECK(nmlp > 0, "MLP variant with zero MLP groups");
    std::vector<uint8_t> blob;
    int kidx;
    if (plan.family == 0) {
        kidx = -1;
        for (int i = 0; i < NWIDE; i++) if (WIDE_ARR[i] == plan.q.K) kidx = i;
        CHECK(kidx >= 0, "bad K %d", plan.q.K);
    } else kidx = 15;
    blob.push_back((uint8_t)((plan.family & 1) | ((plan.g.grid & 7) << 1) |
                             ((kidx & 15) << 4)));
    blob.push_back((uint8_t)(ng & 0xFF));
    blob.push_back(use_mlp ? 1 : 0);
    if (use_mlp) blob.insert(blob.end(), mlp_side.begin(), mlp_side.end());
    { BitWriter bw;
      for (int g = 0; g < wch.ng; g++) bw.put(wch.use[g], 1);
      bw.flush(); blob.insert(blob.end(), bw.out.begin(), bw.out.end()); }
    { BitWriter bw;
      for (int g = 0; g < wch.ng; g++) if (wch.use[g])
          for (int t = 0; t < 4; t++) bw.put((uint32_t)(wch.w[(size_t)g * 4 + t] & 31), 5);
      bw.flush(); blob.insert(blob.end(), bw.out.begin(), bw.out.end()); }
    if (plan.family == 0) {
        std::vector<uint8_t> mask(729, 0);
        for (int u : uk) mask[u] = 1;
        for (int i = 0; i < 92; i++) { // np.packbits big-endian
            uint8_t byte = 0;
            for (int bit = 0; bit < 8; bit++) {
                int idx = i * 8 + bit;
                byte = (uint8_t)((byte << 1) | (idx < 729 && mask[idx] ? 1 : 0));
            }
            blob.push_back(byte);
        }
        std::vector<uint8_t> cmap(729, 0);
        for (size_t gi = 0; gi < plan.q.groups.keylists.size(); gi++)
            for (int u : plan.q.groups.keylists[gi]) cmap[u] = (uint8_t)gi;
        BitWriter bw;
        for (int u : uk) bw.put(cmap[u], gbits);
        bw.flush(); blob.insert(blob.end(), bw.out.begin(), bw.out.end());
    } else {
        int occ = 0;
        for (int gi = 0; gi < ng; gi++)
            if (counts[gi] > 0) occ |= (1 << gi);
        blob.push_back((uint8_t)(occ & 0xFF));
    }
    { BitWriter bw;
      for (int gi = 0; gi < ng; gi++) {
          bw.put((uint32_t)wins[gi].expert, 5); bw.put((uint32_t)back[gi], 2); }
      bw.flush(); blob.insert(blob.end(), bw.out.begin(), bw.out.end()); }
    { BitWriter bwk, bwb;
      for (int gi = 0; gi < ng; gi++) if (back[gi] == 1) {
          bwk.put((uint32_t)wins[gi].k, 4);
          bwb.put((uint32_t)(wins[gi].d & 7), 3); }
      bwk.flush(); bwb.flush();
      blob.insert(blob.end(), bwk.out.begin(), bwk.out.end());
      blob.insert(blob.end(), bwb.out.begin(), bwb.out.end()); }
    // Huffman tables
    std::vector<uint32_t> cd((size_t)ng * 2049, 0);
    std::vector<uint8_t> ln((size_t)ng * 2049, 0);
    for (int gi = 0; gi < ng; gi++) {
        if (back[gi] != 0 || counts[gi] == 0) continue;
        int64_t hist[2049] = {};
        for (size_t n = 0; n < N; n++)
            if (expmap[n] == gi) hist[symplane[n] + 1024]++;
        uint8_t ll[2049]; uint32_t cc[2049];
        canon_tables(hist, ll, cc);
        int A = 0;
        for (int s = 0; s < 2049; s++) if (ll[s]) A++;
        CHECK(A > 0, "empty Huffman table");
        for (int s = 0; s < 2049; s++) {
            cd[(size_t)gi * 2049 + s] = cc[s]; ln[(size_t)gi * 2049 + s] = ll[s]; }
        blob.push_back((uint8_t)(A & 0xFF)); blob.push_back((uint8_t)((A >> 8) & 0xFF));
        for (int s = 0; s < 2049; s++) if (ll[s]) {
            int sv = s - 1024;
            blob.push_back((uint8_t)(sv & 0xFF));
            blob.push_back((uint8_t)((sv >> 8) & 0xFF));
            blob.push_back(ll[s]);
        }
    }
    // H payload (raster order)
    std::vector<int16_t> hsyms; std::vector<uint8_t> hkeys;
    hsyms.reserve(N); hkeys.reserve(N);
    for (size_t n = 0; n < N; n++) if (back[expmap[n]] == 0) {
        int32_t v = symplane[n];
        CHECK(v >= -1024 && v <= 1024, "H payload range");
        hsyms.push_back((int16_t)v); hkeys.push_back((uint8_t)expmap[n]);
    }
    if (!hsyms.empty()) {
        size_t cap = (hsyms.size() * 32) / 8 + 1024;
        std::vector<uint8_t> buf(cap);
        size_t nn = pack_syms(hsyms.data(), hkeys.data(), (int)hsyms.size(),
                              cd.data(), ln.data(), buf.data(), cap);
        CHECK(nn > 0, "pack_syms failed");
        for (int i = 0; i < 4; i++) blob.push_back((uint8_t)((nn >> (8 * i)) & 0xFF));
        blob.insert(blob.end(), buf.begin(), buf.begin() + nn);
    } else for (int i = 0; i < 4; i++) blob.push_back(0);
    // G payload (raster order, bias-subtracted)
    std::vector<int> kbyexp(ng, 0), dbexp(ng, 0);
    for (int gi = 0; gi < ng; gi++) if (back[gi] == 1) {
        kbyexp[gi] = wins[gi].k; dbexp[gi] = wins[gi].d; }
    std::vector<int16_t> gvals; std::vector<uint8_t> gkeys;
    for (size_t n = 0; n < N; n++) if (back[expmap[n]] == 1) {
        int32_t v = symplane[n] - dbexp[expmap[n]];
        CHECK(v >= -1024 && v <= 1024, "G transmitted range");
        gvals.push_back((int16_t)v); gkeys.push_back((uint8_t)expmap[n]);
    }
    if (!gvals.empty()) {
        std::vector<uint8_t> kby(ng);
        for (int gi = 0; gi < ng; gi++) kby[gi] = (uint8_t)kbyexp[gi];
        size_t cap = gvals.size() * 300 + 64;
        std::vector<uint8_t> buf(cap);
        size_t nn = crown6_golomb_pack(gvals.data(), gkeys.data(), kby.data(),
                                       (int)gvals.size(), buf.data(), cap);
        CHECK(nn > 0, "golomb_pack failed");
        for (int i = 0; i < 4; i++) blob.push_back((uint8_t)((nn >> (8 * i)) & 0xFF));
        blob.insert(blob.end(), buf.begin(), buf.begin() + nn);
    } else for (int i = 0; i < 4; i++) blob.push_back(0);
    // rANS groups in gi order
    for (int gi = 0; gi < ng; gi++) if (back[gi] == 2) {
        const RansBlob &bl = wins[gi].blob;
        CHECK(!bl.pay.empty(), "empty rANS payload");
        int cnt = 0;
        for (size_t n = 0; n < N; n++) if (expmap[n] == gi) cnt++;
        CHECK(cnt > 0, "empty rANS group");
        for (int i = 0; i < 4; i++) blob.push_back((uint8_t)((cnt >> (8 * i)) & 0xFF));
        int Ad = (int)bl.syms.size();
        blob.push_back((uint8_t)(Ad & 0xFF)); blob.push_back((uint8_t)((Ad >> 8) & 0xFF));
        for (int i = 0; i < Ad; i++) {
            int sv = bl.syms[i] - 1024;
            blob.push_back((uint8_t)(sv & 0xFF));
            blob.push_back((uint8_t)((sv >> 8) & 0xFF));
            blob.push_back((uint8_t)(bl.freq[i] & 0xFF));
            blob.push_back((uint8_t)((bl.freq[i] >> 8) & 0xFF));
        }
        size_t nn = bl.pay.size();
        for (int i = 0; i < 4; i++) blob.push_back((uint8_t)((nn >> (8 * i)) & 0xFF));
        blob.insert(blob.end(), bl.pay.begin(), bl.pay.end());
    }
    return {blob, nmlp};
}

// ---------- MLP per-channel nets from .npz ----------
struct ChMLP { bool has = false; int nwin = 0;
    double qs[8]; uint8_t win[9]; int16_t qw[9][113];
    std::vector<int32_t> pred; std::vector<uint8_t> side; };

ChMLP load_channel_mlp(const std::string &wdir, const std::string &fn,
                       const std::string &rct, int ci,
                       const std::vector<int32_t> &plane, int H, int W) {
    ChMLP M;
    size_t N = (size_t)H * W;
    NpzNet nt;
    std::string path = wdir + "/crown6_" + fn + ".npz";
    if (!npz_load_ch(path, rct, ci, nt))
        DIE("missing .npz %s [%s ch%d]: train first with driver (MLP training is out of scope)",
            path.c_str(), rct.c_str(), ci);
    std::vector<double> qs_now; std::vector<int> ctx; std::vector<int> E;
    mlp_energy_ctx(plane.data(), H, W, qs_now, ctx, E);
    CHECK(memcmp(nt.qs, qs_now.data(), 64) == 0,
          "stale .npz thresholds for %s [%s ch%d] (quantile mismatch?)",
          fn.c_str(), rct.c_str(), ci);
    // edge-MED fallback plane
    M.pred.resize(N);
    for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
        int eL = (j > 0) ? plane[(size_t)i * W + j - 1]
                         : ((i > 0) ? plane[(size_t)(i - 1) * W] : 0);
        int eT = (i > 0) ? plane[(size_t)(i - 1) * W + j]
                         : ((j > 0) ? plane[(size_t)i * W + j - 1] : 0);
        int eTL = (i > 0 && j > 0) ? plane[(size_t)(i - 1) * W + j - 1] : 0;
        int mn = eL < eT ? eL : eT, mx = eL < eT ? eT : eL;
        M.pred[(size_t)i * W + j] =
            eTL >= mx ? mn : (eTL <= mn ? mx : eL + eT - eTL);
    }
    int nwin = 0;
    double W1[96], b1[8], W2[8];
    for (int k = 0; k < 9; k++) {
        M.win[k] = nt.win[k];
        if (!nt.win[k]) continue;
        CHECK(nt.scid[k] < 5, "bad scale id");
        double S = (double)MLP_SCALES[nt.scid[k]];
        for (int i = 0; i < 96; i++) W1[i] = (double)nt.qw[k][i] / S;
        for (int i = 0; i < 8; i++) b1[i] = (double)nt.qw[k][96 + i] / S;
        for (int i = 0; i < 8; i++) W2[i] = (double)nt.qw[k][104 + i] / S;
        double b2 = (double)nt.qw[k][112] / S;
        int fx[12];
        for (size_t n = 0; n < N; n++) {
            if (ctx[n] != k) continue;
            int i = (int)(n / W), j = (int)(n % W);
            mlp_causal_taps(plane.data(), H, W, i, j, fx);
            M.pred[n] = mlp_fwd_wire(fx, W1, b1, W2, b2);
        }
        nwin++;
    }
    memcpy(M.qs, nt.qs, 64);
    memcpy(M.qw, nt.qw, sizeof(M.qw));
    M.nwin = nwin;
    M.has = nwin > 0;
    if (M.has) {
        const uint8_t *qsb = (const uint8_t *)nt.qs;
        M.side.insert(M.side.end(), qsb, qsb + 64);
        for (int k = 0; k < 9; k++) M.side.push_back(nt.scid[k]);
        for (int k = 0; k < 9; k++) if (nt.win[k]) {
            const uint8_t *qb = (const uint8_t *)nt.qw[k];
            M.side.insert(M.side.end(), qb, qb + 226);
        }
    }
    return M;
}

} // anonymous namespace

int enc_main(int argc, char **argv);

} // namespace crown

int main(int argc, char **argv) { return crown::enc_main(argc, argv); }

namespace crown {
namespace {

struct RctCand { std::string name; int perm, t; };

int enc_main2(const std::string &png, const std::string &wdir,
              const std::string &outp) {
    auto t_all = std::chrono::steady_clock::now();
    std::vector<uint8_t> rgb; int H = 0, W = 0;
    CHECK(load_png_rgb(png, rgb, H, W), "cannot read PNG %s", png.c_str());
    CHECK(H >= 2 && W >= 2, "image too small %dx%d", H, W);
    std::string fn = png.substr(png.find_last_of('/') + 1);
    size_t N = (size_t)H * W;
    const RctCand RCTS[3] = {{"C6", 0, 6}, {"C27", 3, 6}, {"C12", 1, 5}};
    struct BlobInfo { std::vector<uint8_t> blob; std::string rct; };
    // Precompute RCT planes sequentially (cheap; asserts preserved).
    struct RctPlanes { RctCand rc; Img yc; };
    std::vector<RctPlanes> rcts;
    for (auto rc : RCTS) {
        Img yc;
        { TAcc _t(&TSet::rct); rct_fwd(rgb.data(), H, W, rc.perm, rc.t, yc); }
        std::vector<uint8_t> rt(N * 3);
        rct_inv_planes(yc.p[0].data(), yc.p[1].data(), yc.p[2].data(),
                       H, W, rc.perm, rc.t, rt.data());
        CHECK(rt == rgb, "RCT %s round-trip FAIL", rc.name.c_str());
        rcts.push_back({rc, std::move(yc)});
    }
    int jobs = 9;
    if (const char* je = getenv("CROWN_JOBS")) { int j = std::atoi(je); if (j >= 1) jobs = j; }
    // Per-(RCT,channel) pipeline: independent, bit-identical at any job count.
    // Join order fixed; RCT pick uses strict < so bytes identical sequential vs parallel.
    auto channel_job = [&](int ri, int c) -> std::vector<uint8_t> {
        const Img& yc = rcts[ri].yc;
        const RctCand& rc = rcts[ri].rc;
        const std::vector<int32_t>& ch = yc.p[c];
        WFit wf;
        { TAcc _t(&TSet::wfit); wf = weighted_fit_ch(ch.data(), H, W); }
        check_range_plane(wf.res, "WAVG res");
        std::vector<int32_t> l0, l3;
        { TAcc _t(&TSet::lms);
          lms_pred_plane(ch.data(), H, W, 0, l0);
          lms_pred_plane(ch.data(), H, W, 3, l3); }
        ChMLP m1;
        { TAcc _t(&TSet::mlp); m1 = load_channel_mlp(wdir, fn, rc.name, c, ch, H, W); }
        {
            PrepX D; { TAcc _t(&TSet::prep); D = prep_x(ch, H, W); }
            bool has_mlp = m1.has;
            std::vector<int32_t> mlp_res;
            if (has_mlp) {
                mlp_res.resize(N);
                for (size_t n = 0; n < N; n++) mlp_res[n] = ch[n] - m1.pred[n];
                check_range_plane(mlp_res, "MLP plain");
            }
            // Q rows: E16 plain + WAVG + LMS0 + LMS3 [+ MLP], flipped at gather
            std::vector<std::vector<int32_t>> qrows;
            std::vector<int> qelist;
            for (int x = 0; x < 16; x++) {
                std::vector<int32_t> r(N);
                for (size_t n = 0; n < N; n++) r[n] = ch[n] - D.P[x][n];
                qrows.push_back(std::move(r)); qelist.push_back(x);
            }
            qrows.push_back(wf.res); qelist.push_back(EX_WAVG);
            { std::vector<int32_t> r(N);
              for (size_t n = 0; n < N; n++) r[n] = ch[n] - l0[n];
              check_range_plane(r, "lms0");
              qrows.push_back(std::move(r)); qelist.push_back(EX_LMS0); }
            { std::vector<int32_t> r(N);
              for (size_t n = 0; n < N; n++) r[n] = ch[n] - l3[n];
              check_range_plane(r, "lms3");
              qrows.push_back(std::move(r)); qelist.push_back(EX_LMS3); }
            if (has_mlp) {
                qrows.push_back(mlp_res); qelist.push_back(EX_MLP);
                std::vector<int32_t> flip(N);
                for (size_t n = 0; n < N; n++) flip[n] = D.s[n] ? mlp_res[n] * D.s[n] : mlp_res[n];
                check_range_plane(flip, "MLP flip");
            }
            for (int x = 0; x < 16; x++) {
                std::vector<int32_t> flip(N);
                for (size_t n = 0; n < N; n++)
                    flip[n] = D.s[n] ? (ch[n] - D.P[x][n]) * D.s[n] : (ch[n] - D.P[x][n]);
                check_range_plane(flip, "flip E16");
            }
            // flipped Q rows for grouping/costing
            std::vector<std::vector<int32_t>> qfrows;
            for (size_t e = 0; e < qrows.size(); e++) {
                std::vector<int32_t> f(N);
                for (size_t n = 0; n < N; n++)
                    f[n] = D.s[n] ? qrows[e][n] * D.s[n] : qrows[e][n];
                qfrows.push_back(std::move(f));
            }
            std::vector<int32_t> rfM = qfrows[0]; // flipped MED
            // GRID rows: E16 plain + WAVG [+ MLP]
            std::vector<std::vector<int32_t>> grows;
            std::vector<int> gelist;
            for (int x = 0; x < 16; x++) {
                std::vector<int32_t> r(N);
                for (size_t n = 0; n < N; n++) r[n] = ch[n] - D.P[x][n];
                grows.push_back(std::move(r)); gelist.push_back(x);
            }
            grows.push_back(wf.res); gelist.push_back(EX_WAVG);
            if (has_mlp) { grows.push_back(mlp_res); gelist.push_back(EX_MLP); }
            for (int e = 0; e < 17; e++)
                check_range_plane(grows[e], "plain GRID");
            Plan pq = encode_channel_q(D.key, rfM.data(), qfrows, qelist, has_mlp, N);
            Plan pg = encode_channel_grid(D.av, D.bv, D.cv, grows, gelist, N);
            struct Var { size_t len; std::vector<uint8_t> blob; bool mlp; int nmlp; };
            auto variants = [&](const Plan &pl, bool isQ) {
                std::vector<Var> vv;
                // NOTE: Q uses wch with lms fields; GRID uses plain wfit.
                // lms0/l3 carry PREDICTIONS (lms_pred_plane contract); residuals
                // are formed here as ch[]-pred, mirroring wch["lms0"] in driver.
                // (use/w identical; only the Python dict wrapper differs).
                AsmOut a0; { TAcc _t(&TSet::asm_); a0 = assemble_channel(pl, D, ch, wf.res,
                    isQ ? l0 : std::vector<int32_t>(),
                    isQ ? l3 : std::vector<int32_t>(),
                    has_mlp ? mlp_res.data() : nullptr, m1.side,
                    wf, false, H, W); }
                vv.push_back({a0.blob.size(), a0.blob, false, 0});
                const std::vector<Win> &wa = pl.family == 0 ? pl.q.wins_all : pl.g.wins_all;
                bool anyMLP = false;
                for (auto &w : wa) if (w.expert == EX_MLP) anyMLP = true;
                if (has_mlp && anyMLP) {
                    AsmOut a1; { TAcc _t(&TSet::asm_); a1 = assemble_channel(pl, D, ch, wf.res,
                        isQ ? l0 : std::vector<int32_t>(),
                        isQ ? l3 : std::vector<int32_t>(),
                        mlp_res.data(), m1.side, wf, true, H, W); }
                    vv.push_back({a1.blob.size(), a1.blob, true, a1.nmlp});
                }
                // stable: noMLP first wins ties (strict < for MLP)
                Var bestv = vv[0];
                for (size_t i = 1; i < vv.size(); i++)
                    if (vv[i].len < bestv.len) bestv = vv[i];
                return bestv;
            };
            Var vq = variants(pq, true), vg = variants(pg, false);
            Var bestv = vq.len <= vg.len ? vq : vg; // Q wins ties (insertion order)
            if (const char* dd = getenv("CROWN_DUMPGROUPS")) {
                (void)dd;
                // dump_ggroups debug helper never landed in history; no-op to keep build green
            }
            T_merge(); // fold worker thread-local timers (mutex-protected)
            return bestv.blob;
        } // channel_job
    };
    // Dispatch 9 pipelines; fixed join order => bit-identical at any job count.
    std::vector<std::future<std::vector<uint8_t>>> futs(9);
    for (int ri = 0; ri < 3; ri++) {
        for (int c = 0; c < 3; c++) {
            int idx = ri * 3 + c;
            if (jobs <= 1)
                futs[idx] = std::async(std::launch::deferred, channel_job, ri, c);
            else
                futs[idx] = std::async(std::launch::async, channel_job, ri, c);
        }
    }
    std::vector<BlobInfo> cands;
    for (int ri = 0; ri < 3; ri++) {
        std::vector<uint8_t> out;
        out.push_back('C'); out.push_back('6');
        out.push_back((uint8_t)(H & 0xFF)); out.push_back((uint8_t)((H >> 8) & 0xFF));
        out.push_back((uint8_t)(W & 0xFF)); out.push_back((uint8_t)((W >> 8) & 0xFF));
        out.push_back(1);
        int rctid = rcts[ri].rc.name == "C6" ? 0 : (rcts[ri].rc.name == "C27" ? 1 : 2);
        out.push_back((uint8_t)rctid);
        for (int c = 0; c < 3; c++)
            { auto chblob = futs[ri * 3 + c].get(); out.insert(out.end(), chblob.begin(), chblob.end()); }
        cands.push_back({std::move(out), rcts[ri].rc.name});
    }
    size_t bi = 0; // strict <: first (C6) wins ties
    for (size_t i = 1; i < cands.size(); i++)
        if (cands[i].blob.size() < cands[bi].blob.size()) bi = i;
    FILE *f = fopen(outp.c_str(), "wb");
    CHECK(f, "cannot write %s", outp.c_str());
    CHECK(fwrite(cands[bi].blob.data(), 1, cands[bi].blob.size(), f) ==
          cands[bi].blob.size(), "write failed");
    fclose(f);
    double enc_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_all).count();
    double bpp = (double)cands[bi].blob.size() * 8.0 / (double)(N * 3);
    std::printf("%s: bytes=%zu bpp=%.4f rct=%s enc_s=%.1f\n",
                fn.c_str(), cands[bi].blob.size(), bpp,
                cands[bi].rct.c_str(), enc_s);
    if (getenv("CROWN_TIME")) {
        T_merge();
        std::fprintf(stderr, "TIME rct=%.2f wfit=%.2f lms=%.2f mlp=%.2f prep=%.2f "
            "qcoarse=%.2f qrefine=%.2f gcoarse=%.2f grefine=%.2f asm=%.2f | huff=%.2f golomb=%.2f rans=%.2f gather=%.2f (total %.1f)\n",
            T_all.rct, T_all.wfit, T_all.lms, T_all.mlp, T_all.prep,
            T_all.qcoarse, T_all.qrefine, T_all.gcoarse, T_all.grefine, T_all.asm_,
            T_all.huff, T_all.golomb, T_all.rans, T_all.gather, enc_s);
    }
    return 0;
}

} // anonymous namespace

int enc_main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s <in.png> <weights_dir> <out.bin>\n", argv[0]);
        return 2;
    }
    return enc_main2(argv[1], argv[2], argv[3]);
}

} // namespace crown
