// CROWN6 C++ decoder (NEW file). Owns scan/format logic; calls READ-ONLY
// linked C ops for rANS decode, the causal channel scan (crown6_decode_ch)
// and RCT inverse. Framing is asserted exactly (p == len).
//
// Usage: crown_dec <in.bin> <out.rgb>   (raw RGB bytes, H*W*3)
#include "codec.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace crown {
namespace {

int dec_main2(const std::string &inp, const std::string &outp) {
    auto t_all = std::chrono::steady_clock::now();
    FILE *f = fopen(inp.c_str(), "rb");
    CHECK(f, "cannot read %s", inp.c_str());
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    CHECK(sz > 8, "truncated stream");
    std::vector<uint8_t> blob(sz);
    CHECK(fread(blob.data(), 1, sz, f) == (size_t)sz, "read failed");
    fclose(f);
    CHECK(blob[0] == 'C' && blob[1] == '6', "bad magic %c%c", blob[0], blob[1]);
    int H = blob[2] | (blob[3] << 8), W = blob[4] | (blob[5] << 8);
    CHECK(H >= 2 && W >= 2, "bad dims %dx%d", H, W);
    CHECK(blob[6] == 1, "bad version %d", blob[6]);
    int rct = blob[7] & 3;
    CHECK(rct <= 2, "bad rct %d", rct);
    size_t N = (size_t)H * W;
    size_t p = 8;
    std::vector<int16_t> planes[3];
    for (int ch = 0; ch < 3; ch++) {
        CHECK(p + 3 <= blob.size(), "truncated chdr");
        uint8_t b0 = blob[p], ng = blob[p + 1], mlp_flag = blob[p + 2];
        p += 3;
        CHECK(ng >= 1 && ng <= 64, "bad ng %d", ng);
        CHECK(mlp_flag <= 1, "bad mlp flag %d", mlp_flag);
        int fam = b0 & 1, grid = (b0 >> 1) & 7;
        if (getenv("CROWN_DDEC")) fprintf(stderr, "DDEC ch=%d fam=%d grid=%d ng=%d mlp=%d\n",
            ch, fam, grid, ng, mlp_flag);
        std::vector<uint8_t> cmap(729, 0);
        int grid_thr[5] = {0, 0, 0, 0, 0};
        int occ = 0;
        std::vector<int> qmask;
        double mlp_thr[8] = {0};
        uint8_t mlp_win[9] = {0};
        double mlp_w[9 * 113] = {0};
        if (mlp_flag) {
            CHECK(p + 64 + 9 <= blob.size(), "truncated mlp side");
            memcpy(mlp_thr, blob.data() + p, 64); p += 64;
            const uint8_t *scid = blob.data() + p; p += 9;
            for (int k = 0; k < 9; k++) {
                if (scid[k] != 255) {
                    CHECK(scid[k] < 5, "bad scale id %d", scid[k]);
                    CHECK(p + 226 <= blob.size(), "truncated mlp weights");
                    int16_t qw[113];
                    memcpy(qw, blob.data() + p, 226); p += 226;
                    double S = (double)MLP_SCALES[scid[k]];
                    for (int i = 0; i < 113; i++)
                        mlp_w[(size_t)k * 113 + i] = (double)qw[i] / S;
                    mlp_win[k] = 1;
                }
            }
            int nw = 0; for (int k = 0; k < 9; k++) nw += mlp_win[k];
            CHECK(nw > 0, "mlp_present with zero nets (framing)");
        }
        int nbw = (W + GS - 1) / GS, nbh = (H + GS - 1) / GS, ng32 = nbh * nbw;
        size_t nbytes = ((size_t)ng32 + 7) / 8;
        CHECK(p + nbytes <= blob.size(), "truncated wuse");
        std::vector<uint8_t> wuse(ng32, 0);
        { BitReader br(blob.data() + p, nbytes); p += nbytes;
          for (int g = 0; g < ng32; g++) wuse[g] = (uint8_t)br.get(1); }
        int nused = 0; for (auto v : wuse) nused += v;
        nbytes = ((size_t)nused * 20 + 7) / 8;
        std::vector<int8_t> ww((size_t)ng32 * 4, 0);
        if (nbytes) {
            CHECK(p + nbytes <= blob.size(), "truncated wts");
            BitReader br(blob.data() + p, nbytes); p += nbytes;
            for (int g = 0; g < ng32; g++) if (wuse[g])
                for (int t = 0; t < 4; t++) {
                    int v = (int)br.get(5);
                    ww[(size_t)g * 4 + t] = (int8_t)(v >= 16 ? v - 32 : v);
                }
        }
        if (fam == 0) {
            CHECK(p + 92 <= blob.size(), "truncated qmask");
            int kidx = (b0 >> 4) & 15;
            CHECK(kidx >= 0 && kidx < NWIDE, "bad kidx %d", kidx);
            int K = WIDE_ARR[kidx], gbits = 0, b = K - 1;
            while (b > 0) { gbits++; b >>= 1; }
            qmask.assign(729, 0);
            for (int i = 0; i < 92; i++)
                for (int bit = 7; bit >= 0; bit--) {
                    int u = i * 8 + (7 - bit);
                    if (u < 729) qmask[u] = (blob[p + i] >> bit) & 1;
                }
            p += 92;
            int na = 0; for (auto v : qmask) na += v;
            nbytes = ((size_t)na * gbits + 7) / 8;
            CHECK(p + nbytes <= blob.size(), "truncated qmap");
            BitReader br(blob.data() + p, nbytes); p += nbytes;
            for (int u = 0; u < 729; u++) if (qmask[u]) {
                int g = (int)br.get(gbits);
                CHECK(g >= 0 && g < ng, "bad group id %d", g);
                cmap[u] = (uint8_t)g;
            }
        } else {
            CHECK(p + 1 <= blob.size(), "truncated occ");
            occ = blob[p++];
            CHECK(grid >= 0 && grid < 8, "bad grid %d", grid);
            for (int t = 0; t < 5; t++) grid_thr[t] = GRID_TH[grid][t];
        }
        nbytes = ((size_t)ng * 7 + 7) / 8;
        CHECK(p + nbytes <= blob.size(), "truncated meta");
        std::vector<uint8_t> predid(128, 0), backend(128, 0);
        { BitReader br(blob.data() + p, nbytes); p += nbytes;
          for (int gi = 0; gi < ng; gi++) {
              int v = (int)br.get(7);
              predid[gi] = (uint8_t)((v >> 2) & 31);
              backend[gi] = (uint8_t)(v & 3);
              CHECK(predid[gi] <= 19 && backend[gi] <= 2, "bad meta g%d", gi);
              if (predid[gi] == 19)
                  CHECK(mlp_flag == 1, "MLP predid without mlp_present g%d", gi);
          } }
        int nG = 0;
        for (int gi = 0; gi < ng; gi++) if (backend[gi] == 1) nG++;
        std::vector<uint8_t> kvals(128, 0);
        std::vector<int8_t> dbias(128, 0);
        nbytes = ((size_t)nG * 4 + 7) / 8;
        if (nbytes) {
            CHECK(p + nbytes <= blob.size(), "truncated kvals");
            BitReader br(blob.data() + p, nbytes); p += nbytes;
            for (int gi = 0; gi < ng; gi++) if (backend[gi] == 1)
                kvals[gi] = (uint8_t)br.get(4);
        }
        nbytes = ((size_t)nG * 3 + 7) / 8;
        if (nbytes) {
            CHECK(p + nbytes <= blob.size(), "truncated dbias");
            BitReader br(blob.data() + p, nbytes); p += nbytes;
            for (int gi = 0; gi < ng; gi++) if (backend[gi] == 1) {
                int v = (int)br.get(3);
                dbias[gi] = (int8_t)(v >= 4 ? v - 8 : v);
            }
        }
        std::vector<int> nonempty(ng, 0);
        if (fam == 0) {
            for (int u = 0; u < 729; u++)
                if (qmask[u]) nonempty[cmap[u]] = 1;
        } else {
            for (int gi = 0; gi < ng; gi++) nonempty[gi] = (occ >> gi) & 1;
        }
        // Huffman canonical tables (same order rule as encoder)
        std::vector<uint32_t> cd((size_t)128 * 2049, 0);
        std::vector<uint8_t> ln((size_t)128 * 2049, 0);
        for (int gi = 0; gi < ng; gi++) {
            if (backend[gi] != 0 || !nonempty[gi]) continue;
            CHECK(p + 2 <= blob.size(), "truncated H table hdr");
            int A = blob[p] | (blob[p + 1] << 8); p += 2;
            CHECK(A > 0 && p + (size_t)A * 3 <= blob.size(), "truncated H table");
            std::vector<int> lensv(2049, 0);
            for (int i = 0; i < A; i++) {
                int sv = blob[p] | (blob[p + 1] << 8);
                int ll = blob[p + 2]; p += 3;
                int s = (sv + 1024) & 0xFFFF;
                CHECK(s >= 0 && s < 2049 && ll > 0, "bad H entry");
                lensv[s] = ll;
            }
            std::vector<int> syms;
            for (int s = 0; s < 2049; s++) if (lensv[s]) syms.push_back(s);
            std::sort(syms.begin(), syms.end(),
                      [&](int x, int y) { return lensv[x] < lensv[y] ||
                          (lensv[x] == lensv[y] && x < y); });
            uint32_t code = 0; int prev = 0;
            for (int s : syms) {
                code <<= (lensv[s] - prev);
                cd[(size_t)gi * 2049 + s] = code;
                ln[(size_t)gi * 2049 + s] = (uint8_t)lensv[s];
                code++; prev = lensv[s];
            }
        }
        CHECK(p + 4 <= blob.size(), "truncated H len");
        uint32_t hn = (uint32_t)blob[p] | ((uint32_t)blob[p+1] << 8) |
                      ((uint32_t)blob[p+2] << 16) | ((uint32_t)blob[p+3] << 24);
        p += 4;
        CHECK(p + hn <= blob.size(), "truncated H payload");
        const uint8_t *hpay = hn ? blob.data() + p : nullptr; p += hn;
        CHECK(p + 4 <= blob.size(), "truncated G len");
        uint32_t gn = (uint32_t)blob[p] | ((uint32_t)blob[p+1] << 8) |
                      ((uint32_t)blob[p+2] << 16) | ((uint32_t)blob[p+3] << 24);
        p += 4;
        CHECK(p + gn <= blob.size(), "truncated G payload");
        const uint8_t *gpay = gn ? blob.data() + p : nullptr; p += gn;
        // rANS groups: pre-decode via linked rans_decode
        std::vector<int16_t> rsyms;
        std::vector<int32_t> goff; goff.push_back(0);
        for (int gi = 0; gi < ng; gi++) {
            if (backend[gi] != 2) { goff.push_back((int)rsyms.size()); continue; }
            CHECK(p + 4 <= blob.size(), "truncated R cnt");
            int cnt = (int)((uint32_t)blob[p] | ((uint32_t)blob[p+1] << 8) |
                            ((uint32_t)blob[p+2] << 16) | ((uint32_t)blob[p+3] << 24));
            p += 4;
            CHECK(p + 2 <= blob.size(), "truncated R alpha");
            int A = blob[p] | (blob[p + 1] << 8); p += 2;
            CHECK(A > 0 && p + (size_t)A * 4 <= blob.size(), "truncated R table");
            std::vector<int32_t> syms32(A);
            std::vector<uint16_t> freq(A);
            std::vector<int32_t> cum(A);
            int c = 0;
            for (int i = 0; i < A; i++) {
                int sv = blob[p] | (blob[p + 1] << 8);
                int fq = blob[p + 2] | (blob[p + 3] << 8); p += 4;
                syms32[i] = (sv + 1024) & 0xFFFF;
                freq[i] = (uint16_t)fq; cum[i] = c; c += fq;
            }
            CHECK(p + 4 <= blob.size(), "truncated R paylen");
            uint32_t rn = (uint32_t)blob[p] | ((uint32_t)blob[p+1] << 8) |
                          ((uint32_t)blob[p+2] << 16) | ((uint32_t)blob[p+3] << 24);
            p += 4;
            CHECK(p + rn <= blob.size(), "truncated R payload");
            const uint8_t *pay = rn ? blob.data() + p : nullptr; p += rn;
            std::vector<int32_t> slots(16384);
            rans_slots(freq.data(), cum.data(), A, slots.data());
            size_t base = rsyms.size();
            rsyms.resize(base + cnt);
            int rc = rans_decode(pay, rn, cnt, syms32.data(), freq.data(),
                                 cum.data(), A, slots.data(), rsyms.data() + base);
            CHECK(rc == 0, "rANS decode FAIL ch%d g%d", ch, gi);
            goff.push_back((int)rsyms.size());
        }
        std::vector<uint8_t> hasn(128, 0);
        for (int gi = 0; gi < ng; gi++)
            hasn[gi] = (backend[gi] == 0 && nonempty[gi]) ? 1 : 0;
        planes[ch].assign(N, 0);
        static uint8_t empty8[1] = {0};
        int rc = crown6_decode_ch(hn ? hpay : empty8, hn, gn ? gpay : empty8, gn,
            rsyms.empty() ? nullptr : rsyms.data(), goff.data(), ng,
            cd.data(), ln.data(), cmap.data(), predid.data(), backend.data(),
            kvals.data(), dbias.data(), hasn.data(), wuse.data(), ww.data(),
            nbw, fam, grid_thr, planes[ch].data(), H, W, mlp_flag, mlp_thr,
            mlp_win, mlp_w);
        CHECK(rc == 0, "channel decode FAIL ch%d rc=%d", ch, rc);
    }
    CHECK(p == blob.size(), "framing: p=%zu len=%zu", p, blob.size());
    std::vector<uint8_t> rgb(N * 3);
    int rc = crown6_rct_inv(planes[0].data(), planes[1].data(), planes[2].data(),
                            H, W, rct, rgb.data());
    CHECK(rc == 0, "RCT inv range FAIL rc=%d", rc);
    FILE *o = fopen(outp.c_str(), "wb");
    CHECK(o, "cannot write %s", outp.c_str());
    CHECK(fwrite(rgb.data(), 1, rgb.size(), o) == rgb.size(), "write failed");
    fclose(o);
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_all).count();
    std::printf("dec: %dx%d rct=%d bytes=%ld dec_ms=%.1f\n", H, W, rct, sz, ms);
    return 0;
}

} // anonymous namespace

int dec_main(int argc, char **argv);
} // namespace crown

int main(int argc, char **argv) { return crown::dec_main(argc, argv); }

namespace crown {
int dec_main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <in.bin> <out.rgb>\n", argv[0]);
        return 2;
    }
    return dec_main2(argv[1], argv[2]);
}
} // namespace crown