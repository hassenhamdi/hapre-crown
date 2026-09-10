// dump prep_x for kodim23 ch0 (C27 RCT) to compare vs Python prepX
#include <cstdio>
#include <vector>
#include <cstdint>
#include "codec.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int W = 0, H = 0, comp = 0;
    uint8_t *px = stbi_load("/tmp/opencode/autocompress/experiments/real_photos/kodim23.png", &W, &H, &comp, 3);
    if (!px) { printf("img load fail\n"); return 1; }
    size_t N = (size_t)H * W;
    crown::Img yc;
    crown::rct_fwd(px, H, W, 3, 6, yc);
    stbi_image_free(px);
    crown::PrepX D = crown::prep_x(yc.p[0], H, W);
    FILE *f = fopen("/tmp/prepx_cxx_ch0.bin", "wb");
    fwrite(D.av.data(), 4, N, f); fwrite(D.bv.data(), 4, N, f);
    fwrite(D.cv.data(), 4, N, f); fwrite(D.dv.data(), 4, N, f);
    for (int x = 0; x < 16; x++) fwrite(D.P[x].data(), 4, N, f);
    fwrite(D.key.data(), 4, N, f);
    std::vector<int32_t> ss(N); for (size_t n = 0; n < N; n++) ss[n] = D.s[n];
    fwrite(ss.data(), 4, N, f);
    fclose(f);
    printf("dumped H=%d W=%d N=%zu\n", H, W, N);
    return 0;
}
