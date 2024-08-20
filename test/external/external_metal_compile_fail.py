# AssertionError: Error Domain=AGXMetalG15X_B0 Code=3 "Compiler encountered an internal error"

src = """
#include <metal_stdlib>
using namespace metal;
kernel void r_64_32_8_16_4_6_6_4(device float* data0, const device float* data1,
                                 uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 64 */
  int lidx2 = lid.x; /* 8 */
  int gidx1 = gid.y; /* 32 */
  int lidx3 = lid.y; /* 16 */
  int alu0 = ((gidx0*4096)+(gidx1*16)+(lidx2*512)+lidx3);
  int alu1 = ((gidx0*147456)+(gidx1*576)+(lidx2*18432)+(lidx3*36));
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;
  float acc8 = 0.0f;
  float acc9 = 0.0f;
  float acc10 = 0.0f;
  float acc11 = 0.0f;
  float acc12 = 0.0f;
  float acc13 = 0.0f;
  float acc14 = 0.0f;
  float acc15 = 0.0f;
  float acc16 = 0.0f;
  float acc17 = 0.0f;
  float acc18 = 0.0f;
  float acc19 = 0.0f;
  float acc20 = 0.0f;
  float acc21 = 0.0f;
  float acc22 = 0.0f;
  float acc23 = 0.0f;
  float acc24 = 0.0f;
  float acc25 = 0.0f;
  float acc26 = 0.0f;
  float acc27 = 0.0f;
  float acc28 = 0.0f;
  float acc29 = 0.0f;
  float acc30 = 0.0f;
  float acc31 = 0.0f;
  float acc32 = 0.0f;
  float acc33 = 0.0f;
  float acc34 = 0.0f;
  float acc35 = 0.0f;
  for (int ridx0 = 0; ridx0 < 4; ridx0++) {
    int alu2 = (ridx0*6);
    int alu3 = (alu2+1);
    int alu4 = (alu2+2);
    int alu5 = (alu2+3);
    int alu6 = (alu2+4);
    int alu7 = (alu2+5);
    int alu8 = (alu2%7);
    int alu9 = ((alu8+1)%7);
    int alu10 = ((alu8+2)%7);
    int alu11 = ((alu8+3)%7);
    int alu12 = ((alu8+4)%7);
    int alu13 = ((alu8+5)%7);
    int alu14 = ((((alu0+(alu3/21))%262144)*144)+(((alu3/7)%3)*3)+(alu9*36));
    int alu15 = ((((alu0+(alu4/21))%262144)*144)+(((alu4/7)%3)*3)+(alu10*36));
    int alu16 = ((((alu0+(alu5/21))%262144)*144)+(((alu5/7)%3)*3)+(alu11*36));
    int alu17 = ((((alu0+(alu6/21))%262144)*144)+(((alu6/7)%3)*3)+(alu12*36));
    int alu18 = ((((alu0+(alu7/21))%262144)*144)+(((alu7/7)%3)*3)+(alu13*36));
    int alu19 = (alu8%7);
    int alu20 = ((((alu0+(alu2/21))%262144)*144)+(((alu2/7)%3)*3)+(alu19*36));
    bool alu21 = ((alu2<16)&(alu13<4));
    bool alu22 = ((alu2<17)&(alu12<4));
    bool alu23 = ((alu2<18)&(alu11<4));
    bool alu24 = ((alu2<19)&(alu10<4));
    bool alu25 = ((alu2<20)&(alu9<4));
    bool alu26 = ((alu2<21)&(alu19<4));
    float val0 = (alu25?*(data1+alu14+1):0.0f);
    float val1 = (alu25?*(data1+alu14+2):0.0f);
    float val2 = (alu25?*(data1+alu14+9):0.0f);
    float val3 = (alu25?*(data1+alu14+10):0.0f);
    float val4 = (alu25?*(data1+alu14+11):0.0f);
    float val5 = (alu25?*(data1+alu14+18):0.0f);
    float val6 = (alu25?*(data1+alu14+19):0.0f);
    float val7 = (alu25?*(data1+alu14+20):0.0f);
    float val8 = (alu25?*(data1+alu14+27):0.0f);
    float val9 = (alu25?*(data1+alu14+28):0.0f);
    float val10 = (alu25?*(data1+alu14+29):0.0f);
    float val11 = (alu24?*(data1+alu15+1):0.0f);
    float val12 = (alu24?*(data1+alu15+2):0.0f);
    float val13 = (alu24?*(data1+alu15+9):0.0f);
    float val14 = (alu24?*(data1+alu15+10):0.0f);
    float val15 = (alu24?*(data1+alu15+11):0.0f);
    float val16 = (alu24?*(data1+alu15+18):0.0f);
    float val17 = (alu24?*(data1+alu15+19):0.0f);
    float val18 = (alu24?*(data1+alu15+20):0.0f);
    float val19 = (alu24?*(data1+alu15+27):0.0f);
    float val20 = (alu24?*(data1+alu15+28):0.0f);
    float val21 = (alu24?*(data1+alu15+29):0.0f);
    float val22 = (alu23?*(data1+alu16+1):0.0f);
    float val23 = (alu23?*(data1+alu16+2):0.0f);
    float val24 = (alu23?*(data1+alu16+9):0.0f);
    float val25 = (alu23?*(data1+alu16+10):0.0f);
    float val26 = (alu23?*(data1+alu16+11):0.0f);
    float val27 = (alu23?*(data1+alu16+18):0.0f);
    float val28 = (alu23?*(data1+alu16+19):0.0f);
    float val29 = (alu23?*(data1+alu16+20):0.0f);
    float val30 = (alu23?*(data1+alu16+27):0.0f);
    float val31 = (alu23?*(data1+alu16+28):0.0f);
    float val32 = (alu23?*(data1+alu16+29):0.0f);
    float val33 = (alu22?*(data1+alu17+1):0.0f);
    float val34 = (alu22?*(data1+alu17+2):0.0f);
    float val35 = (alu22?*(data1+alu17+9):0.0f);
    float val36 = (alu22?*(data1+alu17+10):0.0f);
    float val37 = (alu22?*(data1+alu17+11):0.0f);
    float val38 = (alu22?*(data1+alu17+18):0.0f);
    float val39 = (alu22?*(data1+alu17+19):0.0f);
    float val40 = (alu22?*(data1+alu17+20):0.0f);
    float val41 = (alu22?*(data1+alu17+27):0.0f);
    float val42 = (alu22?*(data1+alu17+28):0.0f);
    float val43 = (alu22?*(data1+alu17+29):0.0f);
    float val44 = (alu21?*(data1+alu18+1):0.0f);
    float val45 = (alu21?*(data1+alu18+2):0.0f);
    float val46 = (alu21?*(data1+alu18+9):0.0f);
    float val47 = (alu21?*(data1+alu18+10):0.0f);
    float val48 = (alu21?*(data1+alu18+11):0.0f);
    float val49 = (alu21?*(data1+alu18+18):0.0f);
    float val50 = (alu21?*(data1+alu18+19):0.0f);
    float val51 = (alu21?*(data1+alu18+20):0.0f);
    float val52 = (alu21?*(data1+alu18+27):0.0f);
    float val53 = (alu21?*(data1+alu18+28):0.0f);
    float val54 = (alu21?*(data1+alu18+29):0.0f);
    float val55 = (alu26?*(data1+alu20+1):0.0f);
    float val56 = (alu26?*(data1+alu20+2):0.0f);
    float val57 = (alu26?*(data1+alu20+9):0.0f);
    float val58 = (alu26?*(data1+alu20+10):0.0f);
    float val59 = (alu26?*(data1+alu20+11):0.0f);
    float val60 = (alu26?*(data1+alu20+18):0.0f);
    float val61 = (alu26?*(data1+alu20+19):0.0f);
    float val62 = (alu26?*(data1+alu20+20):0.0f);
    float val63 = (alu26?*(data1+alu20+27):0.0f);
    float val64 = (alu26?*(data1+alu20+28):0.0f);
    float val65 = (alu26?*(data1+alu20+29):0.0f);
    float val66 = (alu25?*(data1+alu14):0.0f);
    float val67 = (alu24?*(data1+alu15):0.0f);
    float val68 = (alu23?*(data1+alu16):0.0f);
    float val69 = (alu22?*(data1+alu17):0.0f);
    float val70 = (alu21?*(data1+alu18):0.0f);
    float val71 = (alu26?*(data1+alu20):0.0f);
    acc0 = (acc0+val71);
    acc1 = (acc1+val66);
    acc2 = (acc2+val67);
    acc3 = (acc3+val68);
    acc4 = (acc4+val69);
    acc5 = (acc5+val70);
    acc6 = (acc6+val57+val55);
    acc7 = (acc7+val2+val0);
    acc8 = (acc8+val13+val11);
    acc9 = (acc9+val24+val22);
    acc10 = (acc10+val35+val33);
    acc11 = (acc11+val46+val44);
    acc12 = (acc12+val60+val58+val56);
    acc13 = (acc13+val5+val3+val1);
    acc14 = (acc14+val16+val14+val12);
    acc15 = (acc15+val27+val25+val23);
    acc16 = (acc16+val38+val36+val34);
    acc17 = (acc17+val49+val47+val45);
    acc18 = (acc18+val63+val61+val59);
    acc19 = (acc19+val8+val6+val4);
    acc20 = (acc20+val19+val17+val15);
    acc21 = (acc21+val30+val28+val26);
    acc22 = (acc22+val41+val39+val37);
    acc23 = (acc23+val52+val50+val48);
    acc24 = (acc24+val64+val62);
    acc25 = (acc25+val9+val7);
    acc26 = (acc26+val20+val18);
    acc27 = (acc27+val31+val29);
    acc28 = (acc28+val42+val40);
    acc29 = (acc29+val53+val51);
    acc30 = (acc30+val65);
    acc31 = (acc31+val10);
    acc32 = (acc32+val21);
    acc33 = (acc33+val32);
    acc34 = (acc34+val43);
    acc35 = (acc35+val54);
  }
  *(data0+alu1+1) = acc6;
  *(data0+alu1+2) = acc12;
  *(data0+alu1+3) = acc18;
  *(data0+alu1+4) = acc24;
  *(data0+alu1+5) = acc30;
  *(data0+alu1+6) = acc1;
  *(data0+alu1+7) = acc7;
  *(data0+alu1+8) = acc13;
  *(data0+alu1+9) = acc19;
  *(data0+alu1+10) = acc25;
  *(data0+alu1+11) = acc31;
  *(data0+alu1+12) = acc2;
  *(data0+alu1+13) = acc8;
  *(data0+alu1+14) = acc14;
  *(data0+alu1+15) = acc20;
  *(data0+alu1+16) = acc26;
  *(data0+alu1+17) = acc32;
  *(data0+alu1+18) = acc3;
  *(data0+alu1+19) = acc9;
  *(data0+alu1+20) = acc15;
  *(data0+alu1+21) = acc21;
  *(data0+alu1+22) = acc27;
  *(data0+alu1+23) = acc33;
  *(data0+alu1+24) = acc4;
  *(data0+alu1+25) = acc10;
  *(data0+alu1+26) = acc16;
  *(data0+alu1+27) = acc22;
  *(data0+alu1+28) = acc28;
  *(data0+alu1+29) = acc34;
  *(data0+alu1+30) = acc5;
  *(data0+alu1+31) = acc11;
  *(data0+alu1+32) = acc17;
  *(data0+alu1+33) = acc23;
  *(data0+alu1+34) = acc29;
  *(data0+alu1+35) = acc35;
  *(data0+alu1) = acc0;
}
"""

from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalProgram

if __name__ == "__main__":
  dev = MetalDevice("METAL")
  lib = MetalCompiler(dev).compile(src)
  prg = MetalProgram(dev, "r_64_32_8_16_4_6_6_4", lib)

