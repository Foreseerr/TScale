#include "stdafx.h"
#include "citymurmur.h"

// Some primes between 2^63 and 2^64 for various uses.
static const ui64 k0 = 0xc3a5c85c97cb3127ULL;
static const ui64 k1 = 0xb492b66fbe98f273ULL;
static const ui64 k2 = 0x9ae16a3b2f90404fULL;
static const ui64 k3 = 0xc949d7c7509e6557ULL;

// unaligned load
static ui64 Fetch32(const char *p)
{
    return *(const ui32 *)p;
}

// unaligned load
static ui64 Fetch64(const char *p)
{
    return *(const ui64 *)p;
}

// Equivalent to Rotate(), but requires the second arg to be non-zero.
// On x86-64, and probably others, it's possible for this to compile
// to a single instruction if both args are already in registers.
static ui64 RotateByAtLeast1(ui64 val, int shift)
{
    return (val >> shift) | (val << (64 - shift));
}

static ui64 ShiftMix(ui64 val)
{
    return val ^ (val >> 47);
}

// Hash 128 input bits down to 64 bits of output.
// This is intended to be a reasonably good hash function.
inline ui64 Hash128to64(ui64 xLow, ui64 xHigh)
{
    // Murmur-inspired hashing.
    const ui64 kMul = 0x9ddfea08eb382d69ULL;
    ui64 a = ShiftMix((xLow ^ xHigh) * kMul);
    ui64 b = ShiftMix((xHigh ^ a) * kMul);
    b *= kMul;
    return b;
}


static ui64 HashLen0to16(const char *s, size_t len)
{
    if (len > 8) {
        ui64 a = Fetch64(s);
        ui64 b = Fetch64(s + len - 8);
        return Hash128to64(a, RotateByAtLeast1(b + len, len)) ^ b;
    }
    if (len >= 4) {
        ui64 a = Fetch32(s);
        return Hash128to64(len + (a << 3), Fetch32(s + len - 4));
    }
    if (len > 0) {
        ui8 a = s[0];
        ui8 b = s[len >> 1];
        ui8 c = s[len - 1];
        ui32 y = static_cast<ui32>(a) + (static_cast<ui32>(b) << 8);
        ui32 z = len + (static_cast<ui32>(c) << 2);
        return ShiftMix(y * k2 ^ z * k3) * k2;
    }
    return k2;
}



// A subroutine for CityHash128().  Returns a decent 128-bit hash for strings
// of any length representable in signed long.  Based on City and Murmur.
void CityMurmur(const void *pBuf, size_t len, ui64 seedLow, ui64 seedHigh, ui64 *pResLow, ui64 *pResHigh)
{
    const char *s = (const char *)pBuf;
    ui64 a = seedLow;
    ui64 b = seedHigh;
    ui64 c = 0;
    ui64 d = 0;
    signed long l = len - 16;
    if (l <= 0) {  // len <= 16
        a = ShiftMix(a * k1) * k1;
        c = b * k1 + HashLen0to16(s, len);
        d = ShiftMix(a + (len >= 8 ? Fetch64(s) : c));
    } else {  // len > 16
        c = Hash128to64(Fetch64(s + len - 8) + k1, a);
        d = Hash128to64(b + len, c + Fetch64(s + len - 16));
        a += d;
        do {
            a ^= ShiftMix(Fetch64(s) * k1) * k1;
            a *= k1;
            b ^= a;
            c ^= ShiftMix(Fetch64(s + 8) * k1) * k1;
            c *= k1;
            d ^= c;
            s += 16;
            l -= 16;
        } while (l > 0);
    }
    a = Hash128to64(a, c);
    b = Hash128to64(d, b);
    *pResLow = a ^ b;
    *pResHigh = Hash128to64(b, a);
}
