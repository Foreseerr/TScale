#pragma once

#define NN 312
#define MM 156
#define MATRIX_A ULL(0xB5026F5AA96619E9)
#define UM ULL(0xFFFFFFFF80000000)
#define LM ULL(0x7FFFFFFF)

namespace NPrivate {
    inline static double ToRes53(ui64 v) {
        return double(v * (1.0 / 18446744073709551616.0L));
    }

    class TMersenne64 {
        public:
            inline TMersenne64(ui64 s = ULL(19650218))
                : mti(NN + 1)
            {
                InitGenRand(s);
            }

            inline TMersenne64(const ui64 keys[], size_t len) throw ()
                : mti(NN + 1)
            {
                InitByArray(keys, len);
            }

            inline ui64 GenRand() {
                int i;
                ui64 x;
                ui64 mag01[2] = {
                    ULL(0),
                    MATRIX_A
                };

                if (mti >= NN) {
                    if (mti == NN + 1) {
                        InitGenRand(ULL(5489));
                    }

                    for (i = 0; i < NN - MM; ++i) {
                        x = (mt[i] & UM) | (mt[i + 1] & LM);
                        mt[i] = mt[i + MM] ^ (x >> 1) ^ mag01[(int)(x & ULL(1))];
                    }

                    for (; i < NN - 1; ++i) {
                        x = (mt[i] & UM) | (mt[i + 1] & LM);
                        mt[i] = mt[i + (MM - NN) ] ^ (x >> 1) ^ mag01[(int)(x & ULL(1))];
                    }

                    x = (mt[NN - 1] & UM) | (mt[0] & LM);
                    mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ mag01[(int)(x & ULL(1))];

                    mti = 0;
                }

                x = mt[mti++];

                x ^= (x >> 29) & ULL(0x5555555555555555);
                x ^= (x << 17) & ULL(0x71D67FFFEDA60000);
                x ^= (x << 37) & ULL(0xFFF7EEE000000000);
                x ^= (x >> 43);

                return x;
            }

            inline ui64 GenRand64() {
                return GenRand();
            }

            inline i64 GenRandSigned() {
                return (i64)(GenRand() >> 1);
            }

            inline double GenRandReal1() {
                return (GenRand() >> 11) * (1.0 / 9007199254740991.0);
            }

            inline double GenRandReal2() {
                return (GenRand() >> 11) * (1.0 / 9007199254740992.0);
            }

            inline double GenRandReal3() {
                return ((GenRand() >> 12) + 0.5) * (1.0 / 4503599627370496.0);
            }

            inline double GenRandReal4() {
                return ToRes53(GenRand());
            }

        private:
            inline void InitGenRand(ui64 seed) {
                mt[0] = seed;

                for (mti = 1; mti < NN; ++mti) {
                    mt[mti] = (ULL(6364136223846793005) * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti);
                }
            }

            inline void InitByArray(const ui64 init_key[], size_t key_length) {
                ui64 i = 1;
                ui64 j = 0;
                ui64 k;

                InitGenRand(ULL(19650218));

                k = NN > key_length ? NN : key_length;

                for (; k; --k) {
                    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * ULL(3935559000370003845))) + init_key[j] + j;

                    ++i;
                    ++j;

                    if (i >= NN) {
                        mt[0] = mt[NN - 1];
                        i = 1;
                    }

                    if (j >= key_length) {
                        j = 0;
                    }
                }

                for (k = NN - 1; k; --k) {
                    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 62)) * ULL(2862933555777941757))) - i;

                    ++i;

                    if (i >= NN) {
                        mt[0] = mt[NN - 1];
                        i = 1;
                    }
                }

                mt[0] = ULL(1) << 63;
            }

        private:
            ui64 mt[NN];
            int mti;
    };
}

#undef NN
#undef MM
#undef MATRIX_A
#undef UM
#undef LM


//////////////////////////////////////////////////////////////////////////
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

namespace NPrivate {
    inline static double ToRes53Mix(ui32 x, ui32 y) {
        return ToRes53((ui64)x | ((ui64)y << 32));
    }

    class TMersenne32 {
    public:
        inline TMersenne32(ui32 s  = 19650218UL) throw ()
            : mti(N + 1)
        {
            InitGenRand(s);
        }

        inline TMersenne32(const ui32 init_key[], size_t key_length) throw ()
            : mti(N + 1)
        {
            InitByArray(init_key, key_length);
        }

        inline ui32 GenRand() {
            ui32 y;
            ui32 mag01[2] = { 0x0UL, MATRIX_A };

            if (mti >= N) {
                int kk;

                if (mti == N + 1) {
                    InitGenRand(5489UL);
                }

                for (kk = 0; kk < N - M; ++kk) {
                    y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                    mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
                }

                for (; kk < N - 1; ++kk) {
                    y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                    mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
                }

                y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
                mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

                mti = 0;
            }

            y = mt[mti++];

            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680UL;
            y ^= (y << 15) & 0xefc60000UL;
            y ^= (y >> 18);

            return y;
        }

        inline ui64 GenRand64() {
            ui64 a = GenRand();
            ui64 b = GenRand();
            return (a << 32) + b;
        }

        inline i32 GenRandSigned() {
            return (i32)(GenRand() >> 1);
        }

        inline double GenRandReal1() {
            return GenRand() * (1.0 / 4294967295.0);
        }

        inline double GenRandReal2() {
            return GenRand() * (1.0 / 4294967296.0);
        }

        inline double GenRandReal3() {
            return ((double)GenRand() + 0.5) * (1.0 / 4294967296.0);
        }

        inline double GenRandReal4() {
            const ui32 x = GenRand();
            const ui32 y = GenRand();

            return ToRes53Mix(x, y);
        }

    private:
        inline void InitGenRand(ui32 s) {
            mt[0] = s;

            for (mti = 1; mti < N; ++mti) {
                mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
            }
        }

        inline void InitByArray(const ui32 init_key[], size_t key_length) {
            InitGenRand(19650218UL);

            ui32 i = 1;
            ui32 j = 0;
            ui32 k = ui32(N > key_length ? N : key_length);

            for (; k; k--) {
                mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL)) + init_key[j] + j;

                ++i;
                ++j;

                if (i >= N) {
                    mt[0] = mt[N - 1];
                    i = 1;
                }

                if (j >= key_length) {
                    j = 0;
                }
            }

            for (k = N - 1; k; k--) {
                mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941UL)) - i;

                ++i;

                if (i >= N) {
                    mt[0] = mt[N - 1];
                    i = 1;
                }
            }

            mt[0] = 0x80000000UL;
        }

    private:
        ui32 mt[N];
        int mti;
    };
}

#undef N
#undef M
#undef MATRIX_A
#undef UPPER_MASK
#undef LOWER_MASK

//////////////////////////////////////////////////////////////////////////
namespace NPrivate {
    template <class T>
    struct TMersenneTraits;

    template <>
    struct TMersenneTraits<ui64> {
        typedef TMersenne64 TRealization;

        enum {
            Seed = ULL(19650218)
        };
    };

    template <>
    struct TMersenneTraits<ui32> {
        typedef TMersenne32 TRealization;

        enum {
            Seed = 19650218UL
        };
    };
}

template <class T>
class TMersenne {
public:
    inline TMersenne(T seed = ::NPrivate::TMersenneTraits<T>::Seed) throw ()
        : R_(seed)
    {
    }

    inline TMersenne(const T keys[], size_t len) throw ()
        : R_(keys, len)
    {
    }

    inline T GenRand() {
        return R_.GenRand();
    }

    inline T RandMax() {
        return Max<T>();
    }

    /* generates a random number on [0, 1]-real-interval */
    inline double GenRandReal1() {
        return R_.GenRandReal1();
    }

    /* generates a random number on [0, 1)-real-interval */
    inline double GenRandReal2() {
        return R_.GenRandReal2();
    }

    /* generates a random number on (0, 1)-real-interval */
    inline double GenRandReal3() {
        return R_.GenRandReal3();
    }

    /* generates a random number on [0, 1) with 53-bit resolution */
    inline double GenRandReal4() {
        return R_.GenRandReal4();
    }

    inline ui64 Uniform(ui64 n)
    {
        Y_ASSERT(n < 0x1fffffffffffffffull);
        ui64 new_rand;
        ui64 limit = (0xffffffffffffffffULL / n) * n;
        do {
            new_rand = R_.GenRand64();
        } while (new_rand >= limit);
        return new_rand % n;
    }

private:
    typename ::NPrivate::TMersenneTraits<T>::TRealization R_;
};
