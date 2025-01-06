#pragma once

struct TGuid
{
    union {
        ui32 dw[4];
        ui64 ll[2];
    };

    TGuid() { Zero(*this); }
    TGuid(ui32 a, ui32 b, ui32 c, ui32 d)
    {
        dw[0] = a;
        dw[1] = b;
        dw[2] = c;
        dw[3] = d;
    }
    bool IsEmpty() const { return (ll[0] | ll[1]) == 0; }
};

inline bool operator==(const TGuid &a, const TGuid &b) { return memcmp(&a, &b, sizeof(a)) == 0; }
inline bool operator!=(const TGuid &a, const TGuid &b) { return !(a == b); }

template<> struct nstl::hash<TGuid>
{
    size_t operator()(const TGuid &a) const
    {
        return a.ll[0] + a.ll[1];
    }
};

void CreateGuid(TGuid *res);
TString GetGuidAsString(const TGuid &g);
TString CreateGuidAsString();
TGuid GetGuid(const TString &s);
