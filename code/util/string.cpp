#include "stdafx.h"
#include "string.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
//
ui8 Utf8CodeLength[256] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 255, 255, 255, 255, 255, 255, 255, 255,
};

//void PrintTable()
//{
//    for (yint n1 = 0; n1 < 256; n1 += 16) {
//        for (yint n2 = n1; n2 < n1 + 16; ++n2) {
//            if (n2 & 0x80) {
//                if (n2 & 0x40) {
//                    if (n2 & 0x20) {
//                        if (n2 & 0x10) {
//                            if (n2 & 8) {
//                                DebugPrintf("255, "); // invalid character, can not be part of utf8 encoded text
//                            } else {
//                                DebugPrintf("4, ");
//                            }
//                        } else {
//                            DebugPrintf("3, ");
//                        }
//                    } else {
//                        DebugPrintf("2, ");
//                    }
//                } else {
//                    DebugPrintf("255, "); // not first octet of the character (10xxxxxx octects);
//                }
//            } else {
//                DebugPrintf("1, ");
//            }
//        }
//        DebugPrintf("\n");
//    }
//}

// latin letters
//41..5a
//61..7a
static bool IsUpper1(ui8 c1)
{
    return (c1 >= 0x41) && (c1 <= 0x5a);
}
static void ToLower1(char *p1)
{
    ui8 c1 = *p1;
    if ((c1 >= 0x41) && (c1 <= 0x5a)) {
        *p1 = c1 + 0x20;
    }
}

static void ToUpper1(char *p1)
{
    ui8 c1 = *p1;
    if ((c1 >= 0x61) && (c1 <= 0x7a)) {
        *p1 = c1 - 0x20;
    }
}

// russian letters
//d090 ..d0af
//d081 
//d0b0 ..d0bf, d180 .. d18f
//d191
static bool IsUpper2(ui8 c1, ui8 c2)
{
    return (c1 == 0xd0) && ((c2 == 0x81) || ((c2 >= 0x90) && (c2 <= 0xaf)));
}
static void ToLower2(char *p1, char *p2)
{
    ui8 c1 = *p1;
    if (c1 != 0xd0) {
        return;
    }
    ui8 c2 = *p2;
    if (c2 == 0x81) {
        *p1 = (char)0xd1;
        *p2 = (char)0x91;
        return;
    }
    if (c2 >= 0x90 && c2 <= 0x9f) {
        *p2 = c2 + 0x20;
        return;
    }
    if (c2 >= 0xa0 && c2 <= 0xaf) {
        *p1 = (char)0xd1;
        *p2 = c2 - 0x20;
        return;
    }
}
static void ToUpper2(char *p1, char *p2)
{
    ui8 c1 = *p1;
    if (c1 != 0xd0) {
        return;
    }
    ui8 c2 = *p2;
    if (c1 == 0xd1 && c2 == 0x91) {
        *p1 = (char)0xd0;
        *p2 = (char)0x81;
        return;
    }
    if (c1 == 0xd0 && c2 >= 0xb0 && c2 <= 0xbf) {
        *p2 = c2 - 0x20;
        return;
    }
    if (c1 == 0xd1 && c2 >= 0x80 && c2 <= 0x8f) {
        *p1 = (char)0xd0;
        *p2 = c2 + 0x20;
        return;
    }
}


EWordCase GetWordCase(const TString &str)
{
    EWordCase res = WORD_LOWER_CASE;
    for (yint k = 0, sz = YSize(str); k < sz;) {
        ui8 c = str[k];
        yint len = Utf8CodeLength[c];
        if (k + len > sz) {
            // broken encoding
            res = WORD_MIXED_CASE;
            break;
        }
        if (len == 1 && IsUpper1(str[k])) {
            if (k == 0) {
                res = WORD_CAPITAL_START;
            } else {
                res = WORD_MIXED_CASE;
            }
        }
        if (len == 2 && IsUpper2(str[k], str[k + 1])) {
            if (k == 0) {
                res = WORD_CAPITAL_START;
            } else {
                res = WORD_MIXED_CASE;
            }
        }
        k += len;
    }
    return res;
}


TString ToLower(const TString &str)
{
    TString res = str;
    for (yint k = 0, sz = YSize(str); k < sz; ++k) {
        yint len = Utf8CodeLength[(ui8)res[k]];
        if (k + len > sz) {
            // broken encoding
            break;
        }
        if (len == 1) {
            ToLower1(&res[k]);
        }
        if (len == 2) {
            ToLower2(&res[k], &res[k + 1]);
        }
        k += len;
    }
    return res;
}


TString UpcaseFirstLetter(const TString &str)
{
    if (str.empty()) {
        return "";
    }
    TString res = str;
    yint sz = YSize(res);
    yint len = Utf8CodeLength[(ui8)res[0]];
    if (len > sz) {
        // broken encoding
        return res;
    }
    if (len == 1) {
        ToUpper1(&res[0]);
    }
    if (len == 2) {
        ToUpper2(&res[0], &res[1]);
    }
    return res;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
//
static ui16 CharTable1251[128] = {
    1026, 1027, 8218, 1107,  8222, 8230, 8224, 8225,
    8364, 8240, 1033, 8249,  1034, 1036, 1035, 1039,
    1106, 8216, 8217, 8220,  8221, 8226, 8211, 8212,
    152, 8482, 1113, 8250,   1114, 1116, 1115, 1119,

    160, 1038, 1118, 1032,   164, 1168, 166, 167,
    1025, 169, 1028, 171,    172, 173, 174, 1031,
    176, 177, 1030, 1110,    1169, 181, 182, 183,
    1105, 8470, 1108, 187,   1112, 1029, 1109, 1111,

    1040, 1041, 1042, 1043,  1044, 1045, 1046, 1047,
    1048, 1049, 1050, 1051,  1052, 1053, 1054, 1055,
    1056, 1057, 1058, 1059,  1060, 1061, 1062, 1063,
    1064, 1065, 1066, 1067,  1068, 1069, 1070, 1071,

    1072, 1073, 1074, 1075,  1076, 1077, 1078, 1079,
    1080, 1081, 1082, 1083,  1084, 1085, 1086, 1087,
    1088, 1089, 1090, 1091,  1092, 1093, 1094, 1095,
    1096, 1097, 1098, 1099,  1100, 1101, 1102, 1103
};


const yint MAX_CODE = 9000;
static TAtomic TableReady, TableLock;
static ui8 UnicodeTo1251[MAX_CODE];

static void MakeTables()
{
    if (TableReady) {
        return;
    }
    TGuard<TAtomic> lock(TableLock);
    for (yint k = 0; k < ARRAY_SIZE(UnicodeTo1251); ++k) {
        if (k < 128) {
            UnicodeTo1251[k] = k;
        } else {
            UnicodeTo1251[k] = '?';
        }
    }
    for (yint k = 0; k < ARRAY_SIZE(CharTable1251); ++k) {
        UnicodeTo1251[CharTable1251[k]] = 128 + k;
    }
    TableReady = 1;
}


TString Utf2Win(const TString &utf8)
{
    MakeTables();
    TString res;
    yint sz = YSize(utf8);
    for (yint i = 0; i < sz; ++i) {
        ui8 f = utf8[i];
        ui32 code = 0;

        if (f < 128) {
            code = f;

        } else if ((f & 0xe0) == 0xc0) {
            if (i + 1 >= sz) {
                break;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            code = (c1 & 0x3f) + ((c0 & 0x1f) << 6);

        } else if ((f & 0xf0) == 0xe0) {
            if (i + 2 >= sz) {
                break;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            ui32 c2 = utf8[i + 2];
            code = (c2 & 0x3f) + ((c1 & 0x3f) << 6) + ((c0 & 0xf) << 12);

        } else if ((f & 0xf8) == 0xf0) {
            if (i + 3 >= sz) {
                break;
            }
            ui32 c0 = utf8[i + 0];
            ui32 c1 = utf8[i + 1];
            ui32 c2 = utf8[i + 2];
            ui32 c3 = utf8[i + 3];
            code = (c3 & 0x3f) + ((c2 & 0x3f) << 6) + ((c1 & 0x3f) << 12) + ((c0 & 0x7) << 18);

        } else {
            continue; // not first utf encoding char somehow
        }

        if (code < MAX_CODE) {
            res.push_back(UnicodeTo1251[code]);
        } else {
            res.push_back('?');
        }
    }
    return res;
}


TString Win2Utf(const TString &cp1251)
{
    MakeTables();
    TString res;
    for (ui8 c : cp1251) {
        if (c < 128) {
            res.push_back(c);
        } else {
            ui32 code = CharTable1251[c - 128];
            if (code < 0x800) {
                res.push_back(0xc0 + (code >> 6));
                res.push_back(0x80 + (code & 0x3f));
            } else if (code < 0x10000) {
                res.push_back(0xe0 + (code >> 12));
                res.push_back(0x80 + ((code >> 6) & 0x3f));
                res.push_back(0x80 + (code & 0x3f));
            } else {
                res.push_back(0xf0 + (code >> 18));
                res.push_back(0x80 + ((code >> 12) & 0x3f));
                res.push_back(0x80 + ((code >> 6) & 0x3f));
                res.push_back(0x80 + (code & 0x3f));
            }
        }
    }
    return res;
}


char Unicode2Win(yint key)
{
    MakeTables();
    if (key >= MAX_CODE) {
        return '?';
    }
    if (key < 128) {
        return key;
    }
    return UnicodeTo1251[key];
}


//#include <codecvt>
//#include <locale>
//void BuildTable()
//{
//    TVector<char> cp1251;
//    for (yint k = 128; k < 256; ++k) {
//        cp1251.push_back(k);
//    }
//    cp1251.push_back(0);
//    TVector<wchar_t> wkeyArr;
//    wkeyArr.resize(1000);
//    yint wkeySize = MultiByteToWideChar(1251, 0, cp1251.data(), -1, wkeyArr.data(), YSize(wkeyArr));
//    wkeySize = wkeySize;
//}
