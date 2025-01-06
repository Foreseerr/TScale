#pragma once

enum {
    MATMUL_FP16,
    MATMUL_INT8,
    MATMUL_FP8,
};

enum {
    ATT_FP16,
    ATT_FP8,
};


///////////////////////////////////////////////////////////////////////////////////////////////////
// select matmul

// int8 matmul (requires i8 model matrices)
constexpr int FWD_MATMUL_TYPE = MATMUL_INT8;
constexpr int BWD_MATMUL_TYPE = MATMUL_INT8;
//constexpr int BWD_MATMUL_TYPE = MATMUL_FP16;
typedef i8 TNormStateFloat;
typedef i8 TFastModelFloat;

//// fp8 matmul (requires e4m3 model matrices)
//constexpr int FWD_MATMUL_TYPE = MATMUL_FP8;
//constexpr int BWD_MATMUL_TYPE = MATMUL_FP8;
////constexpr int BWD_MATMUL_TYPE = MATMUL_FP16;
//typedef e4m3 TNormStateFloat;
//typedef e4m3 TFastModelFloat;

//// fp16 matmul
//constexpr int FWD_MATMUL_TYPE = MATMUL_FP16;
//constexpr int BWD_MATMUL_TYPE = MATMUL_FP16;
////typedef i8 TNormStateFloat;
////typedef e4m3 TNormStateFloat;
//typedef half TNormStateFloat;
////typedef i8 TFastModelFloat;
////typedef e4m3 TFastModelFloat;
//typedef half TFastModelFloat;


///////////////////////////////////////////////////////////////////////////////////////////////////
// select attention

// fp8 attention
constexpr int ATT_TYPE = ATT_FP8;
constexpr int ATT_GROUP = 64;
constexpr int ATT_ALIGN = 128;
typedef e4m3 TAttVecFloat;

//// fp16 attention
//constexpr int ATT_TYPE = ATT_FP16;
//constexpr int ATT_GROUP = 16;
//constexpr int ATT_ALIGN = 16;
////typedef i8 TAttVecFloat;
////typedef e4m3 TAttVecFloat;
//typedef half TAttVecFloat;


///////////////////////////////////////////////////////////////////////////////////////////////////
typedef half TEmbedFloat;
