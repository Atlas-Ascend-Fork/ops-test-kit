/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * clear_l0 helper kernel for David-generation Ascend chips (dav-3510, ...).
 *
 * TIK cube intrinsics (mmad/load2dv2/fixpipe) are not supported on David-
 * generation chips, so L0A/L0B/L0C are cleared via tensor_api (the same
 * primitives Blaze uses):
 * GM -> L1 -> L0A/L0B -> mad -> L0C -> fixpipe -> GM.
 *
 * Tile dimensions M_DIM / K_DIM / N_DIM are passed at compile time via -D
 * macros (computed by the host from the target chip's L0A/L0B/L0C sizes).
 * One pass covers the full L0A and L0B, and as much of L0C as a single
 * mmad allows (M*K*2B = L0A, K*N*2B = L0B, M*N*4B <= L0C).
 *
 * Defaults below match Ascend950 (L0A/L0B = 64KB, L0C = 256KB):
 *   M=256, K=128, N=256
 *   L0A tile: 256 * 128 * 2B = 64KB  (full L0A)
 *   L0B tile: 128 * 256 * 2B = 64KB  (full L0B)
 *   L0C tile: 256 * 256 * 4B = 256KB (full L0C)
 *
 * Kernel args (both from TTK helper launch):
 *   input  : (M*K + K*N) * 2B fp16, every element = clean_val
 *            first M*K elements act as A (M x K ND), next K*N as B (K x N ND)
 *   output : M*N * 4B fp32 dummy sink for the fixpipe result (A x B)
 *            host reads it back after sync and verifies every element
 *            == K * clean_val^2, proving the full
 *            GM->L1->L0A/L0B->Mmad->L0C->fixpipe chain ran
 *
 * blockDim = AIC core num; every core runs the identical sequence so the
 * L0A/L0B/L0C of every core are overwritten with deterministic data.
 */

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace {
using namespace AscendC::Te;

#ifndef M_DIM
#define M_DIM 256
#endif
#ifndef K_DIM
#define K_DIM 128
#endif
#ifndef N_DIM
#define N_DIM 256
#endif

constexpr int32_t L1_EVENT_ID = 0;
constexpr int32_t L0_EVENT_ID = 1;
constexpr int32_t L0C_EVENT_ID = 2;
constexpr uint8_t FINAL_ACCUMULATION = 3;
}  // namespace

extern "C" __global__ __aicore__ __cube__ void clear_l0(GM_ADDR input, GM_ADDR output) {
    if ASCEND_IS_AIV {
        return;
    }
    using namespace AscendC::Te;

    auto gmAPtr = reinterpret_cast<__gm__ half *>(input);
    auto gmBPtr = reinterpret_cast<__gm__ half *>(input) + M_DIM * K_DIM;
    auto gmCPtr = reinterpret_cast<__gm__ float *>(output);

    auto gmATensor = MakeTensor(MakeMemPtr(gmAPtr), MakeFrameLayout<NDExtLayoutPtn>(M_DIM, K_DIM));
    auto gmBTensor = MakeTensor(MakeMemPtr(gmBPtr), MakeFrameLayout<NDExtLayoutPtn>(K_DIM, N_DIM));
    auto gmCTensor = MakeTensor(MakeMemPtr(gmCPtr), MakeFrameLayout<NDExtLayoutPtn>(M_DIM, N_DIM));

    __cbuf__ half l1ABuf[M_DIM * K_DIM];
    __cbuf__ half l1BBuf[K_DIM * N_DIM];
    __ca__ half l0ABuf[M_DIM * K_DIM];
    __cb__ half l0BBuf[K_DIM * N_DIM];
    __cc__ float l0CBuf[M_DIM * N_DIM];

    auto l1ATensor = MakeTensor(MakeMemPtr(l1ABuf), MakeFrameLayout<NZLayoutPtn, half>(M_DIM, K_DIM));
    auto l1BTensor = MakeTensor(MakeMemPtr(l1BBuf), MakeFrameLayout<NZLayoutPtn, half>(K_DIM, N_DIM));
    auto l0ATensor = MakeTensor(MakeMemPtr(l0ABuf), MakeFrameLayout<NZLayoutPtn, half>(M_DIM, K_DIM));
    auto l0BTensor = MakeTensor(MakeMemPtr(l0BBuf), MakeFrameLayout<ZNLayoutPtn, half>(K_DIM, N_DIM));
    auto l0CTensor = MakeTensor(MakeMemPtr(l0CBuf), MakeFrameLayout<NZLayoutPtn>(M_DIM, N_DIM));

    auto copyGM2L1Atom = MakeCopy(CopyGM2L1{}, CopyGM2L1TraitDefault{});
    auto copyL12L0AAtom = MakeCopy(CopyL12L0A{}, CopyL12L0ATraitDefault{});
    auto copyL12L0BAtom = MakeCopy(CopyL12L0B{}, CopyL12L0BTraitDefault{});
    auto copyL0C2GMAtom = MakeCopy(CopyL0C2GM{}, CopyL0C2GMTraitDefault{});
    auto mmadAtom = MakeMmad(MmadOperation{}, MmadTraitDefault{});

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);

    // GM -> L1
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
    Copy(copyGM2L1Atom, l1ATensor, gmATensor);
    Copy(copyGM2L1Atom, l1BTensor, gmBTensor);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(L1_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(L1_EVENT_ID);

    // L1 -> L0A/L0B
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);
    Copy(copyL12L0AAtom, l0ATensor, l1ATensor);
    Copy(copyL12L0BAtom, l0BTensor, l1BTensor);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0_EVENT_ID);

    // L0A x L0B -> L0C
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);
    MmadParams mmadParams;
    mmadParams.m = M_DIM;
    mmadParams.n = N_DIM;
    mmadParams.k = K_DIM;
    mmadParams.cmatrixInitVal = true;
    mmadParams.unitFlag = FINAL_ACCUMULATION;
    Mmad(mmadAtom.with(mmadParams), l0CTensor, l0ATensor, l0BTensor);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0C_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);

    // L0C -> GM
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0C_EVENT_ID);
    FixpipeParams fixpipeParams;
    fixpipeParams.unitFlag = FINAL_ACCUMULATION;
    Copy(copyL0C2GMAtom.with(fixpipeParams), gmCTensor, l0CTensor);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0C_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(L1_EVENT_ID);

    // Info-level log: only printed in CPU debug mode; compiled out (empty macro)
    // on real-device builds, so it never floods stdout on board.
    KERNEL_LOG(KERNEL_INFO, "[clear_l0] core=%u l0 injected, cycle=%lu",
               static_cast<uint32_t>(AscendC::GetBlockIdx()), AscendC::GetSystemCycle());
}
