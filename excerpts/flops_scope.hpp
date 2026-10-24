// GEMM coefficients include multiply-add (2 FLOPs) and the FP8-equivalent weight.
namespace flops_coef {
constexpr int64_t FP4 = 1;
constexpr int64_t FP8 = 2;
constexpr int64_t BF16 = 4;
constexpr int64_t TF32 = 8;
constexpr int64_t FP32 = TF32; // Accounting reference, not actual kernel precision.
} // namespace flops_coef

// Estimate main-model work in FP8-equivalent FLOPs; FP32/TF32 use weight 4.
// Count major contractions regardless of bottleneck; exclude draft/padding/replicas.
// MFU measures target-model FLOPs throughput; DFlash is extra compute used
// to accelerate that model, so its FLOPs are excluded.
// Omit small arithmetic (norms/activations). TODO: vision.
// Scheduled q/k ignore acceptance/ZOS corrections.
