#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/renorm_native.h>
#include <ATen/ops/embedding_renorm_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/TensorArg.h>
#endif

namespace at::native {
namespace {

using namespace mps;

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/RenormKernel_metallib.h>
#endif

void renorm_out_mps(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  auto self_sizes = self.sizes();
  dim = c10::maybe_wrap_dim(dim, self_sizes.size());

  DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  Tensor norm = at::linalg_vector_norm(self, p.toDouble(), reduce_dims, /*keepdim=*/true);
  auto factor = at::empty(norm.sizes(), self.options());

  id<MTLDevice> device = MPSDevice::getInstance()->device();
  id<MTLBuffer> normBuffer = getMTLBufferStorage(norm);
  id<MTLBuffer> factorBuffer = getMTLBufferStorage(factor);

  std::string key = "renorm_" + scalarToMetalTypeString(self);
  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
  id<MTLComputePipelineState> renormPSO = lib.getPipelineStateForFunc(key);

  dispatch_sync(mpsStream->queue(), ^() {
    @autoreleasepool {
      // this function call is a no-op if MPSProfiler is not enabled
      getMPSProfiler().beginProfileKernel(renormPSO, key, {norm});

      [computeEncoder setComputePipelineState:renormPSO];
      mtl_setArgs(computeEncoder, norm, factor, maxnorm.to<float>());
      mtl_dispatch1DJob(computeEncoder, renormPSO, norm.numel());

      getMPSProfiler().endProfileKernel(renormPSO);
    }
  });
  at::mul_outf(self, factor, const_cast<Tensor&>(out));
}

} // namespace

TORCH_IMPL_FUNC(renorm_out_mps)
(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm, const Tensor& out) {
  renorm_out_mps(self, p, dim, maxnorm, out);
}

// embedding_renorm_ MPS implementation
// Renormalizes embedding rows specified by indices to have at most max_norm
Tensor& embedding_renorm_mps_(
    Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 2);
  checkDim("embedding_renorm_", self_arg, 2);
  checkScalarTypes("embedding_renorm_", indices_arg, {kLong, kInt});

  auto indices_contig = indices.contiguous();
  auto num_indices = indices.numel();

  if (num_indices == 0) {
    return self;
  }

  // Get unique indices
  auto unique_indices = std::get<0>(at::_unique(indices_contig));

  // For each unique index, compute norm and scale if needed
  for (int64_t i = 0; i < unique_indices.numel(); i++) {
    int64_t idx = unique_indices[i].item<int64_t>();
    auto row = self[idx];
    auto norm = row.norm(norm_type);
    auto norm_val = norm.item<double>();
    if (norm_val > max_norm) {
      auto scale = max_norm / (norm_val + 1e-7);
      row.mul_(scale);
    }
  }

  return self;
}

} // namespace at::native
