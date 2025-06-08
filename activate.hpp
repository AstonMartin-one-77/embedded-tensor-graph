
#ifndef TG_ACTIVATE
#define TG_ACTIVATE

#include "tensor.hpp"
#include <cmath>

namespace tg
{

template <typename T>
Tensor<T> sigmoid(Tensor<T>& input) {
    auto res = input.copy();
    auto shape = res.shape();
    size_t rowElems = *shape.rbegin();
    size_t nRows = res.getRowsNumber();
    for (size_t r = 0; r < nRows; ++r) {
        auto* rowPtr = res.getRow(r);
        for (size_t i = 0; i < rowElems; ++i) {
            rowPtr[i] = (T)1 / ((T)1+exp(-rowPtr[i]));
        }
    }
    return res;
}

template <typename T>
Tensor<T> dSigmoid(Tensor<T>& dA, Tensor<T>& aSaved) {
    auto res = dA.copy();
    auto shape = res.shape();
    size_t rowElems = *shape.rbegin();
    size_t nRows = res.getRowsNumber();
    for (size_t r = 0; r < nRows; ++r) {
        auto* rowPtr = res.getRow(r);
        auto* aSavedRowPtr = aSaved.getRow(r);
        for (size_t i = 0; i < rowElems; ++i) {
            rowPtr[i] *= ((T)1-aSavedRowPtr[i]) * aSavedRowPtr[i];
        }
    }
    return res;
}

template <typename T>
Tensor<T> relu(Tensor<T>& input) {
    auto res = input.copy();
    auto shape = res.shape();
    size_t rowElems = *shape.rbegin();
    size_t nRows = res.getRowsNumber();
    for (size_t r = 0; r < nRows; ++r) {
        auto* rowPtr = res.getRow(r);
        for (size_t i = 0; i < rowElems; ++i) {
            rowPtr[i] = std::max((T)0, rowPtr[i]);
        }
    }
    return res;
}

template <typename T>
Tensor<T> dRelu(Tensor<T>& dA, Tensor<T>& aSaved) {
    auto res = dA.copy();
    auto shape = res.shape();
    size_t rowElems = *shape.rbegin();
    size_t nRows = res.getRowsNumber();
    for (size_t r = 0; r < nRows; ++r) {
        auto* rowPtr = res.getRow(r);
        auto* aRowPtr = aSaved.getRow(r);
        for (size_t i = 0; i < rowElems; ++i) {
            rowPtr[i] = (aRowPtr[i] <= 0) ? 0 : rowPtr[i];
        }
    }
    return res;
}

} // namespace tg


#endif // TG_ACTIVATE
