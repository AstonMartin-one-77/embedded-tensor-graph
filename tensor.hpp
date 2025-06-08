
#ifndef TG_TENSOR
#define TG_TENSOR

#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <type_traits>
#include <utility>
#include <typeinfo>
#include <numeric>
#include <functional>
#include <cmath>
#include <random>

namespace tg
{

#ifndef TENSOR_ALIGNMENT
/* default memory alignment of Tensors in bytes */
#define TENSOR_ALIGNMENT (32u)
#endif

template <typename T>
class Tensor
{
    T* m_memory;
    void* m_rawMemory;
    size_t m_memorySize;
    std::vector<size_t> m_stride;
    std::vector<size_t> m_shape;
    std::vector<size_t> m_paddedShape;
    std::vector<size_t> m_lpadding;
    std::vector<size_t> m_rpadding;
    size_t m_alignment;

public:
    Tensor() noexcept : m_memory{nullptr}, m_rawMemory{nullptr}, m_memorySize{0}, m_alignment{0} {}
    ~Tensor() {
        if (m_rawMemory != nullptr) std::free(m_rawMemory);
        m_rawMemory = nullptr;
        m_memory = nullptr;
        m_memorySize = 0;
    }

    template<size_t R, size_t C>
    Tensor(const T (&array)[R][C]):
        Tensor(std::vector<size_t>{R,C}) {
        for (size_t i = 0; i < R; ++i) {
            memcpy(getRow(i), array[i], sizeof(T)*C);
        }
    }

    Tensor(const std::vector<size_t> &shape):
        Tensor(shape, std::vector<size_t>(shape.size(), 0)) {}

    Tensor(const std::vector<size_t> &shape, const std::vector<size_t> &padding):
        Tensor(shape, padding, padding, TENSOR_ALIGNMENT) {}

    Tensor(const std::vector<size_t> &shape, const std::vector<size_t> &lpadding, const std::vector<size_t> &rpadding, const size_t alignment):
        m_stride{shape},
        m_shape{shape},
        m_paddedShape{shape},
        m_lpadding{lpadding},
        m_rpadding{rpadding},
        m_alignment{alignment} {
        static_assert(std::is_floating_point<T>::value, "T must be floating point");
        if (0 != (alignment) % sizeof(T)) throw std::invalid_argument("Alignment must be multiple of sizeof(T)");
        if (0 == shape.size()) throw std::invalid_argument("Shape must have at least one dim");
        if (shape.size() != lpadding.size()) throw std::invalid_argument("Shape and padding must have the same length");
        if (shape.size() != rpadding.size()) throw std::invalid_argument("Shape and padding must have the same length");

        for (uint32_t i = 0; i < m_paddedShape.size(); ++i) {
            if (0 == m_shape[i]) throw std::invalid_argument("Shape must have non-zero elements");
            m_paddedShape[i] += m_lpadding[i] + m_rpadding[i];
        }

        /* Calculate aligned (last) dim of shape for stride (other stride are based on it) */
        size_t alignedElems = getAlignedSize((*m_paddedShape.rbegin()) * sizeof(T), m_alignment) / sizeof(T);
        m_memorySize = 1u;
        /* Calculate total number of elements in shape */
        for (auto itr = m_paddedShape.begin(); itr != m_paddedShape.end(); ++itr) {
            m_memorySize *= *itr;
        }
        m_memorySize /= *m_paddedShape.rbegin(); // Exclude last dim of shape
        m_memorySize *= alignedElems; // replace with aligned elements
        m_memorySize *= sizeof(T);
         
        /* Allocate memory for Tensor */
        m_rawMemory = malloc(m_memorySize+m_alignment);
        if (m_rawMemory == nullptr) throw std::bad_alloc();
        memset(m_rawMemory, 0, m_memorySize+m_alignment);
        /* Align memory */
        m_memory = reinterpret_cast<T*>(((uintptr_t)m_alignment)*(((uintptr_t)m_rawMemory+((uintptr_t)m_alignment)-1u)/((uintptr_t)m_alignment)));
        
        size_t totalElements = m_memorySize / sizeof(T);
        /* Calculate stride */
        for (auto itr = m_stride.begin(); itr != m_stride.end(); ++itr) {
            totalElements /= *itr; // exclude dims of shape step by step (stride initialised by shape)
            *itr = totalElements;
        }
    }

    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;

    Tensor(Tensor&& other) noexcept {
        m_rawMemory = std::exchange(other.m_rawMemory, nullptr);
        m_memory = std::exchange(other.m_memory, nullptr);
        m_memorySize = std::exchange(other.m_memorySize, 0);
        m_shape = std::move(other.m_shape);
        m_paddedShape = std::move(other.m_paddedShape);
        m_lpadding = std::move(other.m_lpadding);
        m_rpadding = std::move(other.m_rpadding);
        m_stride = std::move(other.m_stride);
        m_alignment = std::exchange(other.m_alignment, 0);
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        if (m_rawMemory != nullptr) std::free(m_rawMemory);
        m_rawMemory = std::exchange(other.m_rawMemory, nullptr);
        m_memory = std::exchange(other.m_memory, nullptr);
        m_memorySize = std::exchange(other.m_memorySize, 0);
        m_shape = std::move(other.m_shape);
        m_paddedShape = std::move(other.m_paddedShape);
        m_lpadding = std::move(other.m_lpadding);
        m_rpadding = std::move(other.m_rpadding);
        m_stride = std::move(other.m_stride);
        m_alignment = std::exchange(other.m_alignment, 0);
        return *this;
    }

    Tensor& operator+=(const Tensor& other) {
        _applyOperation(other, [](T& dest, const T& src) { dest += src; });
        return *this;
    }

    Tensor& operator*=(const Tensor& other) {
        _applyOperation(other, [](T& dest, const T& src) { dest *= src; });
        return *this;
    }

    Tensor& operator/=(const Tensor& other) {
        _applyOperation(other, [](T& dest, const T& src) { dest /= src; });
        return *this;
    }

    Tensor& operator*=(T scalar) {
        _applyScalarOperation(scalar, [](T& dest, const T& scalar) { dest *= scalar; });
        return *this;
    }

    Tensor& operator+=(T scalar) {
        _applyScalarOperation(scalar, [](T& dest, const T& scalar) { dest += scalar; });
        return *this;
    }

    Tensor operator+(const Tensor& other) const {
        auto result = this->copy();
        result += other;
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        auto result = this->copy();
        result *= other;
        return result;
    }

    Tensor operator/(const Tensor& other) const {
        auto result = this->copy();
        result /= other;
        return result;
    }

    Tensor operator*(T scalar) {
        auto result = this->copy();
        result *= scalar;
        return result;
    }

    Tensor operator+(T scalar) {
        auto result = this->copy();
        result += scalar;
        return result;
    }

    Tensor transpose() const {
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        std::vector<size_t> out_shape{m_shape.rbegin(), m_shape.rend()};
        std::vector<size_t> out_lpad{m_lpadding.rbegin(), m_lpadding.rend()};
        std::vector<size_t> out_rpad{m_rpadding.rbegin(), m_rpadding.rend()};
        Tensor output{out_shape, out_lpad, out_rpad, m_alignment};

        _transposeRecursion(output, *this, 0,0,out_shape.size());
        return output;
    }

    Tensor sum() const { return sum(0); }
    Tensor sum(size_t axis) const {
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        if (axis >= m_shape.size()) throw std::invalid_argument("Wrong axis index: out of bounds");
        std::vector<size_t> out_shape;
        std::vector<size_t> out_lpad;
        std::vector<size_t> out_rpad;
        size_t nOuterCycles = std::accumulate(m_shape.begin(), m_shape.begin()+axis, 1, std::multiplies<>());

        for (size_t i = 0; i < m_shape.size(); ++i) {
            if (i != axis) {
                out_shape.push_back(m_shape[i]);
                out_lpad.push_back(m_lpadding[i]);
                out_rpad.push_back(m_rpadding[i]);
            }
        }
        Tensor output{out_shape, out_lpad, out_rpad, m_alignment};
        size_t out_stride = (axis > 0) ? output.m_stride[axis-1u] : 0;
        if ((axis+1u) == m_shape.size()) {
            for (size_t oc = 0; oc < nOuterCycles; ++oc) {
                T* outPtr = &output.m_memory[out_stride*oc];
                const T* inPtr = getRow(oc);
                for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                    *outPtr += inPtr[s];
                }
            }
        } else {
            size_t nInnerCycles = std::accumulate(m_shape.begin()+axis+1u, m_shape.end()-1u, 1, std::multiplies<>());
            size_t in_axisStride = m_stride[axis];
            size_t in_outStride = (axis > 0) ? m_stride[axis-1u] : 0;
            size_t in_lastStride = (m_stride.size()>1u) ? *(m_stride.rbegin()+1u) : 0;
            size_t out_lastStride = (output.m_stride.size()>1u) ? *(output.m_stride.rbegin()+1u) : 0;
            // {dim0, dim1, dim2, dim3} -> {dim0, dim2, dim3}
            // dim0 -> number of outer cycles
            for (size_t oc = 0; oc < nOuterCycles; ++oc) {
                // dim2 -> number of inner cycles
                for (size_t ic = 0; ic < nInnerCycles; ++ic) {
                    size_t in_stride = in_outStride*oc+in_lastStride*ic;
                    T* outPtr = &output.m_memory[out_stride*oc+out_lastStride*ic];
                    // stride of dim1 -> summation along it
                    for (size_t i = 0; i < m_shape[axis]; ++i) {
                        T* inPtr = &m_memory[in_stride+in_axisStride*i];
                        for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                            outPtr[s] += inPtr[s];
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor matmul(const Tensor& other) const {
        if ((m_shape.size() < 2u) || (other.m_shape.size() < 2u)) throw std::invalid_argument("A and/or B is not a Matrix");
        if (m_shape.size() < other.m_shape.size()) throw std::invalid_argument("A matrix cannot be smaller then B");
        if (*m_paddedShape.rbegin() != other.m_paddedShape[0]) throw std::invalid_argument("Wrong last matrix dims");

        std::vector<size_t> out_shape{m_shape};
        std::vector<size_t> out_lpad{m_lpadding};
        std::vector<size_t> out_rpad{m_rpadding};
        *out_shape.rbegin() = *other.m_shape.rbegin();
        *out_lpad.rbegin() = *other.m_lpadding.rbegin();
        *out_rpad.rbegin() = *other.m_rpadding.rbegin();
        Tensor output{out_shape, out_lpad, out_rpad, m_alignment};

        size_t aRow_stride = *(m_stride.rbegin()+1u);
        size_t bRow_stride = *(other.m_stride.rbegin()+1u);
        size_t outRow_stride = *(output.m_stride.rbegin()+1u);
        auto get_matStride = [] (auto strides, auto shape) { return (strides.size() > 2u) ? *(strides.rbegin()+2u) : strides[0]*shape[0]; };
        size_t aMtx_stride = get_matStride(m_stride, m_shape);
        size_t bMtx_stride = get_matStride(other.m_stride, other.m_shape);
        size_t outMtx_stride = get_matStride(output.m_stride, output.m_shape);
        size_t numMatmuls = m_memorySize / (aMtx_stride * sizeof(T));

        size_t m_dim = *(output.m_paddedShape.rbegin()+1u);
        size_t n_dim = *output.m_paddedShape.rbegin();
        size_t k_dim = *m_paddedShape.rbegin();

        if (2u == other.m_shape.size()) {
            
            for (size_t l = 0; l < numMatmuls; ++l) {
                const T* ptr = &m_memory[l*aMtx_stride];
                T* out = &output.m_memory[l*outMtx_stride];
                for (size_t i = 0; i < m_dim; ++i) {
                    for (size_t j = 0; j < n_dim; ++j) {
                        T sum{0};
                        for (size_t k = 0; k < k_dim; ++k) {
                            sum += ptr[i*aRow_stride+k] * other.m_memory[k*bRow_stride+j];
                        }
                        out[i*outRow_stride+j] = sum;
                    }
                }
            }

        } else {
            if (m_shape.size() != other.m_shape.size()) throw std::invalid_argument("For multi-dim cases: number of A and B dims must be equal");
            
            for (size_t l = 0; l < numMatmuls; ++l) {
                const T* a_ptr = &m_memory[l*aMtx_stride];
                const T* b_ptr = &other.m_memory[l*bMtx_stride];
                T* out = &output.m_memory[l*outMtx_stride];
                for (size_t i = 0; i < m_dim; ++i) {
                    for (size_t j = 0; j < n_dim; ++j) {
                        T sum{0};
                        for (size_t k = 0; k < k_dim; ++k) {
                            sum += a_ptr[i*aRow_stride+k] * b_ptr[k*bRow_stride+j];
                        }
                        out[i*outRow_stride+j] = sum;
                    }
                }
            }
        }
        return output;
    }

    Tensor& log() {
        static_assert(std::is_floating_point<T>::value, "log(): only floating point supported");
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        auto rows = this->getRowsNumber();
        for (size_t r = 0; r < rows; ++r) {
            T* inPtr = getRow(r);
            for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                inPtr[s] = std::log(inPtr[s]);
            }
        }
        return *this;
    }

    Tensor& sqrt() {
        static_assert(std::is_floating_point<T>::value, "sqrt(): only floating point supported");
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        auto rows = this->getRowsNumber();
        for (size_t r = 0; r < rows; ++r) {
            T* inPtr = getRow(r);
            for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                inPtr[s] = std::sqrt(inPtr[s]);
            }
        }
        return *this;
    }

    Tensor& square() {
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        auto rows = this->getRowsNumber();
        for (size_t r = 0; r < rows; ++r) {
            T* inPtr = getRow(r);
            for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                inPtr[s] = inPtr[s] * inPtr[s];
            }
        }
        return *this;
    }

    Tensor& setZeros() {
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        auto rows = this->getRowsNumber();
        for (size_t r = 0; r < rows; ++r) {
            memset(getRow(r), 0, sizeof(T) * *(m_paddedShape.rbegin()));
        }
        return *this;
    }

    Tensor& randn(float scale) {
        static std::mt19937 generator{std::random_device{}()};
        static std::normal_distribution<T> distribution((T)0.0, (T)1.0);
        static_assert(std::is_floating_point<T>::value, "sqrt(): only floating point supported");
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        auto rows = this->getRowsNumber();
        for (size_t r = 0; r < rows; ++r) {
            T* inPtr = getRow(r);
            for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                inPtr[s] = distribution(generator) * scale;
            }
        }
        return *this;
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        static_assert((std::is_convertible<Indices, size_t>::value && ...), "All indices must be convertible to size_t");
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        std::array<size_t, sizeof...(Indices)> castedIndices{static_cast<size_t>(indices)...};
        if (castedIndices.size() > m_stride.size()) throw std::invalid_argument("Number of indices must match tensor rank or be less");

        size_t offset = 0;
        for (size_t i = 0; i < castedIndices.size(); ++i) {
            if (castedIndices[i] >= m_paddedShape[i]) throw std::invalid_argument("Index is out of bounds");
            offset += castedIndices[i] * m_stride[i];
        }
        return m_memory[offset];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        return const_cast<Tensor*>(this)->operator()(indices...);
    }

    T* getRow(size_t idx) {
        return _getRow(idx);
    }

    size_t getRowsNumber() const {
        if (!m_shape.size()) throw std::runtime_error("Shape of Tensor is not constructed");
        return std::accumulate(m_shape.begin(), m_shape.end()-1u, 1, std::multiplies<>());
    }

    const T* getRow(size_t idx) const {
        return const_cast<Tensor*>(this)->_getRow(idx);
    }

    std::vector<size_t>& shape() {
        return m_paddedShape;
    }

    const std::vector<size_t>& shape() const {
        return m_paddedShape;
    }

    void info(void) const {
        std::cout << "Tensor<" << typeid(m_memory[0]).name() << ">: shape{";
        for (auto i : m_shape) std::cout << i << ",";
        std::cout << "} lpad{";
        for (auto i : m_lpadding) std::cout << i << ",";
        std::cout << "} rpad{";
        for (auto i : m_rpadding) std::cout << i << ",";
        std::cout << "} strides{";
        for (auto i : m_stride) std::cout << i << ",";
        std::cout << "} alignment{" << m_alignment << 
                     "} memory_size{" << m_memorySize << "}" << std::endl;
    }

    bool equalRank(const Tensor& other) const {
        bool res = (m_shape != other.m_shape);
        res |= (m_lpadding != other.m_lpadding);
        res |= (m_rpadding != other.m_rpadding);
        res |= (m_stride != other.m_stride);
        res |= (m_memorySize != other.m_memorySize);
        return !res;
    }

    void copy(const Tensor& source) {
        if (getRowsNumber() != source.getRowsNumber()) throw std::invalid_argument("copy(): number of rows is different");
        if (*(shape().rbegin()) != *(source.shape().rbegin())) throw std::invalid_argument("copy(): number of rows is different");
        for (size_t i = 0; i < getRowsNumber(); ++i) {
            memcpy(getRow(i), source.getRow(i), *(shape().rbegin())*sizeof(T));
        }
    }

    Tensor copy() const {
        Tensor res{this->m_shape, this->m_lpadding, this->m_rpadding, this->m_alignment};
        if (res.m_memorySize != this->m_memorySize) throw std::runtime_error("Wrong copy of Tensor");
        memcpy(res.m_memory, this->m_memory, res.m_memorySize);
        return res;
    }

    Tensor sigmoid(Tensor& input) {
        Tensor res{this->m_shape, this->m_lpadding, this->m_rpadding, this->m_alignment};
        if (res.m_memorySize != this->m_memorySize) throw std::runtime_error("Wrong size of Tensor");
        memcpy(res.m_memory, this->m_memory, res.m_memorySize);
        return res;
    }

    static size_t getAlignedSize(size_t size, size_t alignment) {
        return (alignment) * ((size + (alignment) - 1u) / (alignment));
    }

private:
    T* _getRow(size_t idx) {
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        size_t stride = (m_stride.size() >= 2u) ? *(m_stride.rbegin()+1u) : 0;
        size_t maxRows = (stride > 0) ? m_memorySize / stride : 1u;

        if (idx >= maxRows) throw std::invalid_argument("Index of Row out of bounds");
        return &m_memory[idx * stride];
    }

    static void _transposeRecursion(Tensor& out, const Tensor& src, size_t out_offset, size_t src_offset, size_t depth) {
        if (1u == depth){
            size_t out_stride = out.m_stride[0];
            for (size_t i = 0; i < out.m_shape[0]; ++i) {
                out.m_memory[out_stride*i+out_offset] = src.m_memory[src_offset+i];
            }
        } else { // recursion
            size_t out_stride = out.m_stride[depth-1u];
            size_t src_stride = src.m_stride[src.m_stride.size()-depth];
            for (size_t i = 0; i < src.m_shape[src.m_stride.size()-depth]; ++i) {
                _transposeRecursion(out, src, out_offset+i*out_stride, src_offset+i*src_stride, depth-1u);
            }
        }
    }

    Tensor& _applyOperation(const Tensor& other, std::function<void(T&, const T&)> operation) {
        if ((nullptr == m_memory) || (nullptr == other.m_memory)) throw std::runtime_error("Memory is not allocated");
        if (m_alignment != other.m_alignment) throw std::invalid_argument("Alignments are mixed");

        if (m_shape.size() > other.m_shape.size()) { // broadcasting: N-dim Tensor + (N-diff)-dim Tensor
            size_t diff = m_shape.size() - other.m_shape.size();
            for (size_t i = 0; i < other.m_shape.size(); ++i) {
                if (m_shape[diff+i] != other.m_shape[i]) throw std::invalid_argument("Last dims of N-dim Tensor and (N-diff)-dims of Tensor must be the same for *= operation");
                if (m_lpadding[diff+i] != other.m_lpadding[i]) throw std::invalid_argument("Last dims of N-dim Tensor and (N-diff)-dims of Tensor must be the same for *= operation");
                if (m_rpadding[diff+i] != other.m_rpadding[i]) throw std::invalid_argument("Last dims of N-dim Tensor and (N-diff)-dims of Tensor must be the same for *= operation");
                if (m_stride[diff+i] != other.m_stride[i]) throw std::invalid_argument("Last dims of N-dim Tensor and (N-diff)-dims of Tensor must be the same for *= operation");
            }
            size_t numLines = m_memorySize / (m_stride[diff] * sizeof(T));
            for (size_t l = 0; l < numLines; ++l) {
                for (size_t i = 0; i < m_stride[diff]; ++i) {
                    operation(m_memory[l*m_stride[diff]+i], other.m_memory[i]);
                }
            }
        } else {
            if (true != equalRank(other)) throw std::invalid_argument("Ranks of Tensors must be the same for *= operation");
            auto rows = this->getRowsNumber();
            for (size_t r = 0; r < rows; ++r) {
                T* dest = this->getRow(r);
                const T* src = other.getRow(r);
                for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                    operation(dest[s], src[s]);
                }
            }
        }
        return *this;
    }

    Tensor& _applyScalarOperation(const T scalar, std::function<void(T&, const T&)> operation) {
        if (nullptr == m_memory) throw std::runtime_error("Memory is not allocated");
        auto rows = this->getRowsNumber();
        for (size_t r = 0; r < rows; ++r) {
            T* dest = this->getRow(r);
            for (size_t s = 0; s < *(m_paddedShape.rbegin()); ++s) {
                operation(dest[s], scalar);
            }
        }
        return *this;
    }
};

} // namespace tg
#endif // TG_TENSOR