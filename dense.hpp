
#ifndef TG_DENSE
#define TG_DENSE

#include "layer.hpp"
#include <functional>
#include <algorithm>
#include "activate.hpp"
#include "optimiser.hpp"

namespace tg
{
template <typename T>
class Dense : public Layer<T> {
private:
    std::vector<size_t> m_shape; // Output shape
    std::string m_activation;
    std::string m_optimiser;
    float m_learningRate;
    Tensor<T> m_kernel;
    Tensor<T> m_bias;
    std::function<Tensor<T>(Tensor<T>&)> activate;
    std::function<Tensor<T>(Tensor<T>&, Tensor<T>&)> dActivate;
    std::function<Tensor<T>(Tensor<T>&, T)> applyKernelGrad;
    std::function<Tensor<T>(Tensor<T>&, T)> applyBiasGrad;

    std::shared_ptr<Tensor<T>> m_savedInput;
    std::shared_ptr<Tensor<T>> m_savedZ;
    std::shared_ptr<Tensor<T>> m_savedA;

    // Adam optimiser estimators
    Adam<T> m_kernelEst;
    Adam<T> m_biasEst;

public:
    Dense(): m_shape(1, 0) {}
    ~Dense() = default;

    Dense(const Dense& other) = delete;
    Dense& operator=(const Dense& other) = delete;

    Dense(Dense&& other) noexcept {
        m_shape = std::move(other.m_shape);
        m_activation = std::move(other.m_activation);
        m_learningRate = other.m_learningRate;
        m_kernel = std::move(other.m_kernel);
        m_bias = std::move(other.m_bias);
        activate = std::exchange(other.activate, nullptr);
        dActivate = std::exchange(other.dActivate, nullptr);
        m_savedInput = std::move(other.m_savedInput);
        m_savedZ = std::move(other.m_savedZ);
        m_savedA = std::move(other.m_savedA);
    }

    Dense& operator=(Dense&& other) noexcept {
        if (this == &other) return *this;
        m_shape = std::move(other.m_shape);
        m_activation = std::move(other.m_activation);
        m_learningRate = other.m_learningRate;
        m_kernel = std::move(other.m_kernel);
        m_bias = std::move(other.m_bias);
        activate = std::exchange(other.activate, nullptr);
        dActivate = std::exchange(other.dActivate, nullptr);
        m_savedInput = std::move(other.m_savedInput);
        m_savedZ = std::move(other.m_savedZ);
        m_savedA = std::move(other.m_savedA);
        return *this;
    }

    Dense(size_t units, const std::shared_ptr<Layer<T>>& input_layer):
        Dense(units, "None", 0.001f, input_layer) {}
    
    Dense(size_t units, const std::string activation, 
        const std::shared_ptr<Layer<T>>& input_layer):
        Dense(units, activation, 0.001f, input_layer) {}
    
    Dense(size_t units, const std::string activation, float learningRate,
        const std::shared_ptr<Layer<T>>& input_layer) : 
        Dense(units, activation, "sgd", learningRate, input_layer) {}

    Dense(size_t units, const std::string activation, 
        const std::string optimiser, float learningRate,
        const std::shared_ptr<Layer<T>>& input_layer):
        m_shape{input_layer.get()->shape()},
        m_activation{activation},
        m_optimiser{optimiser},
        m_learningRate{learningRate},
        m_kernel{std::vector<size_t>{*(input_layer.get()->shape().rbegin()), units}}, 
        m_bias{std::vector<size_t>{units}} {
        *m_shape.rbegin() = units; // Replace last dim with units (output shape)
        std::transform(m_activation.begin(), m_activation.end(), m_activation.begin(), [](int c){ return std::tolower(c); });
        if ("none" == m_activation) {
            activate = [](Tensor<T>& input) { return std::move(input); };
            dActivate = [](Tensor<T>& dA, Tensor<T>& aSaved) { return std::move(dA); };
        } else if ("sigmoid" == m_activation) {
            activate = [](Tensor<T>& input) { return sigmoid(input); };
            dActivate = [](Tensor<T>& dA, Tensor<T>& aSaved) { return dSigmoid(dA, aSaved); };
        } else if ("relu" == m_activation) {
            activate = [](Tensor<T>& input) { return relu(input); };
            dActivate = [](Tensor<T>& dA, Tensor<T>& aSaved) { return dRelu(dA, aSaved); };
        } else {
            throw std::invalid_argument("Unsupported activation function");
        }

        std::transform(m_optimiser.begin(), m_optimiser.end(), m_optimiser.begin(), [](int c){ return std::tolower(c); });
        if ("adam" == m_optimiser) {
            m_kernelEst = std::move(Adam<T>{m_kernel});
            m_biasEst = std::move(Adam<T>{m_bias});
            applyKernelGrad = [this](Tensor<T>& grad, T learningRate) {
                auto estRes = m_kernelEst.estimate(grad);
                estRes *= learningRate;
                return estRes;
            };
            applyBiasGrad = [this](Tensor<T>& grad, T learningRate) { 
                auto estRes = m_biasEst.estimate(grad);
                estRes *= learningRate;
                return estRes;
            };
        } else if ("sgd" == m_optimiser) {
            applyKernelGrad = [](Tensor<T>& grad, T learningRate) { 
                grad *= learningRate;
                return std::move(grad);
            };
            applyBiasGrad = applyKernelGrad;
        } else {
            throw std::runtime_error("Unsupported optimiser");
        }
        auto prevLayerUnits = *(input_layer.get()->shape().rbegin());
        m_kernel.randn(std::sqrt(2.f/(float)prevLayerUnits));
    }

    const std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        std::vector<std::shared_ptr<Tensor<T>>> result;
        m_savedInput = inputs[0];
        auto z = inputs[0]->matmul(m_kernel);
        z += m_bias;
        m_savedA = std::make_shared<Tensor<T>>(std::move(activate(z)));
        m_savedZ = std::make_shared<Tensor<T>>(std::move(z));
        result.push_back(m_savedA);
        return result;
    }
    
    const std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        static_assert(std::is_floating_point<T>::value, "Only floating points are supported (learning rate specified in Floats)");
        std::vector<std::shared_ptr<Tensor<T>>> result;
        size_t nExamples = *(inputs[0]->shape().begin());
        float avgCoeff = 1.f / (float)nExamples;
        auto dZ = dActivate(*(inputs[0].get()), *(m_savedA.get())); // == db
        auto dW = m_savedInput->transpose().matmul(dZ);
        dW *= avgCoeff;
        auto dA = std::make_shared<Tensor<T>>(std::move(dZ.matmul(m_kernel.transpose())));
        result.push_back(dA);
        m_kernel += applyKernelGrad(dW, -1.f*m_learningRate);
        dZ = dZ.sum();
        dZ *= avgCoeff;
        m_bias += applyBiasGrad(dZ, -1.f*m_learningRate);
        return result;
    }

    std::vector<size_t>& shape() override {
        return m_shape;
    }

    void set(std::string name, const std::shared_ptr<Tensor<T>>& val) override {
        if ("kernel" == name) {
            m_kernel.copy(*val.get());
        } else if ("bias" == name) {
            m_bias.copy(*val.get());
        } else {
            throw std::invalid_argument("Unknown name of parameter. See info() function to learn more");
        }
    }

    void info() override {
        std::cout << "==========<Dense layer>==========" << std::endl;
        std::cout << "|---> Available to set(): " << std::endl;
        std::cout << "    |---> kernel: ";
        m_kernel.info();
        std::cout << "    |---> bias: ";
        m_bias.info();
        std::cout << "|---> Activation: " << m_activation << std::endl;
        std::cout << "|---> output shape";
        std::cout << this->shapeToString() << std::endl;;
    }

    void reconstructionToStream(std::ostream& os, std::string graphId, std::string layerId, std::string prevLayerId) override {
        auto kRows = m_kernel.getRowsNumber();
        auto kCols = *(m_kernel.shape().rbegin());
        auto bCols = *(m_bias.shape().rbegin());
        // Construct layer
        os << "    auto " << layerId << " = " << graphId << " + std::make_shared<tg::Dense<" 
           << this->typeToString() << ">>(" << std::to_string(*(m_bias.shape().rbegin())) 
           << ", \"" << m_activation << "\", \"" << m_optimiser << "\", " 
           << std::to_string(m_learningRate) << ", " << prevLayerId << ");";
        // Initialisation of weights & bias
        os << "\n    static " << this->typeToString() << " " << layerId << "_kernel[";
        os << std::to_string(kRows) << "][" << std::to_string(kCols) << "] = {";
        for (size_t r = 0; r < kRows; ++r) {
            os << " {";
            auto kPtr = m_kernel.getRow(r);
            for (size_t i = 0; i < kCols; ++i) {
                os << std::to_string(kPtr[i]) << ",";
            }
            os << "},";
        }
        os << "};\n";
        os << "    " << "auto " + layerId << "_kernelTensor = std::make_shared<tg::Tensor<" 
           << this->typeToString() << ">>(tg::Tensor<" << this->typeToString() 
           << ">(" << layerId << "_kernel));\n";
        os << "    " << layerId << "->set(\"kernel\", " << layerId << "_kernelTensor);\n";
        os << "\n    static " << this->typeToString() << " " << layerId << "_bias[";
        os << std::to_string(1u) << "][" << std::to_string(bCols) << "] = { {";
        auto bPtr = m_bias.getRow(0);
        for (size_t i = 0; i < bCols; ++i) {
            os << std::to_string(bPtr[i]) << ",";
        }
        os << "} };\n";
        os << "    " << "auto " + layerId << "_biasTensor = std::make_shared<tg::Tensor<" 
           << this->typeToString() << ">>(tg::Tensor<" 
           << this->typeToString() << ">(" << layerId << "_bias));\n";
        os << "    " << layerId << "->set(\"bias\", " << layerId << "_biasTensor);\n";
    }
};

} // namespace tg
#endif // TG_DENSE
