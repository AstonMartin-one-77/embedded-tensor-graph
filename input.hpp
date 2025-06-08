
#ifndef TG_INPUT
#define TG_INPUT

#include "layer.hpp"

namespace tg
{
template <typename T>
class Input : public Layer<T> {
private:
    std::vector<size_t> m_shape;
    
public:
    Input() = default;
    ~Input() = default;

    Input(const std::vector<size_t>& shape): m_shape{shape} {}

    Input(const Input& other) = delete;
    Input& operator=(const Input& other) = delete;

    Input(Input&& other) noexcept {
        m_shape = std::move(other.m_shape);
    }

    Input& operator=(Input&& other) noexcept {
        if (this == &other) return *this;
        m_shape = std::move(other.m_shape);
        return this;
    }

    const std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        return std::move(inputs);
    }
    
    const std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        return std::move(inputs);
    }

    std::vector<size_t>& shape() override {
        return m_shape;
    }

    void set(std::string name, const std::shared_ptr<Tensor<T>>& val) override {
        throw std::invalid_argument("Unsupported method for Input layer");
    }

    void info() override {
        std::cout << "==========<Input layer>==========" << std::endl;
        std::cout << "|---> shape{";
        for (auto i : m_shape) std::cout << i << ",";
        std::cout << "}" << std::endl;
    }

    void reconstructionToStream(std::ostream& os, std::string graphId, std::string layerId, std::string prevLayerId) override {
        // Construct layer
        os << "    auto " << layerId << " = " << graphId << " + std::make_shared<tg::Input<" 
           << this->typeToString() << ">>(" << "std::vector<size_t>" << this->shapeToString() << ");" << std::endl;
    }
};
    
} // namespace tg

#endif // TG_INPUT
