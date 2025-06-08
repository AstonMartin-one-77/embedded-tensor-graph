
#ifndef TG_LAYER
#define TG_LAYER

#include "tensor.hpp"
#include <memory>
#include <cstdint>
#include <string>

namespace tg
{

template <typename T>
class Layer {
public:
    virtual ~Layer() = default;
    virtual const std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) = 0;
    virtual const std::vector<std::shared_ptr<Tensor<T>>> backward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) = 0;
    virtual void set(std::string name, const std::shared_ptr<Tensor<T>>& val) = 0;
    virtual std::vector<size_t>& shape() = 0;
    virtual void info() = 0;
    virtual void reconstructionToStream(std::ostream& os, std::string graphId, std::string layerId, std::string prevLayerId) = 0;

    std::string typeToString() {
        if constexpr (std::is_same<T, float>::value) {
            return std::string{"float"};
        } else if (std::is_same<T, double>::value) {
            return std::string{"double"};
        } else if (std::is_same<T, int32_t>::value) {
            return std::string{"int32_t"};
        } else if (std::is_same<T, int16_t>::value) {
            return std::string{"int16_t"};
        } else if (std::is_same<T, int8_t>::value) {
            return std::string{"int8_t"};
        } else if (std::is_same<T, uint8_t>::value) {
            return std::string{"uint8_t"};
        } else {
            throw std::invalid_argument("Unsupported data type");
        }
    }

    std::string shapeToString() {
        std::string result{"{"};
        for (auto dim: shape()) {
            result += std::to_string(dim);
            result += ",";
        }
        result += "}";
        return result;
    }
};

} // namespace tg
#endif // TG_LAYER