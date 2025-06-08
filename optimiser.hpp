

#ifndef TG_OPTIMISER
#define TG_OPTIMISER

#include "tensor.hpp"
#include <cmath>

namespace tg
{
#define ADAM_B1 (0.9f)
#define ADAM_B2 (0.999f)
#define ADAM_EPS (1e-7f)

template<typename T>
class Adam {
    Tensor<T> m_firstEst;
    Tensor<T> m_secondEst;
    size_t m_counter{1};
public:
    Adam() = default;
    ~Adam() = default;

    Adam(const Adam&) = delete;
    Adam operator=(const Adam&) = delete;

    Adam(Adam&& other) noexcept {
        m_firstEst = std::move(other.m_firstEst);
        m_secondEst = std::move(other.m_secondEst);
        m_counter = std::exchange(other.m_counter, 0);
    }

    Adam& operator=(Adam&& other) noexcept {
        if (this == &other) return *this;
        m_firstEst = std::move(other.m_firstEst);
        m_secondEst = std::move(other.m_secondEst);
        m_counter = std::exchange(other.m_counter, 0);
        return *this;
    }

    Adam(Tensor<T>& target) :
        m_firstEst{std::move(target.copy())},
        m_secondEst{std::move(target.copy())} {
        static_assert(std::is_floating_point<T>::value, "Adam optimiser supports floating point only");
    }
    
    Tensor<T> estimate(Tensor<T>& grad) {
        m_firstEst *= ADAM_B1;
        m_firstEst += grad * ((T)1 - ADAM_B1);
        m_secondEst *= ADAM_B2;
        grad = std::move(grad.square());
        grad *= ((T)1 - ADAM_B2);
        m_secondEst += grad;
        auto firstEst_hat = m_firstEst * ((T)1 / ((T)1 - (T)std::pow(ADAM_B1, m_counter)));
        auto secondEst_hat = m_secondEst * ((T)1 / ((T)1 - (T)std::pow(ADAM_B2, m_counter)));
        secondEst_hat = std::move(secondEst_hat.sqrt());
        secondEst_hat += ADAM_EPS;
        ++m_counter;
        firstEst_hat /= secondEst_hat;
        return firstEst_hat;
    }

    void reset() {
        m_firstEst.setZeros();
        m_secondEst.setZeros();
        m_counter = 1;
    }

};

} // namespace tg


#endif // TG_OPTIMISER