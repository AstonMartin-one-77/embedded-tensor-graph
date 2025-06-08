
// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../graph.hpp"
#include "../input.hpp"
#include "../dense.hpp"
#include "../layer.hpp"
#include "../tensor.hpp"

namespace py = pybind11;

template <typename T>
std::shared_ptr<tg::Tensor<T>> numpy_to_tensor(const py::array_t<T>& input) {
    auto buffer = input.request();
    std::vector<size_t> shape{buffer.shape.begin(), buffer.shape.end()};
    auto tensor = std::make_shared<tg::Tensor<T>>(shape);

    size_t numRows = tensor->getRowsNumber();

    auto np_ptr = static_cast<float*>(buffer.ptr);
    size_t np_rowStride = (buffer.strides.size() > 1u) ? *(buffer.strides.rbegin()+1u)/sizeof(T) : 0; // elems, not bytes
    size_t np_colStride = *(buffer.strides.rbegin())/sizeof(T); // elems, not bytes
    if ((numRows > 1u) && (0 == np_rowStride)) throw std::runtime_error("Wrong NumPy stride");
    if (np_rowStride*numRows*sizeof(T) > (size_t)input.nbytes()) throw std::runtime_error("Wrong NumPy stride: out of memory");
    
    for (size_t r = 0; r < numRows; ++r) {
        T* rowPtr = tensor->getRow(r);
        for (size_t i = 0; i < *(shape.rbegin()); ++i) {
            // Note: to cover virtual transpose of numpy, strides are used instead of direct copy of rows (see tensor_to_numpy)
            rowPtr[i] = np_ptr[np_colStride*i];
        }
        np_ptr += np_rowStride;
    }
    return tensor;
}

template <typename T>
py::array_t<T> tensor_to_numpy(const std::shared_ptr<tg::Tensor<T>>& tensor) {
    auto shape = tensor->shape();
    auto np_array = py::array_t<float>(shape);
    auto buffer = np_array.request();
    
    size_t row_size = *shape.rbegin() * sizeof(T);
    size_t numRows = tensor->getRowsNumber();

    auto np_ptr = reinterpret_cast<std::byte*>(buffer.ptr);
    size_t np_stride = (buffer.strides.size() > 1u) ? *(buffer.strides.rbegin()+1u) : 0;
    if ((numRows > 1u) && (0 == np_stride)) throw std::runtime_error("Wrong NumPy stride");
    if (np_stride*numRows > (size_t)np_array.nbytes()) throw std::runtime_error("Wrong NumPy stride: out of memory");
    
    for (size_t r = 0; r < numRows; ++r) {
        memcpy(np_ptr, tensor->getRow(r), row_size);
        np_ptr += np_stride;
    }
    return np_array;
}

template <typename T>
std::vector<std::shared_ptr<tg::Tensor<T>>> getTensorFromPyArgs(py::args args) {
    std::vector<std::shared_ptr<tg::Tensor<T>>> tensors;

    if ((1u == args.size()) && py::isinstance<py::list>(args[0])) {
        for (const auto& arg : args[0]) {
            if (!py::isinstance<py::array>(arg)) throw std::invalid_argument("Unsupported input");
            tensors.push_back(numpy_to_tensor<T>(arg.cast<py::array_t<T>>()));
        }
    } else if ((1u == args.size()) && py::isinstance<py::array>(args[0])) {
        tensors.push_back(numpy_to_tensor<T>(args[0].cast<py::array_t<T>>()));
    } else {
        for (const auto& arg : args) {
            if (!py::isinstance<py::array>(arg)) throw std::invalid_argument("Unsupported input");
            tensors.push_back(numpy_to_tensor<T>(arg.cast<py::array_t<T>>()));
        }
    }
    return tensors;
}

template <typename T>
std::vector<py::array_t<T>> forward_np_wrapper(std::shared_ptr<tg::Layer<float>> self, py::args args) {
    auto tensors = getTensorFromPyArgs<T>(args);
    auto t_outputs = self->forward(tensors);
    std::vector<py::array_t<T>> np_outputs;
    for (const auto& t : t_outputs) { np_outputs.push_back(tensor_to_numpy<T>(t)); }
    return np_outputs;
}

template <typename T>
std::vector<py::array_t<T>> backward_np_wrapper(std::shared_ptr<tg::Layer<float>> self, py::args args) {
    auto tensors = getTensorFromPyArgs<T>(args);
    auto t_outputs = self->backward(tensors);
    std::vector<py::array_t<T>> np_outputs;
    for (const auto& t : t_outputs) { np_outputs.push_back(tensor_to_numpy<T>(t)); }
    return np_outputs;
}

template <typename T>
std::vector<py::array_t<T>> fit_np_wrapper(std::shared_ptr<tg::Graph<float>> self, py::array_t<T>& input, py::array_t<T>& labels, std::string loss) {
    std::vector<std::shared_ptr<tg::Tensor<T>>> inTensor;
    std::vector<std::shared_ptr<tg::Tensor<T>>> labelsTensor;
    inTensor.push_back(numpy_to_tensor<T>(input));
    labelsTensor.push_back(numpy_to_tensor<T>(labels));
    auto t_outputs = self->fit(inTensor, labelsTensor, loss);
    std::vector<py::array_t<T>> np_outputs;
    for (const auto& t : t_outputs) { np_outputs.push_back(tensor_to_numpy<T>(t)); }
    return np_outputs;
}

template <typename T>
void set_np_wrapper(std::shared_ptr<tg::Layer<float>> self, std::string name, py::array_t<T>& val) {
    self->set(name, numpy_to_tensor<T>(val));
}

#include <fstream>

void toFile(std::shared_ptr<tg::Graph<float>> self, std::string fileName, std::string graphId) {
    std::ofstream file(fileName); 
    if (!file) throw std::runtime_error("File is not opened");
    self->reconstructionToStream(file, graphId, "", "");
}

PYBIND11_MODULE(embeddedtensor, m) {

    py::class_<tg::Tensor<float>, std::shared_ptr<tg::Tensor<float>>>(m, "Tensor");
    
    py::class_<tg::Layer<float>, std::shared_ptr<tg::Layer<float>>>(m, "Layer");

    py::class_<tg::Graph<float>,  tg::Layer<float>, std::shared_ptr<tg::Graph<float>>>(m, "Graph")
        .def(py::init<>())
        .def("__add__", &tg::Graph<float>::operator+)
        .def("forward", &forward_np_wrapper<float>)
        .def("backward", &backward_np_wrapper<float>)
        .def("fit", &fit_np_wrapper<float>, py::arg("input"), py::arg("labels"), py::arg("loss"))
        .def("toFile", &toFile, py::arg("fileName"), py::arg("graphId"))
        .def("info", &tg::Graph<float>::info);

    py::class_<tg::Input<float>, tg::Layer<float>, std::shared_ptr<tg::Input<float>>>(m, "Input")
        .def(py::init<>())
        .def(py::init<const std::vector<size_t>&>())
        .def("forward", &forward_np_wrapper<float>)
        .def("backward", &backward_np_wrapper<float>)
        .def("info", &tg::Input<float>::info)
        .def("shape", &tg::Input<float>::shape, py::return_value_policy::reference);

    py::class_<tg::Dense<float>, tg::Layer<float>, std::shared_ptr<tg::Dense<float>>>(m, "Dense")
        .def(py::init<>())
        .def(py::init<size_t, const std::shared_ptr<tg::Layer<float>>&>())
        .def(py::init<size_t, const std::string, const std::shared_ptr<tg::Layer<float>>&>())
        .def(py::init<size_t, const std::string, float, const std::shared_ptr<tg::Layer<float>>&>())
        .def(py::init<size_t, const std::string, std::string, float, const std::shared_ptr<tg::Layer<float>>&>())
        .def("forward", &forward_np_wrapper<float>)
        .def("backward", &backward_np_wrapper<float>)
        .def("info", &tg::Dense<float>::info)
        .def("set", &set_np_wrapper<float>, py::arg("name"), py::arg("val"))
        .def("shape", &tg::Dense<float>::shape, py::return_value_policy::reference);

}
