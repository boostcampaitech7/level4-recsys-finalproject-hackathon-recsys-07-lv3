#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <vector>
#include <stdexcept>
#include <string>
#include <iostream>

namespace py = pybind11;

py::array_t<int> get_samples_cpp(int user_cnt,
                                 int item_cnt,
                                 py::array_t<int, py::array::c_style | py::array::forcecast> sample_num_per_user_arr,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> neg_item_probs,
                                 py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
                                 py::array_t<int, py::array::c_style | py::array::forcecast> indices) {
    auto buf_neg = neg_item_probs.request();
    auto buf_indptr = indptr.request();
    auto buf_indices = indices.request();
    auto buf_sample = sample_num_per_user_arr.request();

    if (buf_neg.ndim != 2)
        throw std::runtime_error("neg_item_probs must be a 2-dimensional array.");
    if (buf_indptr.ndim != 1)
        throw std::runtime_error("indptr must be a 1-dimensional array.");
    if (buf_indices.ndim != 1)
        throw std::runtime_error("indices must be a 1-dimensional array.");
    if (buf_sample.ndim != 1)
        throw std::runtime_error("sample_num_per_user_arr must be a 1-dimensional array.");

    if (buf_neg.shape[0] != static_cast<size_t>(user_cnt) || buf_neg.shape[1] != static_cast<size_t>(item_cnt))
        throw std::runtime_error("The shape of neg_item_probs does not match user_cnt and item_cnt.");
    if (buf_sample.shape[0] != static_cast<size_t>(user_cnt))
        throw std::runtime_error("sample_num_per_user_arr must have length equal to user_cnt.");

    float* neg_ptr = static_cast<float*>(buf_neg.ptr);
    int*   indptr_ptr = static_cast<int*>(buf_indptr.ptr);
    int*   indices_ptr = static_cast<int*>(buf_indices.ptr);
    int*   sample_ptr = static_cast<int*>(buf_sample.ptr);

    int total_samples = 0;
    for (int u = 0; u < user_cnt; u++) {
        total_samples += sample_ptr[u];
    }

    auto result = py::array_t<int>({ total_samples, 3 });
    auto buf_result = result.request();
    int* result_ptr = static_cast<int*>(buf_result.ptr);

    std::random_device rd;
    std::mt19937 gen(rd());

    int offset = 0;
    for (long long u = 0; u < user_cnt; u++) {
        int start = indptr_ptr[u];
        int end = indptr_ptr[u + 1];
        int pos_count = end - start;

        if (pos_count <= 0)
            throw std::runtime_error("User " + std::to_string(u) + " does not have any positive items.");

        std::vector<float> weights(item_cnt);
        for (long long j = 0; j < item_cnt; j++) {
            weights[j] = neg_ptr[u * item_cnt + j];
        }
        std::discrete_distribution<> neg_dist(weights.begin(), weights.end());
        std::uniform_int_distribution<> pos_dist(0, pos_count - 1);

        int cur_sample_count = sample_ptr[u];

        for (int s = 0; s < cur_sample_count; s++) {
            int pos_idx = pos_dist(gen);
            int pos_item = indices_ptr[start + pos_idx];
            int neg_item = neg_dist(gen);

            result_ptr[offset * 3 + 0] = u;
            result_ptr[offset * 3 + 1] = pos_item;
            result_ptr[offset * 3 + 2] = neg_item;
            offset++;
        }
    }
    return result;
}


PYBIND11_MODULE(sampler_cpp, m) {
    m.doc() = "Accelerated sample generation module using pybind11";
    m.def("get_samples_cpp", &get_samples_cpp,
          "Generates sample pairs using a CSR matrix and negative item probabilities, "
          "with a per-user specified number of sample pairs.",
          py::arg("user_cnt"),
          py::arg("item_cnt"),
          py::arg("sample_num_per_user_arr"),
          py::arg("neg_item_probs"),
          py::arg("indptr"),
          py::arg("indices"));
}
