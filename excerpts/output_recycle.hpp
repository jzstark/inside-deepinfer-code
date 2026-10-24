void release(py::array_t<uint64_t> ptrs_py) {
    const uint64_t *ptrs = ptrs_py.data();
    int size = ptrs_py.size();
    for (int i = 0; i < size; ++i) {
        auto p = reinterpret_cast<RequestOutput *>(ptrs[i]);
        std::destroy_at(p);
        free_.push_back(p);
    }
}
