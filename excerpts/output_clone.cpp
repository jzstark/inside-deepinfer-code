py::class_<RequestOutput>(engine, "RequestOutput")
    .def_static(
        "clone", [](uint64_t ptr) { return std::make_unique<RequestOutput>(*reinterpret_cast<RequestOutput *>(ptr)); }, py::arg("ptr"))
