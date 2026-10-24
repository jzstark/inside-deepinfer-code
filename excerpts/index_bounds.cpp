for (int r : args.sparse_compress_ratios) {
    py::str name = std::to_string(r);
    auto klen = get_or_create_tensor(klen_dict, has_out, name, {total_q}, torch::kInt32);
    auto p_klen = klen.mutable_data_ptr<int32_t>();
    i_q = 0;
    for (auto p_seq : seq_metadata_list) {
        for (int32_t j = p_seq->seqlen_k - p_seq->seqlen_q; j < p_seq->seqlen_k; ++j, ++i_q) {
            p_klen[i_q] = (j + 1) / r;
        }
    }
    TORCH_CHECK_EQ(i_q, total_q);
