for (int32_t i = 0; i < static_cast<int32_t>(seq_metadata_list.size()); ++i) {
    const auto *seq = seq_metadata_list[i];
    remap_ptr[i] = remap_indices[i];
    seqlens_q_ptr[i] = seq->seqlen_q;
    seqlens_k_ptr[i] = seq->seqlen_k - (nextn > 0 ? seq->seqlen_q : 0);
}
for (int64_t i = seq_metadata_list.size(); i < n_seq_with_padding; ++i) {
    remap_ptr[i] = -1;
    seqlens_q_ptr[i] = 1;
    seqlens_k_ptr[i] = 1;
}
