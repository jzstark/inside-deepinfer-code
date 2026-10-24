auto slot_mapping_store_swa = get_or_create_tensor(result, has_out, "slot_mapping_store_swa", {total_q}, torch::kInt32);
auto p_slot_mapping_store_swa = slot_mapping_store_swa.mutable_data_ptr<int32_t>();
int32_t i_q = 0;
for (auto p_seq : seq_metadata_list) {
    for (int32_t j = 0; j < p_seq->seqlen_q; ++j, ++i_q) {
        if (p_seq->seqlen_q - j <= swa_block_size)
            p_slot_mapping_store_swa[i_q] = p_seq->state_id * swa_block_size + (p_seq->seqlen_k - p_seq->seqlen_q + j) % swa_block_size;
        else
            p_slot_mapping_store_swa[i_q] = -1;
    }
}
TORCH_CHECK_EQ(i_q, total_q);
