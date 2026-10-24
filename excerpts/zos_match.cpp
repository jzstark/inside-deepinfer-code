bool has_overlap = same_order && !seq_metadata_list.empty();
if (!same_order) {
    std::unordered_map<int64_t, int32_t> prev_indices;
    prev_indices.reserve(prev_seq_metadata_list.size());
    for (int32_t i = 0; i < static_cast<int32_t>(prev_seq_metadata_list.size()); ++i) {
        if (!prev_seq_metadata_list[i]->is_placeholder) {
            prev_indices.insert_or_assign(prev_seq_metadata_list[i]->seq_id, i);
        }
    }
    for (int32_t i = 0; i < static_cast<int32_t>(seq_metadata_list.size()); ++i) {
        auto prev = prev_indices.find(seq_metadata_list[i]->seq_id);
        remap_indices[i] = prev == prev_indices.end() ? -1 : prev->second;
        has_overlap |= remap_indices[i] >= 0;
    }
}
if (!has_overlap) {
    return {std::nullopt, std::nullopt, std::nullopt, false};
}
