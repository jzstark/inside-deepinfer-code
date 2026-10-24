void add_sequence(std::shared_ptr<Sequence> seq) {
    seq->request.arrival_time = get_timestamp_ms();
    if (engine_type == EngineType::NORMAL) {
        seq->reserve_output_tokens();
        seq->cache.block_table.reserve(_block_table_capacity(*seq));
    }
    waiting.push_back(std::move(seq));
}
