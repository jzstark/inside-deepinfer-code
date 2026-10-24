if (seq.structured_output.status == -1) {
    seq.schedule.status = SequenceStatus::FINISHED_SO_FAILED;
    seq.append_token_id(0);
    seq.append_compressed_token_id(0);
    token_cnt++;
