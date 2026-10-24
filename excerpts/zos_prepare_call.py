remap_cpu, seqlens_q_cpu, seqlens_k_cpu, has_overlap = prepare_zos_inputs(
    seq_metadata_list, prev.seq_metadata_list, n_seq_with_padding, self.nextn
)
if not has_overlap:
    return None
remap = remap_cpu.cuda(non_blocking=True)
seqlens_q = seqlens_q_cpu.cuda(non_blocking=True)
seqlens_k = seqlens_k_cpu.cuda(non_blocking=True)
