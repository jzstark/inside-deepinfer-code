assert self.yoco_compaction_enabled
view = copy.copy(self)
qi = self.yoco_consumer_query_indices
view.total_q = self.yoco_consumer_total_q
view.cu_seqlens_q = self.yoco_consumer_cu_seqlens_q
view.positions = self.yoco_consumer_positions
view.slot_mapping_store_swa = self.yoco_consumer_slot_mapping_store_swa
view.ks = self.yoco_consumer_ks
view.ke = self.yoco_consumer_ke
view.kv_ca_index_offset_len = self.yoco_consumer_kv_ca_index_offset_len
