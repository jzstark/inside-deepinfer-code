result = (latent_slice * score_slice.softmax(dim=0)).sum(dim=0).to(torch.bfloat16)
kv_compressed[compressed_start + blk_id - qstart // compress_ratio] = result
