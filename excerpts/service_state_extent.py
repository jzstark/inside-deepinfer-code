state_size = cls.total_bytes_inference if TO_CE else cls.state_size_to_disk
if state_size == 0:
    return None
src = state_cache[state_id, :state_size]
if state_tensor is None:
    return src
state_tensor.copy_(src)
return state_tensor
