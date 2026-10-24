recv_slot = dict(experts_to_recv)
for i in range(num_local):
    if origin_local[i] == current_local[i]:
        continue
    src = i if received[i] else recv_slot[current_local[i]]
    for weight, buf in zip(weights, recv_buffers, strict=True):
        weight[i].copy_(buf[src])
