else:
    for j in T.Parallel(B):
        index[j] = offset // block_size * block_size + (offset + i * B + j) % block_size
for j in T.Parallel(B):
    index[j] = T.if_then_else(i * B + j < length, index[j], -1)
T.copy(index, kv_index[bx, i * B : (i + 1) * B])
