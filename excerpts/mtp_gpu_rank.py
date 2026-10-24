# Higher rate wins; tie-break by flat index (lower index wins)
for j in T.serial(nextn):
    my_flat_idx = T.alloc_var(T.int32)
    my_flat_idx = pid * nextn + j
    beats = T.alloc_var(T.int32)
    beats = 0
    if other_rate > my_rates[j]:
        beats = 1
    if other_rate == my_rates[j] and other_idx < my_flat_idx:
        beats = 1
    rank_reducer[j] += beats
