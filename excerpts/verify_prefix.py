req = T.alloc_var(T.int32, row_to_req[row])
start = T.alloc_var(T.int32, ftp[req])
alive = T.alloc_var(T.bool, True)
prev = T.alloc_var(T.int32, start)
while alive and prev < row:
    prev_accept = T.alloc_var(T.bool, raw_accept[prev])
    if has_golden:
        prev_golden = T.alloc_var(T.int32, golden[prev])
        if prev_golden != -1:
            prev_draft_row = T.alloc_var(T.int32, read_rows[prev])
            prev_accept = prev_golden == draft_token_ids_pool[prev_draft_row]
    alive = is_verify[prev] and prev_accept
    prev += 1
token_ids_out[row] = token
accept_out[row] = alive and is_verify[row] and accept
is_final_out[row] = alive and (not is_verify[row] or not accept)
