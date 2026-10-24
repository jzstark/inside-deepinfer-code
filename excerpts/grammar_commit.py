try:
    if so_status[idx] == 2:
        for token in seq_metadata.last_accepted_token + [seq_metadata.last_output_token]:
            if not matcher.accept_token(token):
                raise RuntimeError(f"matcher rejected step token {token}, upstream invariant broken")
    elif so_status[idx] == 1:
        so_status[idx] = 2
    else:
        raise RuntimeError(f"unexpected so_status {so_status[idx]}")
