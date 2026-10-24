reselected_swa = _stored_swa(reselected_state)
modified_swa = _stored_swa(modified_state)
assert torch.equal(reselected_swa, modified_swa)
