if is_prompt or not envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING:
    request_outputs = np.concatenate([request_outputs, self._post_process_step(curr_step_state)])
    self.prev_step_state = None
else:
    self.prev_step_state = curr_step_state

return request_outputs
