request_outputs = self._post_process_step(self.prev_step_state)
self.prev_step_state = None
return request_outputs
