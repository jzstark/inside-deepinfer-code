"""The unified entry for model execution: every step (prompt/decode, ZOS on/off,
placeholder) goes through this Forward-Post_process-Sample round.
  1. Forward current step (main model)
  2. Post-process previous step
  3. Sample current step
  4. Post-process current step: in place for prompt / non-ZOS steps;
     ZOS decode steps defer it to the next step to overlap with forward
