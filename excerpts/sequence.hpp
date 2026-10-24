void append_token_id(int token_id) { token_ids.push_back(token_id); }
int get_len() const { return static_cast<int>(token_ids.size()); }
int get_prompt_len() const { return prompt_len; }
int get_output_len() const { return get_len() - prompt_len; }
const std::vector<int> &get_token_ids() const { return token_ids; }
int get_last_token_id() const { return token_ids.back(); }

std::vector<int> token_ids;
int prompt_len{0};
