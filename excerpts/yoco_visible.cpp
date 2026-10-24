int32_t abs_lo = std::clamp(std::max({pos - window_size + 1, rect_lo, valid_start}), 0, pos + 1);
