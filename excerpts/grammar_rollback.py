        num_accepted_steps += 1
    else:
        break
matcher.rollback(num_accepted_steps)
