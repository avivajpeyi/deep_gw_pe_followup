

def get_m1_ln_weights(m1_vals, orig_pri, new_pri):
    return new_pri.ln_prob(m1_vals) - orig_pri.ln_prob(m1_vals)
