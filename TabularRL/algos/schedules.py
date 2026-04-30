def constant_schedule(value):
    def schedule(t):
        return value
    return schedule


def inverse_schedule(c=1.0, min_value=0.01):
    def schedule(t):
        return max(min_value, c / max(1, t))
    return schedule


def linear_decay_schedule(start=1.0, end=0.05, decay_steps=1000):
    def schedule(t):
        fraction = min(t / decay_steps, 1.0)
        return start + fraction * (end - start)
    return schedule
