import random

seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def random_string(num=16):
    return ''.join(random.sample(seed, num))
