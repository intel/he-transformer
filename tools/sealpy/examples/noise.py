import numpy as np
import pyseal
import time
import random

from pyseal import EncryptionParameters, \
                  scheme_type, \
                  SEALContext, \
                  KeyGenerator, \
                  Encryptor, \
                  Evaluator, \
                  Decryptor, \
                  CKKSEncoder, \
                  Plaintext, \
                  Ciphertext, \
                  MemoryPoolHandle, \
                  DoubleVec, \
                  ComplexVec

import math

#P here is used in the re-linearization process. We increase the modulus,
#then we key-switch, then we mod switch. The value of P right now is a
#guesstimate.
P = 2**100

message1 = 4
message2 = 4
sigma = 3.2
N = 8192
h = 64
l = 3
q = 2**218
p = 2**60


#noise bound on a freshly encrypted ciphertext
def fresh():
    return 8 * math.sqrt(2) * sigma * N + 6 * sigma * math.sqrt(
        N) + 16 * sigma * math.sqrt(h * N)


#noise bound on adding two ciphertexts
def add(noise1, noise2):
    return noise1 + noise2


#noise bound on key switching operation
def key_switch():
    return 8 * sigma * N / math.sqrt(3)


#noise bound on re-scaling operation
def rescale():
    return math.sqrt(N / 3) * (3 + 8 * math.sqrt(h))


#noise bound on multiplication, assuming we relinearize straight away
def mult(message1, message2, noise1, noise2):
    return noise1 * message2 + noise2 * message1 + noise1 * noise2


def mult_rescale(noise):
    return noise + (1 / P) * q * key_switch() + rescale()


def noise_budget(q, l, p, noise):
    return math.log(q, 2) - math.log(noise, 2) + l * math.log(p, 2) - 1


def simple_circuit_noise():
    noise = fresh()
    noise = add(noise, noise)
    noise = mult(message1, message2, noise, noise)
    return noise_budget(q, l, p, noise)


def l_mults_circuit_noise_budget():
    noise = fresh()
    level = l
    modulus = q
    while level > 0:
        print("\nThe input noise budget is: " +
              str(noise_budget(modulus, level, p, noise)))
        noise = add(noise, noise)
        noise = mult(message1, message2, noise, noise)
        print(
            "\nThe noise budget after a multiplication and a relinearizaton is: "
            + str(noise_budget(modulus, level, p, noise)))
        modulus = int(modulus / p)
        level -= 1


def new_noise_budget(message, scale, noise):
    return message * scale - noise


def l_mults_circuit_new_noise_budget():
    noise = fresh()
    level = l
    scale = p
    m = 4
    while level > 0:
        print("\nThe input noise budget is: " +
              str(new_noise_budget(message1, p, noise)))
        noise = add(noise, noise)
        noise = mult(message1, message2, noise, noise)
        print(
            "\nThe noise budget after a multiplication and a relinearizaton is: "
            + str(new_noise_budget(message1, p, noise)))
        level -= 1
