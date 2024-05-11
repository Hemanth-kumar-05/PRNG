import streamlit as st
import numpy as np
from math import sqrt
from PIL import Image

# PRNG Algorithms
class LCG:
    def __init__(self, seed, a, c, m, additional_factor=0):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m
        self.additional_factor = additional_factor

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        if self.additional_factor == 0:
            return self.state
        else:
            return self.state % self.additional_factor;

class MersenneTwister:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        np.random.seed(self.state)
        self.state = np.random.randint(0, 2**31)
        return self.state
    
class XORShift:
    def __init__(self, seed, a, b, c, max_bits=32):
        self.state = seed
        self.a = a
        self.b = b
        self.c = c
        self.bitmask = 2**max_bits - 1

    def next(self):
        self.state ^= (self.state << self.a) & self.bitmask
        self.state ^= (self.state >> self.b)
        self.state ^= (self.state << self.c) & self.bitmask
        return self.state

class MiddleSquare:
    def __init__(self, seed, num_digits):
        self.state = seed
        self.num_digits = num_digits

    def next(self):
        self.state = int(str(self.state**2).zfill(2*self.num_digits)[self.num_digits//2:self.num_digits + self.num_digits//2])
        return self.state
    
class LaggedFibonacci:
    def __init__(self, seed, lag):
        self.state = seed
        self.lag = lag
        self.sequence = [seed]

    def next(self):
        if len(self.sequence) < self.lag:
            self.sequence.append(self.state)
            return self.state
        else:
            self.state = (self.sequence[-self.lag] + self.sequence[-self.lag+1]) % 2**32
            self.sequence.append(self.state)
            return self.state

class Lehmer:
    def __init__(self, seed, a, m):
        self.state = seed
        self.a = a
        self.m = m

    def next(self):
        self.state = (self.a * self.state) % self.m
        return self.state
    
class LFSR:
    def __init__(self, seed, taps):
        self.state = seed
        self.taps = taps

    def next(self):
        new_bit = 0
        for tap in self.taps:
            new_bit ^= (self.state >> tap) & 1
        self.state = (self.state >> 1) | (new_bit << 31)
        return self.state

# Generate random numbers
def generate_sequence(model, length):
    sequence = []
    for _ in range(length):
        sequence.append(model.next())
    return sequence

def generate_image(model, length, zoom_factor=1):
    sequence = generate_sequence(model, length)
    img_array = np.array(sequence).flatten()
    new_width = int(sqrt(length)) // zoom_factor
    new_height = int(sqrt(length)) // zoom_factor
    img_array = img_array[:new_width * new_height].reshape(new_height, new_width)
    img = Image.fromarray(img_array.astype(np.uint8))
    return img

# Streamlit App
st.title("Pseudo Random Number Generators")
st.write("A pseudorandom number generator (PRNG) is an algorithm for generating a sequence of numbers whose properties approximate the properties of sequences of random numbers. The PRNG-generated sequence is not truly random, because it is completely determined by an initial value, called the PRNG's seed (which may include truly random values). Although sequences that are closer to truly random can be generated using hardware random number generators, pseudorandom number generators are important in practice for their speed in number generation and their reproducibility.")

# Sidebar
sidebar = st.sidebar
sidebar.title("Pseudo Random Number Generators - Algorithms")
choose_prng = sidebar.multiselect("Choose a PRNG technique", ["Linear Congruential Generator", "Mersenne Twister", "XOR Shift", "Middle Square Method", "Lagged Fibonacci Generator", "Lehmer Random Number Generator", "Linear Feedback Shift Register"])
# Sidebar that has only square numbers
num_random_numbers = sidebar.selectbox("Width and Height", [100, 200, 256, 300])

# PRNG Algorithms

if "Linear Congruential Generator" in choose_prng:
    st.header("Linear Congruential Generator")
    st.write("The Linear Congruential Generator (LCG) is a simple pseudorandom number generator that generates a sequence of numbers calculated with a linear equation. The method represents one of the oldest and best-known pseudorandom number generator algorithms.")

    st.subheader("Parameters")
    input_col = st.columns(3)
    a = input_col[0].number_input("Multiplier (a)", min_value=0, value=1664525)
    c = input_col[1].number_input("Increment (c)", min_value=0, value=1013904223)
    m = input_col[2].number_input("Modulus (m)", min_value=0, value=4294967296)
    seed = st.slider("Seed", min_value=0, value=0, key="Seed LCG")
    lcg = LCG(seed, a, c, m)
    random_numbers = generate_image(lcg, num_random_numbers**2)
    show_image = st.checkbox("Show Image", key="Show Image LCG")
    if show_image:
        st.image(random_numbers, caption="Random Number Image", use_column_width=True)
    else:
        st.write("Random Numbers: ", np.array(random_numbers), key="Random Numbers LCG")

if "Mersenne Twister" in choose_prng:
    st.header("Mersenne Twister")
    st.write("The Mersenne Twister is a pseudorandom number generator (PRNG). It is by far the most widely used general-purpose PRNG. Its name derives from the fact that period length is chosen to be a Mersenne prime.")

    st.subheader("Parameters")
    seed = st.slider("Seed", min_value=0, value=0, key="Seed Mersenne Twister")
    mt = MersenneTwister(seed)
    random_numbers = generate_image(mt, num_random_numbers**2)
    show_image = st.checkbox("Show Image", key="Show Image Mersenne Twister")
    if show_image:
        st.image(random_numbers, caption="Random Number Image", use_column_width=True)
    else:
        st.write("Random Numbers: ", np.array(random_numbers), key="Random Numbers Mersenne Twister")


if "XOR Shift" in choose_prng:
    st.header("XOR Shift")
    st.write("Xorshift is a pseudorandom number generator introduced by George Marsaglia. It is based on a linear feedback shift register. It is one of the most efficient algorithms for generating random numbers with a very long period.")

    st.subheader("Parameters")
    input_col = st.columns(3)
    a = input_col[0].number_input("a", min_value=0, value=13)
    b = input_col[1].number_input("b", min_value=0, value=17)
    c = input_col[2].number_input("c", min_value=0, value=5)
    seed = st.slider("Seed", min_value=0, value=0, key="Seed XOR Shift")
    xorshift = XORShift(seed, a, b, c)
    random_numbers = generate_image(xorshift, num_random_numbers**2)
    show_image = st.checkbox("Show Image", key="Show Image XOR Shift")
    if show_image:
        st.image(random_numbers, caption="Random Number Image", use_column_width=True)
    else:
        st.write("Random Numbers: ", np.array(random_numbers), key="Random Numbers XOR Shift")

if "Middle Square Method" in choose_prng:
    st.header("Middle Square Method")
    st.write("The middle-square method is a method of generating pseudorandom numbers. In the middle-square method, the digits of the number are squared, and the middle digits of the result are used as the next number. The process is repeated until the number has enough digits.")

    st.subheader("Parameters")
    seed = st.slider("Seed", min_value=0, max_value=100000 ,value=0, key="Seed Middle Square Method")
    middle_square = MiddleSquare(seed, 4)
    random_numbers = generate_image(middle_square, num_random_numbers**2)
    show_image = st.checkbox("Show Image", key="Show Image Middle Square Method")
    if show_image:
        st.image(random_numbers, caption="Random Number Image", use_column_width=True)
    else:
        st.write("Random Numbers: ", np.array(random_numbers), key="Random Numbers Middle Square Method")

if "Lagged Fibonacci Generator" in choose_prng:
    st.header("Lagged Fibonacci Generator")
    st.write("The Lagged Fibonacci Generator is a pseudorandom number generator that generates a sequence of numbers calculated with a linear equation. The method represents one of the oldest and best-known pseudorandom number generator algorithms.")

    st.subheader("Parameters")
    seed = st.slider("Seed", min_value=0, value=0, key="Seed Lagged Fibonacci Generator")
    lag = st.number_input("Lag", min_value=0, value=2)
    lagged_fibonacci = LaggedFibonacci(seed, lag)
    random_numbers = generate_image(lagged_fibonacci, num_random_numbers**2)
    show_image = st.checkbox("Show Image", key="Show Image Lagged Fibonacci Generator")
    if show_image:
        st.image(random_numbers, caption="Random Number Image", use_column_width=True)
    else:
        st.write("Random Numbers: ", np.array(random_numbers), key="Random Numbers Lagged Fibonacci Generator")

if "Lehmer Random Number Generator" in choose_prng:
    st.header("Lehmer Random Number Generator")
    st.write("The Lehmer random number generator is a type of linear congruential generator. It is a simple pseudorandom number generator that generates a sequence of numbers calculated with a linear equation. The method represents one of the oldest and best-known pseudorandom number generator algorithms.")

    st.subheader("Parameters")
    input_col = st.columns(2)
    a = input_col[0].number_input("Multiplier (a)", min_value=0, value=13)
    m = input_col[1].number_input("Modulus (m)", min_value=0, value=100)
    seed = st.slider("Seed", min_value=0, value=0, key="Seed Lehmer")
    lehmer = Lehmer(seed, a, m)
    random_numbers = generate_image(lehmer, num_random_numbers**2)
    show_image = st.checkbox("Show Image", key="Show Image Lehmer")
    if show_image:
        st.image(random_numbers, caption="Random Number Image", use_column_width=True)
    else:
        st.write("Random Numbers: ", np.array(random_numbers), key="Random Numbers Lehmer")

if "Linear Feedback Shift Register" in choose_prng:
    st.header("Linear Feedback Shift Register")
    st.write("A linear-feedback shift register (LFSR) is a shift register whose input bit is a linear function of its previous state. The most commonly used linear function of single bits is exclusive-or (XOR). Thus, an LFSR is most often a shift register whose input bit is driven by the XOR of some bits of the overall shift register value.")

    st.subheader("Parameters")
    seed = st.slider("Seed", min_value=0, max_value=100000, value=0, key="Seed LFSR")
    taps = st.multiselect("Taps", [i for i in range(32)])
    lfsr = LFSR(seed, taps)
    random_numbers = generate_image(lfsr, num_random_numbers**2)
    show_image = st.checkbox("Show Image", key="Show Image LFSR")
    if show_image:
        st.image(random_numbers, caption="Random Number Image", use_column_width=True)
    else:
        st.write("Random Numbers: ", np.array(random_numbers), key="Random Numbers LFSR")
