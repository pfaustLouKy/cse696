# Define the secret string s
s = '101'
n = len(s)

# Dictionary to store f(x) values
oracle_outputs = {}

# Counter to generate unique outputs
output_counter = 0

# Function to simulate Simon's oracle
def f(x):
    global output_counter
    x_int = int(x, 2)
    s_int = int(s, 2)
    x_xor_s = x_int ^ s_int
    x_xor_s_bin = format(x_xor_s, f'0{n}b')

    # Ensure both x and x⊕s map to the same output
    if x in oracle_outputs:
        return oracle_outputs[x]
    elif x_xor_s_bin in oracle_outputs:
        oracle_outputs[x] = oracle_outputs[x_xor_s_bin]
        return oracle_outputs[x]
    else:
        # Assign a new unique output
        output = format(output_counter, '05b')  # 5-bit output
        oracle_outputs[x] = output
        oracle_outputs[x_xor_s_bin] = output
        output_counter += 1
        return output

# Test
x = '000'
print(f"f({x}) = {f(x)}")
