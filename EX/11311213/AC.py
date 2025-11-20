import math

def count_square_numbers(a, b):
    return math.floor(math.sqrt(b)) - math.ceil(math.sqrt(a)) + 1

while True:
    a, b = map(int, input().split())
    if a == 0 and b == 0:
        break
    print(count_square_numbers(a, b))
