import math

while True:
    a, b = list(map(int, input().split()))

    if a == 0 and b == 0:
        break 

    else:
        start = min(a, b)
        end = max(a, b) + 1
        square_root = 0

        for i in range(start, end):
            if (math.sqrt(i) - int(math.sqrt(i)) == 0):
                square_root += 1
        
        print(square_root)