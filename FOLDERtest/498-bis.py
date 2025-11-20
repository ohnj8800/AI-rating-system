while True:
    try:
        x=int(input())
        a=list(map(int,input().split()))
        n=a[::-1]
        sum=0
        for i in range(1,len(a)):
            sum+=i*n[i]*x**(i-1)
        print(sum)
    except EOFError:
        break