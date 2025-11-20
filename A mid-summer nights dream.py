while True:
    try:
        loop=int(input())
        a=[]
        for _ in range(loop):
            n=int(input())
            a.append(n)
        a.sort()
        ct=0
        if len(a)%2==1:
            mid1=a[len(a)//2]
            mid2=a[len(a)//2]
        else:
            mid1=a[len(a)//2-1]
            mid2=a[len(a)//2]
        for i in range(len(a)):
            if a[i]==mid1 or a[i]==mid2:
                ct+=1
        print(mid1,ct,mid2-mid1+1)
    except EOFError:
        break