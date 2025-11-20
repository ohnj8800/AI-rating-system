def nine(n):
    ct=0
    if n==9:
        ct+=1
    else:
        while n>=10:
            sum=0
            for i in range(len(str(n))):
                sum=sum+n%10
                n=n//10
            n=sum
            ct+=1
    if n%9==0:
        return (ct)
    else:
        return (0)
while True:
    n=int(input())
    if n==0:
        break
    if nine(n)==0:
        print(n,"is not a multiple of 9.")
    else:
        print(n,"is a multiple of 9 and has 9-degree %d."%(nine(n)))