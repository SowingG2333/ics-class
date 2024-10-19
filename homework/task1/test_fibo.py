print('Hello World!')

a = 1
b = 1

def printFibo(n):
    global a, b
    if n == 1:
        print(a)
        return
    elif n == 2:
        print(a)
        print(b)
        return
    elif a == 1 & b == 1:
        print(a)
        print(b)
    count = 0
    c = a + b
    print(c)
    a = b
    b = c
    if not n == 0:
        printFibo(n-1)
    else:
        return

printFibo(10)
