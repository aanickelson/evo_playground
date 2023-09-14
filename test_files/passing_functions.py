

def fun1(x):
    return x + 1

def fun2(x):
    return x * 2

def fun3(f0, x):
    return f0(x)

if __name__ == '__main__':
    funs = [fun1, fun2]
    for f in funs:
        print(fun3(f, 5))
