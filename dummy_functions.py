from multiprocessing import Process

a = 0


def func_print(item):
    for i in range(5):
        print(item)


if __name__ == "__main__":
    items = ['Hi', 'Hello', 'Bye']
    for item in items:
        p = Process(target=func_print, args=(item,))
        p.start()
