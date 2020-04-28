for n in range(0, 100):
    for x in range(2, n):
        if n % x ==0:
            print(n, "egale", x, "*", n/x)
            break
    else:
        print(n, "est un nombre premier")