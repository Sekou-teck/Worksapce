# for
for i in range(4):
    print("i a pour valeur", i)

for n in range (2, 100):
    for x in range (2, n):
        if n % x == 0:
            print(n, "égal", x, "*", n/x)
            break
    else:
        print(n, "est un nombre premier")

# while
i = 0
while i < 4:
    print("i est égal à", i)
    i += 1

# faire do while (faire tant ... que)
while True:
    n = int(input("Donnez un entier > 0 :"))
    print("Vous avez forni", n)
    if n > 0:
        break
print("Répose correcte")


