dico = {}

a = ['apple', 'orange', 'banana']
b = ['red', 'orange', 'yellow']

for i in range(len(a)):
	dico[a[i]] = b[i]

print(dico)