def add(a,b):
	print(f"adding {a} and {b}")
	return a+ b

def subtracting(a,b):
	print(f"subtracting {b} from {a}")
	return a - b

def multiply(a,b):
	print(f"mutl {b} with {a}")
	return a * b

def divide(a,b):
	print(f"div {b} by {a}")
	return b/a

print("let's do some math")
age = add(30,2)
height = subtracting(180, 2)
iq = divide(2, 200)

print(f''' My age is {age}
	my height is {height}
	my iq is {iq}''')