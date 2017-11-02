import random

wordlist = ['cat', 'dog', 'cartoon', 'tennis', 'soccer', 'ballet', 'bedroom',
 'bathroom', 'garden', 'park', 'swimmimng pool', 'living room', 'house',
 'orange', 'pineapple', 'watermelon', 'pumpkin', 'halloween', 'thanksgiving',
 'table', 'surfing', 'elephant', 'giraffe', 'rhinoceros', 'hippopotamus'
 'savanah', 'africa', 'brooklyn', 'horse', 'pony', 'goldfish', 'swimming',
 'moana', 'tahiti', 'moorea', 'fishing', 'lagoon', 'boat', 'kitchen', 'ukulele',
 'surfboard', 'wetsuit', 'computer', 'telephone', 'guitar', 'piano', 'television',
 'subway', 'airplane', 'electricity', 'baby', 'bottle', 'apartment', 'building',
 'castle', 'princess', 'island', 'shark', 'dolphin', 'whale', 'balloon', 'leopard',
 'cheetah', 'zebra', 'mango', 'banana']

class Word:
	def __init__(self, name):
		self.name = name
		self.length = len(self.name)
		self.guessed = 0
	def changE(self):
		self.name = random.choice(wordlist)
		self.length = len(self.name)
		self.guessed = 0
	def iterate(self):
		self.guessed = self.guessed + 1

newword = Word('')
newword.changE()
nwlist = list(newword.name)
nw = list(newword.name)



i = 0
for i in range(0,len(nw)):
	nw[i] = '_'

#print(newword.name)
print('you have 10 chances to guess')
print(nw)
print(newword.length)
print(newword.guessed)

counter = 0

while counter < 10:
	guess = input('Which letter?')
	j = 0
	match = 0
	for j in range(0,newword.length):
		if nwlist[j] == guess:
			nw[j] = guess
			newword.iterate()
			match = match + 1
			print(nw)
		else: print(nw)
	print(counter)
	if match == 0:
		counter = counter + 1
	print('counter')
	print(counter)

	if newword.guessed == newword.length:
		print(newword.name)
		print('Congratulations, you won with')
		print(counter)
		print('misses')
		break

if counter == 10:
	print('You lost')
	print(newword.name)

