import os
word = input('Word to guess   ')

#word = 'blaireau'

word = list(word)


wip = list(word)

length = len(wip)

i = 0
for i in range(0,length):
	wip[i] = '_'

#print(word)

print (wip)

counter = 10
ltrnbr = length

while counter > 0:

	letter = input('Which letter   ?')
#	letter = 'o'

	miss = 0

	i = 0
	for i in range(0,length):
		if word[i] == letter:
			wip[i] = letter
			ltrnbr = ltrnbr -1
		else: miss = miss + 1

	if miss > 0:
		counter = counter - 1


	print(wip)

	print('Remaining Attempts')
	print(counter)

	print(ltrnbr)

	if ltrnbr == 0:
		print('Congratulations, you won in')
		print(10 - counter)
		print('attempts')
		break
	else: continue

if counter == 0:
    print('You lost')