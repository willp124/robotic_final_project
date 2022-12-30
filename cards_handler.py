import random
debug = False
class Hands_divider:
	#cards is in the form ['S13', 'H5', 'C3', 'D9', ...]
	#S deotes Spade, H denotes Heart, D denote Diamond, C denotes Club
	def __init__(self, cards):
		if len(cards) != 13:
			raise Exception("There should be 13 cards.")
		self.suit_spade = [False for i in range(13)]
		self.suit_heart = [False for i in range(13)]
		self.suit_diamond = [False for i in range(13)]
		self.suit_club = [False for i in range(13)]
		self.hands = []
		self.four_of_a_kind = []
		self.three_of_a_kind = []
		self.pairs = []
		self.have_straight_flush = False
		self.suit_correspondence = {'S': self.suit_spade, 'H': self.suit_heart, 'D': self.suit_diamond, 'C': self.suit_club}
		self.valid_checker(cards)
		
	def valid_checker(self, cards):
		for card in cards:
			try:
				if int(card[1:]) <= 13 or int(card[1:]) >= 1:				
					value = int(card[1:]) - 1 
				else:
					raise Exception(f'invalid value on card "{card}"')
			except ValueError:
				raise Exception(f'invalid value on card "{card}"')

			try:
				suit = self.suit_correspondence[card[0]]
			except KeyError:
				raise Exception(f'invalid suit on card "{card}"')

			if suit[value]:
				raise Exception(f'duplicated card "{card}"')
			suit[value] = True

	def four_of_a_kind_handler(self):
		# Take four of a kind out
		for i in self.four_of_a_kind:
			for suit in self.suit_correspondence.values():
				suit[i-1] = False
	
	def last_hand_checker(self):
		# If there is no four-of-a-kind
		if len(self.hands) == 10:
			for key, suit in self.suit_correspondence.items():
				for i, card in enumerate(suit):
					if card:
						self.hands += [f'{key}{i+1}']
						suit[i] = False
			return True
		# If there is a four-of-a-kind and another hand
		elif len(self.hands) == 5 and len(self.four_of_a_kind) == 1:
			pair = []
			# If there is a straight-flush
			if self.have_straight_flush:
				# Take the pair out and break
				if len(self.pairs) == 1:
					for key, suit in self.suit_correspondence.items():
						if suit[self.pairs[0]-1]:
							pair += [f'{key}{self.pairs[0]}']
							suit[self.pairs[0]-1] = False
				# A straight-flush plus the four-of-a-kind
				self.hands += [f'S{self.four_of_a_kind[0]}', f'H{self.four_of_a_kind[0]}', f'D{self.four_of_a_kind[0]}', f'C{self.four_of_a_kind[0]}']
				# Plus the rest cards
				for key, suit in self.suit_correspondence.items():
					for i in range(13):
						if suit[i]:
							self.hands += [f'{key}{i+1}']
							suit[i] = False
				# Plus the pair
				self.hands += pair
			# If there is no straight-flush
			else:
				# Should arrange the four-of-a-kind to go first
				rest = []
				# Take the pair out
				if len(self.pairs) == 1:
					for key, suit in self.suit_correspondence.items():
						if suit[self.pairs[0]-1]:
							pair += [f'{key}{self.pairs[0]}']
							suit[self.pairs[0]-1] = False
				# Make the four-of-a-kind
				rest += [f'S{self.four_of_a_kind[0]}', f'H{self.four_of_a_kind[0]}', f'D{self.four_of_a_kind[0]}', f'C{self.four_of_a_kind[0]}']
				for key, suit in self.suit_correspondence.items():
					for i in range(13):
						if suit[i]:
							rest += [f'{key}{i+1}']
							suit[i] = False
				# Arrange the four-of-a-kind to go first
				self.hands = rest[:5] + self.hands + pair
				# If there are some cards in rest still
				if len(self.hands) != 13:
					self.hands += rest[5:]
			return True
		# If there are two four-of-a-kind
		elif len(self.four_of_a_kind) == 2:
			# Take out the most number of cards of a kind
			a_kind = []
			if len(self.three_of_a_kind) == 1:
				for key, suit in self.suit_correspondence.items():
					if suit[self.three_of_a_kind[0]-1] == True:
						a_kind += [f'{key}{self.three_of_a_kind[0]}']
						suit[self.three_of_a_kind[0]-1] = False
			elif len(self.pairs) != 0:
				for key, suit in self.suit_correspondence.items():
					if suit[self.pairs[0]-1] == True:
						a_kind += [f'{key}{self.pairs[0]}']
						suit[self.pairs[0]-1] = False
			# Find the rest cards
			rest = []
			for key, suit in self.suit_correspondence.items():
				for i in range(13):
					if suit[i]:
						rest += [f'{key}{i+1}']
						suit[i] = False
			self.hands += [f'S{self.four_of_a_kind[0]}', f'H{self.four_of_a_kind[0]}', f'D{self.four_of_a_kind[0]}', f'C{self.four_of_a_kind[0]}']
			self.hands += [rest[0]]
			self.hands += [f'S{self.four_of_a_kind[1]}', f'H{self.four_of_a_kind[1]}', f'D{self.four_of_a_kind[1]}', f'C{self.four_of_a_kind[1]}']
			self.hands += [rest[1]]
			self.hands += a_kind
			# If there are still cards in rest
			if len(self.hands) != 13:
				self.hands += rest[2:]
			return True
		# If there are three possible four-of-a-kind
		elif len(self.four_of_a_kind) == 3:
			self.hands += [f'S{self.four_of_a_kind[0]}', f'H{self.four_of_a_kind[0]}', f'D{self.four_of_a_kind[0]}', f'C{self.four_of_a_kind[0]}']
			# Find the last card
			for key, suit in self.suit_correspondence.items():
				for i in range(13):
					if suit[i]:
						self.hands += [f'{key}{i+1}']
						suit[i] = False
						break
			self.hands += [f'S{self.four_of_a_kind[1]}', f'H{self.four_of_a_kind[1]}', f'D{self.four_of_a_kind[1]}', f'C{self.four_of_a_kind[1]}']
			self.hands += [f'S{self.four_of_a_kind[2]}', f'H{self.four_of_a_kind[2]}', f'D{self.four_of_a_kind[2]}', f'C{self.four_of_a_kind[2]}']
			return True
		return False

	# def empty_suit_updator(self):
	# 	for suit in self.suit_correspondence.values():
	# 		empty = True
	# 		for card in suit:
	# 			if card:
	# 				empty = False
	# 				break
	# 		if empty:
	# 			suit = []

	def straight_flush_handler(self):
		hands = []
		def handle(key, hands):
			count = 0
			for i, card in enumerate(self.suit_correspondence[key]):
				if card == True:
					count += 1
					if count == 5:
						print(f'Straight-flush from {key}{i-3}')
						self.have_straight_flush = True
						for j in range(5):
							hands += [f'{key}{i-3+j}']
							self.suit_correspondence[key][i-j] = False
				else:
					count = 0

		for key in self.suit_correspondence:
			handle(key, hands)

		#handle two straight flushes, have the large one go first
		if len(hands) == 5:
			self.hands = hands
		elif len(hands) == 10:
			if self.card_comparator(hands[4], hands[9]) == 0:
				self.hands = hands
			else:
				self.hands = hands[5:] + hands[:5]

	def same_kind_finder(self):
		for i in range(12, -1, -1):
			count = 0
			for suit in self.suit_correspondence.values():
				if suit[i]:
					count += 1
			if count == 4:
				print("Four-of-a-kind at", i+1)
				self.four_of_a_kind += [i+1]
			if count == 3:
				print("Three-of-a-kind at", i+1)
				self.three_of_a_kind += [i+1]
			if count == 2:
				print("pair at", i+1)
				self.pairs += [i+1]
		return self.four_of_a_kind, self.three_of_a_kind, self.pairs

	def full_house_handler(self):
		# Find possible full-house
		number = min(len(self.three_of_a_kind), len(self.pairs))
		# Take out full-house
		for i in range(number):
			for key, suit in self.suit_correspondence.items():
				# Take three-of-a-kind from large side
				if suit[self.three_of_a_kind[i]-1]:
					self.hands += [f'{key}{self.three_of_a_kind[i]}']
					suit[self.three_of_a_kind[i]-1] = False
				# Take pairs from small side
				if suit[self.pairs[-(i+1)]-1]:
					self.hands += [f'{key}{self.pairs[-(i+1)]}']
					suit[self.pairs[-(i+1)]-1] = False

		# Delete those being taken out
		if len(self.three_of_a_kind) == number:
			self.three_of_a_kind = []
		else:
			self.three_of_a_kind = self.three_of_a_kind[number:]
		if len(self.pairs) == number:
			self.pairs = []
		else:
			self.pairs = self.pairs[:len(self.pairs)-number]
		

	def flush_handler(self):
		suit_checked = []
		# Find flush from large to small
		for i in range(12, -1, -1):
			for key, suit in self.suit_correspondence.items():
				# If the suit is not checked
				if suit[i] and key not in suit_checked:
					# Count the number of cards of the same suit
					count = 0
					for card in suit:
						if card:
							count += 1
					# If count is equal to 5
					if count == 5:
						for j in range(i, -1, -1):
							if suit[j]:
								self.hands += [f'{key}{j+1}']
								suit[j] = False
								if j+1 in self.pairs:
									self.pairs.remove(j+1)
								elif j+1 in self.three_of_a_kind:
									self.three_of_a_kind.remove(j+1)
					# If count is greater than 5
					elif count > 5:
						# Take out the largest one
						self.hands += [f'{key}{i+1}']
						suit[i] = False
						# First round: Skip the cards forming a pair/three-of-a-kind
						for j in range(i):
							if suit[j]:
								# Skip the cards forming a pair/three-of-a-kind
								if j+1 not in self.three_of_a_kind and j+1 not in self.pairs:
									self.hands += [f'{key}{j+1}']
									suit[j] = False
									# Already get five cards
									if len(self.hands) % 5 == 0:
										break
						# Second round: Skip the cards forming a three-of-a-kind
						if len(self.hands) % 5 != 0:
							for j in range(i):
								if suit[j]:
									# Skip the cards forming a three-of-a-kind
									if j+1 not in self.three_of_a_kind:
										self.hands += [f'{key}{j+1}']
										suit[j] = False
										self.pairs.remove(j+1)
										# Already get five cards
										if len(self.hands) % 5 == 0:
											break
						# Third round: Take card no matter what
						if len(self.hands) % 5 != 0:
							for j in range(i):
								if suit[j]:
									# Take the card out
									self.hands += [f'{key}{j+1}']
									suit[j] = False
									self.three_of_a_kind.remove(j+1)
									for k, m in enumerate(self.pairs):
										if m < j+1:
											self.pairs.insert(k, j+1)
											break
									# Already get five cards
									if len(self.hands) % 5 == 0:
										break
					suit_checked += [key]

	def straight_handler(self):
		# Remember to delete three of a kind
		record = [False for i in range(13)]
		for i in range(12, -1, -1):
			for key, suit in self.suit_correspondence.items():
				if suit[i]:
					record[i] = True
		count = 0
		for i in range(12, -1, -1):
			if record[i]:
				count += 1
			else:
				count = 0
			if count == 5:
				for j in range(5):
					for key, suit in self.suit_correspondence.items():
						if suit[i+j]:
							self.hands += [f'{key}{i+j+1}']
							suit[i+j] = False
							if i+j+1 in self.three_of_a_kind:
								self.three_of_a_kind.remove(i+j+1)
								for k, m in enumerate(self.pairs):
									if m < i+j+1:
										self.pairs.insert(k, i+j+1)
										break
								if self.pairs == []:
									self.pairs = [i+j+1]
							elif i+j+1 in self.pairs:
								self.pairs.remove(i+j+1)
							break
				count = 0

	def three_of_a_kind_handler(self):
		# Remember to delete three of a kind
		three_of_a_kind_count = 0
		for i, j in enumerate(self.three_of_a_kind):
			if i + len(self.four_of_a_kind) == 2:
				break
			three_of_a_kind_count += 1
			for key, suit in self.suit_correspondence.items():
				if suit[j-1]:
					self.hands += [f'{key}{j}']
					suit[j-1] = False
			# Count the rest two cards
			count = 0
			for k in range(13):
				for key, suit in self.suit_correspondence.items():
					if count == 2:
						break
					if suit[k] and k+1 not in self.three_of_a_kind:
						self.hands += [f'{key}{k+1}']
						suit[k] = False
						count += 1
			if count < 2:
				for k in range(13):
					for key, suit in self.suit_correspondence.items():
						if count == 2:
							break
						if suit[k]:
							self.hands += [f'{key}{k+1}']
							suit[k] = False
							count += 1
		self.three_of_a_kind = self.three_of_a_kind[three_of_a_kind_count:]

	# def pairs_handler(self):
	# 	# Remember to delete pairs
	# 	while len(self.four_of_a_kind) + len(self.hands) // 5 != 2:
	# 		if

	def high_card_handler(self):
		# Remember to delete three of a kind
		while len(self.four_of_a_kind) + len(self.hands) // 5 != 2:
			# If there is a pair
			if len(self.pairs) != 0:
				# Take the pair out
				for key, suit in self.suit_correspondence.items():
					if suit[self.pairs[0]-1]:
						self.hands += [f'{key}{self.pairs[0]}']
						suit[self.pairs[0]-1] = False
				self.pairs = self.pairs[1:]
				# If there is another pair
				if len(self.pairs) != 0:
					# Take the pair out
					for key, suit in self.suit_correspondence.items():
						if suit[self.pairs[-1]-1]:
							self.hands += [f'{key}{self.pairs[-1]}']
							suit[self.pairs[-1]-1] = False
					self.pairs = self.pairs[:-1]
				# Finish the hand
				if len(self.hands) % 5 != 0:
					count = 0
					for i in range(13):
						if len(self.hands) % 5 == 0:
							break
						for key, suit in self.suit_correspondence.items():
							if suit[i] and i+1 not in self.pairs:
								self.hands += [f'{key}{i+1}']
								suit[i] = False
								if len(self.hands) % 5 == 0:
									break
					for i in range(13):
						if len(self.hands) % 5 == 0:
							break
						for key, suit in self.suit_correspondence.items():
							if suit[i]:
								self.hands += [f'{key}{i+1}']
								suit[i] = False
								if len(self.hands) % 5 == 0:
									break
			else:	
				count = 0
				for i in range(12, -1, -1):
					if count == 1:
						break
					for key, suit in self.suit_correspondence.items():
						if suit[i]:
							self.hands += [f'{key}{i+1}']
							suit[i] = False
							count += 1
							break
				for i in range(13):
					if count == 5:
						break
					for key, suit in self.suit_correspondence.items():
						if suit[i]:
							self.hands += [f'{key}{i+1}']
							suit[i] = False
							count += 1
							if count == 5:
								break

	def card_comparator(self, card1, card2):
		if int(card1[1:]) > int(card2[1:]):
			return 0
		elif int(card1[1:]) < int(card2[1:]):
			return 1
		else:
			return 0 if card1[0] > card2[0] else 1

	def divide(self):
		self.straight_flush_handler()
		if self.last_hand_checker():
			self.hands.reverse()
			return self.hands
		
		self.same_kind_finder()
		self.four_of_a_kind_handler()
		if self.last_hand_checker():
			self.hands.reverse()
			return self.hands

		self.full_house_handler()
		if self.last_hand_checker():
			self.hands.reverse()
			return self.hands

		self.flush_handler()
		if self.last_hand_checker():
			self.hands.reverse()
			return self.hands
		
		if debug:
			print("p:", cards_divider.pairs)
			print("t:", cards_divider.three_of_a_kind)
		self.straight_handler()
		if debug:
			print("p:", cards_divider.pairs)
			print("t:", cards_divider.three_of_a_kind)
		if self.last_hand_checker():
			self.hands.reverse()
			return self.hands

		self.three_of_a_kind_handler()
		if self.last_hand_checker():
			self.hands.reverse()
			return self.hands
		'''
		These two can be combined
		self.two_pairs_handler()
		self.one_pair_handler()
		'''
		# self.pairs_handler()
		# if self.last_hand_checker():
		# 	self.hands.reverse()
		# 	return self.hands

		self.high_card_handler()
		if self.last_hand_checker():
			self.hands.reverse()
			return self.hands

		return "Not finished"

	def display_cards(self):
		for key, suit in self.suit_correspondence.items():
			for i, card in enumerate(suit):
				if card == True:
					print(f'{key}{i+1}', end=' ')
		print()

def random_samples():
	suits = ['S', 'H', 'D', 'C']
	values = [f'{i+1}' for i in range(13)]
	cards = []
	card = ''
	while len(cards) != 13:
		card = random.choice(suits) + random.choice(values)
		if card not in cards:
			cards += [card]
	return cards

if __name__ == '__main__':
	# Tested cases: one/two straight-flush, one straight-flush and one four-of-a-kind, two four-of-a-kind, one full-house and a four-of-a-kind, two full-house, one/two flush...
	# cards_divider = Hands_divider(['S13', 'H5', 'C3', 'D9', 'S2', 'H10', 'C8', 'D7', 'S1', 'S3', 'D5', 'H12', 'H11'])
	# cards_divider = Hands_divider(['H13', 'H5', 'C9', 'H9', 'S2', 'H12', 'C8', 'S5', 'S1', 'S3', 'S4', 'H10', 'H11'])
	# cards_divider = Hands_divider(['H13', 'H5', 'C9', 'H9', 'S2', 'C13', 'C8', 'S5', 'S1', 'S3', 'S4', 'D13', 'S13'])
	# cards_divider = Hands_divider(['H13', 'H5', 'H8', 'H9', 'H1', 'C13', 'C8', 'S5', 'S1', 'D1', 'C1', 'D13', 'S13'])
	# cards_divider = Hands_divider(['H13', 'H5', 'C5', 'H9', 'H1', 'C13', 'D5', 'S5', 'S1', 'D1', 'C1', 'D13', 'S13'])
	# cards_divider = Hands_divider(['H13', 'H5', 'H8', 'H9', 'H1', 'C9', 'C8', 'S5', 'S1', 'D1', 'C2', 'D13', 'S13'])
	# cards_divider = Hands_divider(['H13', 'H5', 'H8', 'H9', 'H1', 'C9', 'C8', 'C5', 'C1', 'C13', 'H2', 'H4', 'H7'])
	# cards_divider = Hands_divider(['S13', 'H5', 'C5', 'D9', 'S2', 'H10', 'C4', 'D7', 'S1', 'S3', 'D5', 'H12', 'H11'])
	# cards_divider = Hands_divider(['C13', 'H4', 'H6', 'H7', 'H1', 'C9', 'C11', 'S7', 'S1', 'D1', 'C2', 'D13', 'S13'])
	# cards_divider = Hands_divider(['S13', 'H7', 'C3', 'D13', 'S2', 'H2', 'C8', 'D8', 'S1', 'S3', 'D5', 'H1', 'H11'])
	cards_divider = Hands_divider(random_samples())
	cards_divider.display_cards()
	print(cards_divider.divide())
	cards_divider.display_cards()
	if debug:
		print("p:", cards_divider.pairs)
		print("t:", cards_divider.three_of_a_kind)
		print(cards_divider.hands)