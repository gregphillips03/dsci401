#William (Greg) Phillips
#10/16/2017
#DSCI 401
#assignment 1

#flatten function
#take a list of irregular lists and return them as a single list
def flatten(input):
	import collections;
	#check to see if the object is already part of our collection 
	#using the isinstance builtin
	if isinstance(input, collections.Iterable):
		return [j for i in input for j in flatten(i)];
	else:
		return [input];
		
#powerset function
#take a list and return all the possible subsets (powerset) for a set. 
def powerset(input):
	result = [[]];
	for item in input:
		#extend appends to the list at the end
		result.extend([j + [item] for j in result]);
	return result;
	
#all permutations function
#take a list and return the permutations of that list
#we could just import 'from itertools import permutations' but that'd be too easy
def all_perms(input):
	#escape for nothing
	if len(input) == 0:
		return[]; 
	#escape for one element
	if len(input) == 1:
		return [input]; 
	l = []; 
	for i in range(len(input)):
		item = input[i]; 
		restoflist = input[:i] + input[i+1:]; 
		for j in all_perms(restoflist):
			l.append([item] + j);
	return l;

def spiral(n, end_corner):
	f = matrix([[x+y*n for x in range(n)] for y in range(n)]);
	f.ec = end_corner; 
	f.rotate_me(); 
	f.print_matrix();

class matrix:
	#constructor
	def __init__(this, ary):
		this.size = len(ary);
		this.array = [[None for y in range(this.size)] for y in range(this.size)];
		this.xpos = 0; 
		this.ypos = 0;
		this.availDirections = [this.down, this.right, this.up, this.left];
		this.direction = 0;
		this.fill(ary);
		this.ec = 1; 
	#fill up the array we created
	def fill(this, ary):
		for row in reversed(ary):
			for elem in reversed(row):
				this.array[this.xpos][this.ypos] = elem;
				this.go_to_next();
	#search for another position
	def find_next_pos(this):
		np = this.get_next_pos();
		if np[1] in range(this.size) and np[0] in range(this.size):
			return this.array[np[0]][np[1]] == None;
		return False;
	#go to another position
	def go_to_next(this):
		i = 0;
		while not this.find_next_pos() and i < 4:
			this.direction = (this.direction + 1) % 4;
			i += 4;
		this.xpos, this.ypos = this.get_next_pos();
	#rotate the matrix based off where we want the end corner to fall
	#yeah I know it's ugly, but it works!
	def rotate_me(this):
		if(this.ec == 2):
			this.array = zip(*this.array[::-1]); 
		if(this.ec == 3):
			this.array = zip(*this.array[::-1]);
			this.array = zip(*this.array[::-1]); 
			this.array = zip(*this.array[::-1]); 			
		if(this.ec == 4):
			this.array = zip(*this.array[::-1]); 
			this.array = zip(*this.array[::-1]); 
	#where should the next position actually be?
	#returns the helper function and the actual x and y positions
	def get_next_pos(this):
		return this.availDirections[this.direction](this.xpos, this.ypos);
	#helper to move down
	def down(this, x, y):
		return x + 1, y;
	#helper to move right
	def right(this, x, y):
		return x, y + 1;
	#helper to move up
	def up(this, x, y):
		return x - 1, y
	#helpter to move left
	def left(this, x, y):
		return x, y - 1;
	def print_matrix(this):
		for row in this.array:
			print(row);