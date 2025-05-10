# DATASET
# We need a dataset to train on. For this project, we use the tine shakespeare dataset.

with open('input.txt', 'r', encoding='utf-8')
text = f.read()
print("Length of dataset in characters: ", len(text))

# Next, we get all the unique characters that appear in the text in the form of a sorted list.
chars = sorted(list(set(text))) # note: set removes all the duplicates

# Create a mapping from characters to integers and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s] # encoder: take a string and output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers and output a string

