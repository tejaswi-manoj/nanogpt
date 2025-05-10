import torch

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

# Encode the entire dataset and put it in a tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split the validation and train data 
n = int(0.9*len(data)) # first 90% of the data is used for training, rest for val
train_data = data[:n] 
val_data = data[n:]

# block size: the maximum context length (length of each chunk of data used to train the model in a single forward/back pass)
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} the target is: {target}")

# batch size: the number of independent sequences that will be processed in parallel in a single forward/backward pass
torch.manual_seed(1337) # ensures that the random numbers generated do not change every time you run the script
batch_size = 4
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) #returns an array of batch_size number of valid random starting indices to get batch from
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")
    