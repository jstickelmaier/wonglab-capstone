# PyTorch

## Tensors

### Construction
 - `x=torch.empty()`
 - `x=torch.ones(2,3,4)`
 - `x=torch.rand(2,2)`
 - `x=torch.ones(2,2, dtype=torch.float32)`
 
 ### Methods
  - `.size`
  - `torch.add(x,y)`
  - `.add_` - in place (all trailing underscores are inplace)
  - `torch.sub`
  - `torch.mul`
  - `torch.div`
  - Typical slicing i.e. `x[:,3]`: 4th column of x
  - `x[1,1].item()`
  - `y=x.view(16)`... will only work if x has 16 vales...

  ### Autograd
  - `requires_grad` attribute will track operations in backpropogation gradient function
  - Values will be summed up every time `.backward()` is called, be sure to empyty `.grad.zero_()` after each training step... also need to do after using optimizers i.e. `optimizer.zero_grad()`

  ### Backpropagation
  ... TBD

## Dataset and DataLoader
Divide total set into batches, do opt only on batches, calculations and iterations handled by Dataset and DataLoader...

**epoch**: one forward and backward pass of ALL training smaples
`batch_size` = number of training samples in forward and backward pass
**num iterations**: number of passes, each using `batch_size` number of samples


Ex. 
```
  import torch
  import torchvision
  from torch.utils.data import Dataset, DataLoader
  import numpy as np
  import math

  class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter = ',', dtype=np.float32, skiprows =1) #load dataset, skips header row
        self.x = torch.from_nump(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    dataiter = iter(dataloader)
    data = dataiter.next()

    #display first batch!
    features, labels = data
    print(features, labels)

    #training loop

    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/4)

    for epoch in range(num_epoch):
        for i, (inputs, labels) in enumerate(dataloader):
            #forward, backward, update
            if (i+1) % 5 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}') #summary for debugging


```

