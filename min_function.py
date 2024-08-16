import torch

if __name__ == "__main__":
    x = torch.zeros(2, requires_grad=True)
    f = lambda x : (x + torch.tensor([0, 0])).pow(2).sum()
    lr = 0.1

    for i in range(10):
        y = f(x) #get function
        y.backward() #calculate gradiant to x
        gradiant = x.grad
        x.data.add_(-lr*gradiant)
        x.grad.zero_() #reset the gradiant for the next loop
        print("Step {}: x[0]={}, x[1]={}".format(i,x[0],x[1]))