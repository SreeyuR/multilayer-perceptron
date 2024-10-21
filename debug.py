import torch
from torch.autograd import Variable
import mlp

x=torch.tensor(7.0,requires_grad=True)

# Define the equation
y = (x**2)+3

# Differentiate using torch
#Uses backward function to compute the gradient value
y.backward()

# Print the derivative value
# of y i.e dy/dx = 2x  = 2 X 7.0 = 14.
print(y.grad)

print("MINE")

model = mlp.MLP([3, 2]) # reLU, softmax
model.train(x, y, epochs=1)
print(model.gradients)

