# 3 Hyper parameters
lr = 0.04       | Learning rate
batch_size = 10 | Data batch size
N_epochs = 160  | number of training cycles


## Learning rate
Smaller is not always better. Ex: From .2 to .001 is too low, but .1 is better

You could also change the depth, width.

## ADAM 
    - More prone to overfitting but trains way faster 
SGD-M with momentum
- Adds to gradient descent by W <- W - lr*grad **+ momentum * (W - W_prev)**

Rescaled Gradients
- Typically RMSprop to scale the grad to be smaller as you go in the above equation

## Learning Rate Scheduler
Move to a smaller learning rate as time goes on
- torch.optim.lr_scheduler
```py
scheduler = ExponentialLR(optimizer, gamma=0.9)
# Ex:
for epoch in range(num):
    for input, target in dataset:
    ...
    scheduler.step()
```

## Comparison of Activation Functions
Sigmoid, tanh, LeakyReLU

## Loss functions
Ex: nn.MSELoss | Creates criterion measures mean square error (squared L2 norm) between each element in input x and target y
- nn.CrossEntropyLoss | Computes entropy loss between input logits and target

## Multiclass Classification Network
Output of nn should be score
- score(logit) of each class. Higher score, more likely class is true output

NN -> Score -> SoftMax() -> Prob -> argmax(Probabilities) -> Predicted Output/Label
- CrossEntropy is the standard loss for classification problems
- Last layer of NN should be linear, not SoftMax
    - Using SoftMax at the end of the NN basically makes SoftMax used twice which will run but will be wrong




