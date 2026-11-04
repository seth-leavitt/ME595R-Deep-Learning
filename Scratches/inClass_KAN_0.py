import kan
import torch
import matplotlib.pyplot as plt


SEED = 42
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
dataset = kan.create_dataset(f, n_var=2)

model = kan.KAN(width=[2,5,1], grid=3, k=3, seed=SEED)
model(dataset['train_input'])
model.plot()
plt.show()

# model.train does not work contrary to the docs, the actual function is model.fit
model.fit(dataset, opt="LBFGS", steps=50)
model.plot()
plt.show()

model = model.prune()
model.plot()
plt.show()

model.fit(dataset, opt="LBFGS", steps=50)
model.plot()
plt.show()

model.auto_symbolic(weight_simple=0.1)

model.fit(dataset, opt="LBFGS", steps=50)

from kan.utils import ex_round
print(ex_round(model.symbolic_formula()[0][0],4)) # 0th index is the formulas, the 1st index is the


