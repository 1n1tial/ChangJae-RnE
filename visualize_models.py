from .models import *
from .preprocess import train_loader
from torchviz import make_dot

yhat = model2(next(iter(train_loader))[0], 100)

make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
