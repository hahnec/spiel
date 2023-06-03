from spiel_net.model_spiel import SpielNet
from torchinfo import summary

net = SpielNet(device='cpu')
summary(net)

# INxOUT+BIAS, e.g. 8x32+32=288 or 4x1+1=5
# see also https://stackoverflow.com/questions/56745343/fully-connected-layer-dimensions