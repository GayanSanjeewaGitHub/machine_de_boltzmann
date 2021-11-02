
# MACHINE DE BOLTZMANN



# PART 1 - Preprocessing..


# Librairies
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# Importation du jeu de données
movies = pd.read_csv("ml-1m/movies.dat",
                     sep="::",
                     header=None,
                     engine="python",
                     encoding="latin-1")
users = pd.read_csv("ml-1m/users.dat",
                     sep="::",
                     header=None,
                     engine="python",
                     encoding="latin-1")
ratings = pd.read_csv("ml-1m/ratings.dat",
                     sep="::",
                     header=None,
                     engine="python",
                     encoding="latin-1")


# Préparation du jeu d'entrainement et du jeu de test
training_set = pd.read_csv("ml-100k/u1.base", delimiter="\t", header=None)   #(delimiter="\t")=tabulation  (header=None)=pas d'entête
training_set = np.array(training_set, dtype="int")
test_set = pd.read_csv("ml-100k/u1.test", delimiter="\t", header=None)   
test_set = np.array(test_set, dtype="int")


# Obtenir le nombre d'utilisateurs et le nombre de films
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Conversion des données en matrice/liste de liste
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[data[:, 0] == id_users, 1]
        id_ratings = data[data[:, 0] == id_users, 2]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)  
test_set = convert(test_set) 


# Conversion des données en tenseurs/matrice à 2 dimensions = format pour pytorch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Conversion des notes en 1 (=aime) et 0 (=n'aime pas)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

"""

# PART 2 - Construction


# Créer l'architecture du réseau de neurones

class RBM():
    
    #fonction d'initialisation
    def __init__(self, nv, nh):   #nv=neurones visibles  nh=neurones cachés
        self.W = torch.randn(nh, nv)   #w=poids
        self.a = torch.randn(1, nh)   #biais pour les neurones cachés
        self.b = torch.randn(1, nv)   #biais pour les neurones visibles
        
    #fonction d'échantillonnage de Gibs , calcule des neurones caché en fonction des neurones visibles
    def sample_h(self, x):   #x=neurones visibles à partir desquel on calcule les neurones cachés 
        wx = torch.mm(x, self.W.t())   #somme les poids multiplié par les neurornes
        activation = wx + self.a.expand_as(wx)   #ajoute le biais
        p_h_given_v = torch.sigmoid(activation)   #fonction d'activation = probabilité du neurone caché en fonction du neurone visible
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    #fonction inverse , calcule des neurones visibles en fonction des neurones cachés
    def sample_v(self, y):   #y=neurones cachés à partir desquel on calcule les neurones visibles
        wy = torch.mm(y, self.W)   #somme les poids multiplié par les neurornes
        activation = wy + self.b.expand_as(wy)   #ajoute le biais
        p_v_given_h = torch.sigmoid(activation)   #fonction d'activation = probabilité du neurone visible en fonction du neurone caché
        return p_v_given_h, torch.bernoulli(p_v_given_h)    
        
    #fonction qui entraine le modèle
    def train(self, v0, vk, ph0, phk):   #v0=valeur des node visibles à l'état initiale, vk=valeur des node visibles après k iterration d'échantillonage
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)   #ph0=proba du node caché à v0, phk=proba du node caché à vk
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Paramétrage             
nv = len(training_set[0])
nh = 100
batch_size = 100

#Création de la RBM
rbm = RBM(nv, nh)



# PART 3 - Entrainement


nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0   #variable de cout qui est incrémenté à chaque batch
    s = 0.   #compteur en float
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):   #échantillonnage de Gibs
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))

"""

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
