# Un exemple de réseau avec pytorch ####

## A. Chargement des images à partir de la base CIFAR-100 ####

# importation des images
meubles <- keras::dataset_cifar100()

get_images <- function(categorie,ech="train"){
  
  # selection des images
  if(ech=="train"){
    images = meubles$train$x[meubles$train$y==categorie,,,]
  }
  else{
    images = meubles$test$x[meubles$test$y==categorie,,,]
  }
  
  #conversion des images en niveaux de gris
  grayscale = 0.298936*images[,,,1] + 0.587043*images[,,,2] + 0.114021*images[,,,3]
  grayscale = apply(grayscale,c(1,2,3),"as.integer")
  
  
  #affichage des images pour vérification
  par(mfcol=c(5,5))
  par(mar=c(0, 0, 4, 1), xaxs='i', yaxs='i')
  
  for(i in 1:25){
    image(1:32,1:32,grayscale[i,,],col = gray((0:255)/255))
  }
  
  #conversion en data.frame (1 ligne par image)
  grayscale <- data.frame(array_reshape(grayscale,dim=c(dim(grayscale)[1],32*32)))
  #grayscale <- apply(grayscale,2,"as.integer")
  
  grayscale["type"] <-  categorie
  
  return(grayscale)
}

# importation de l'échantillon d'entrainement
beds <- get_images("5",ech="train")
lits <- get_images("20",ech="train")
df_train <- rbind(beds,lits)

# importation de l'échantillon de test
beds <- get_images("5",ech="test")
lits <- get_images("20",ech="test")
df_test <- rbind(beds,lits)

# fonction de normalisation des données
normalise <- function(x){
  x = as.numeric(x)
  x = (x-0)/255
}

df_train[,1:(ncol(df_train)-1)] = apply(df_train[,1:(ncol(df_train)-1)],c(1,2),"normalise")
df_test[,1:(ncol(df_test)-1)] = apply(df_test[,1:(ncol(df_test)-1)],c(1,2),"normalise")


# ## B.a Lancement d'un réseau de neurones avec pytorch
# 
# Le package `torch` doit être installé au préalable sous R. Mieux vaut l'installer par ligne de commande pour éviter des conflits au niveau du Notebook.
# Les commandes sont :
# * `sudo -i R` (lance R)
# * `install.packages("torch")`
# 
# Le package peut ensuite être chargé de façon usuelle (cf. ci-dessous). Un nouveau lancement du notebook peut être nécessaire.
library("torch")

# L'estimation est réalisée sur la totalité de l'échantillon, sans batchs. Sinon, il est nécessaire d'utiliser le `dataloader` (cf. ci-dessous). Le chargement est un peu plus complexe. 
# 2. Convert our input data to matrices and labels to vectors.
x_train = as.matrix(apply(df_train[, 1:ncol(df_train)-1],2,"as.numeric"))
y_train = ifelse(df_train$type=="20",2,1)
x_test = as.matrix(apply(df_test[, 1:ncol(df_test)-1],2,"as.numeric"))
y_test = ifelse(df_test$type=="20",2,1)

# 3. Convert our input data and labels into tensors.
x_train = torch_tensor(x_train, dtype = torch_float())
y_train = torch_tensor(y_train, dtype = torch_long())
x_test = torch_tensor(x_test, dtype = torch_float())
y_test = torch_tensor(y_test, dtype = torch_long())

#### Définition du modèle
# cette ligne sert à vérifier si l'usage de GPU est possible
# dans cet exemple, cela n'est pas le cas.
cuda_is_available()

net = nn_module(
  "class_net",
  
  initialize = function(){
    
    # première couche de neurones : 1024 valeurs en entrée et 20 en sortie 
    self$linear1 = nn_linear(1024,150)
    # deuxième couche de neurones : 20 valeurs en entrée et 2 sorties
    self$linear2 = nn_linear(150,2)
    
  },
  
  forward = function(x){
    
    x %>%
      # 1ere couche
      self$linear1() %>%
      # les 25 neurones ont une fonction relu
      nnf_relu() %>%
      # 2eme couche
      self$linear2() %>%
      # fonction d'activation softmax pour transformer les sorties en probas
      nnf_softmax(2)
    
  }
  
)

model = net()


# Définition de la fonction de coût et de l'algorithme à utiliser
criterion = nn_cross_entropy_loss()  
optimizer = optim_adam(model$parameters)


epochs = 200

# Train the net
for(i in 1:epochs){
  
  # mise à zéro du gradient
  optimizer$zero_grad()
  
  y_pred = model(x_train)
  loss = criterion(y_pred, y_train)
  
  # retropropagation  
  loss$backward()
  
  # mise à jour des poids
  optimizer$step()
  
  
  # Check Training
  if(i %% 10 == 0){
    
    winners = y_pred$argmax(dim=2)
    corrects = (winners == y_train)
    accuracy = corrects$sum()$item() / y_train$size()
    
    cat(" Epoch:", i,"Loss: ", loss$item()," Accuracy:",accuracy,"\n")
  }
  
}

y_pred = model(x_test)
winners = y_pred$argmax(dim=2)
corrects = (winners == y_test)
accuracy = corrects$sum()$item() / y_test$size()
print(accuracy)


y_pred = model(x_train)
winners = y_pred$argmax(dim=2)
corrects = (winners == y_train)
accuracy = corrects$sum()$item() / y_train$size()
print(accuracy)

library(torch)
library(dplyr)

# Le chargement des données doit être effectué via une fonction, définie ci-dessous :
images_dataset <- dataset(
  name = "images_dataset",
  
  initialize = function(indices) {
    # on applique la fonction prepare_images_data de mise en forme des données (definie ci-aprés)
    # elle retourne une liste avec 2 elements
    data <- self$prepare_images_data(df_train[indices, ])
    # variables explicatives numériques
    self$xnum <- data[[1]]
    # variable à expliquer
    self$y <- data[[2]]
  },
  
  .getitem = function(i) {
    xnum <- self$xnum[i, ]
    y <- self$y[i, ]
    
    list(x = list(xnum), y = y)
  },
  
  # la dimension de la sortie est 1 : on predit cette fois une variable binaire (= une proba)
  .length = function() {
    dim(self$y)[1]
  },
  
  prepare_images_data = function(input) {
    
    input <- input %>%
      mutate() 
    
    # mise en forme de la variable a predire
    target_col <- ifelse(input$type=="20",1,0) %>%
      as.numeric() %>%
      as.matrix()
    
    # mise en forme des variables explicatives
    numerical_cols <- input %>%
      select(-type) %>%
      as.matrix()
    
    
    list(torch_tensor(numerical_cols,dtype=torch_float()),
         torch_tensor(target_col,dtype=torch_float()))
  }
)

#### Calcul des batches de données
# 
# On choisit une taille de 32 (batch_size) car c'est le paramètre par défaut sous `Keras`. On mélange les données (`shuffle=TRUE`).
# Lors de cette étape, l'écahntillon d'entraînement est lui-même séparé en `train`et `test`. C'est utile pour le choix des hyper-paramètres (nombre de neurones de la couche cachée).

train_indices <- sample(1:nrow(df_train), size = floor(0.8 * nrow(df_train)))
valid_indices <- setdiff(1:nrow(df_train), train_indices)

train_ds <- images_dataset(train_indices)
train_dl <- train_ds %>% dataloader(batch_size = 32, shuffle = TRUE)

valid_ds <- images_dataset(valid_indices)
valid_dl <- valid_ds %>% dataloader(batch_size = 32, shuffle = FALSE)

#### Définition du modèle avec une seule sortie 


net = nn_module(
  "class_net",
  
  initialize = function(cc){
    
    # première couche de neurones : 1024 valeurs en entrée et 20 en sortie 
    self$linear1 = nn_linear(1024,cc)
    # deuxième couche de neurones : 20 valeurs en entrée et 2 sorties
    self$linear2 = nn_linear(cc,1)
    
  },
  
  forward = function(x){
    
    x %>%
      # 1ere couche
      self$linear1() %>%
      # les 25 neurones ont une fonction relu
      nnf_relu() %>%
      # 2eme couche
      self$linear2() %>%
      # fonction d'activation softmax pour transformer les sorties en probas
      nnf_sigmoid()
    
  }
  
)

model = net(20)


#### Estimation du modèle

# On retrouve les étapes de la partie A. mais sur des batchs de données (il y a une boucle sur l'élément `train_dl`.
# On utilise l'entropie croisée binaire car on a une probabilité en sortie.



optimizer <- optim_adam(model$parameters)

for (epoch in 1:20) {
 
 model$train()
 train_losses <- c()  
 
 coro::loop(for (b in train_dl) {
   optimizer$zero_grad()
   y_pred <- model(b$x[[1]])
   loss <- nnf_binary_cross_entropy(y_pred, b$y)
   loss$backward()
   optimizer$step()
   train_losses <- c(train_losses, loss$item())
 })
 
 model$eval()
 valid_losses <- c()
 
 coro::loop(for (b in valid_dl) {
   output <- model(b$x[[1]])
   loss <- nnf_binary_cross_entropy(output, b$y)
   valid_losses <- c(valid_losses, loss$item())
 })
 
 cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
}

#### Evaluation de la performance sur l'échantillon `test`initial

# 2. Convert our input data to matrices and labels to vectors.
x_train = as.matrix(apply(df_train[, 1:ncol(df_train)-1],2,"as.numeric"))
y_train = ifelse(df_train$type=="20",2,1)
x_test = as.matrix(apply(df_test[, 1:ncol(df_test)-1],2,"as.numeric"))
y_test = ifelse(df_test$type=="20",2,1)

# 3. Convert our input data and labels into tensors.
x_train = torch_tensor(x_train, dtype = torch_float())
y_train = torch_tensor(y_train, dtype = torch_long())
x_test = torch_tensor(x_test, dtype = torch_float())
y_test = torch_tensor(y_test, dtype = torch_long())



y_pred = model(x_test)
winners = ifelse(y_pred>0.5,1,0)
corrects = (winners == (y_test-1))
accuracy = corrects$sum()$item() / y_test$size()
print(accuracy)

y_pred = model(x_train)
winners = ifelse(y_pred>0.5,1,0)
corrects = (winners == (y_train-1))
accuracy = corrects$sum()$item() / y_train$size()
print(accuracy)

# ## C. Nombre optimal de neurones dans la couche cachée
# 
# Comme avec la librairie `Keras`, il est possible de rechercher le nombre de neurones de la couche cachée sur une grille de valeurs.

optimizer <- optim_adam(model$parameters)
acc_train <- c()
acc_test <- c()

for (cc in seq(10,200,10)){
  
  print(cc)
  write.csv(cc,"test.csv")
  # initialisation du modele
  model = net(cc)
  optimizer <- optim_adam(model$parameters)
  
  # estimation sur des batchs sur le modèle
  for (epoch in 1:100) {
    
    
    model$train()
    train_losses <- c()  
    
    coro::loop(for (b in train_dl) {
      optimizer$zero_grad()
      y_pred <- model(b$x[[1]])
      loss <- nnf_binary_cross_entropy(y_pred, b$y)
      loss$backward()
      optimizer$step()
      train_losses <- c(train_losses, loss$item())
    })
    
    model$eval()
    valid_losses <- c()
    
    coro::loop(for (b in valid_dl) {
      output <- model(b$x[[1]])
      loss <- nnf_binary_cross_entropy(output, b$y)
      valid_losses <- c(valid_losses, loss$item())
    })
    
    #cat(sprintf("Loss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
    
    
  }
  
  
  
  # evaluation de la performance
  y_pred = model(x_train)
  winners = ifelse(y_pred>0.5,1,0)
  corrects = (winners == (y_train-1))
  accuracy = corrects$sum()$item() / y_train$size()
  write(accuracy,"train_pytorch.txt",append=TRUE)
  acc_train <- c(acc_train,accuracy)
  
  y_pred = model(x_test)
  winners = ifelse(y_pred>0.5,1,0)
  corrects = (winners == (y_test-1))
  accuracy = corrects$sum()$item() / y_test$size()
  write(accuracy,"test_pytorch.txt",append=TRUE)
  acc_test <- c(acc_test,accuracy)
  
}


# Pour un nouvel export des précisions :
write.csv(acc_test,"test_pytorch.txt",row.names=FALSE)
write.csv(acc_train,"train_pytorch.txt",row.names=FALSE)

# 
# #### Tracé des résultats sur les échantillons test et d'entraînement
# 
# (les fichiers des résultats obtenus avec *Keras* doivent figurer dans */home/jovyan*)
# 


# import des résultats avec Keras
keras_test <- read.csv("test_keras_s.txt",sep=",",header=FALSE)
colnames(keras_test) <- c("id","keras_test")
keras_train <- read.csv("train_keras_s.txt",sep=",",header=FALSE)
colnames(keras_train) <- c("id","keras_train")


# import des résultats avec pytorch
pytorch_test <- read.csv("test_pytorch.txt",sep=",",header=TRUE)
colnames(pytorch_test) <- c("pytorch_test")
pytorch_train <- read.csv("train_pytorch.txt",sep=",",header=TRUE)
colnames(pytorch_train) <- c("pytorch_train")

graph <- data.frame(cbind(c(rep("kerastest",20),rep("kerastrain",20),rep("pytorchtest",20),rep("pytorchtrain",20)),c(seq(10,200,10),seq(10,200,10),seq(10,200,10),seq(10,200,10)),
                          c(keras_test$keras_test,keras_train$keras_train,pytorch_test$pytorch_test,pytorch_train$pytorch_train)))
colnames(graph) <- c("source","x","y")
graph[,2:ncol(graph)] <- apply(graph[,2:ncol(graph)],2,"as.numeric")


library("tikzDevice")
library(tidyverse)
library(ggplot2)
theme_set(theme_classic())


# Combine into single data frame and add interpolation column
#ggplot(graph, aes(x, y)) +
#  geom_point(aes(colour = source)) + geom_smooth(method = "lm", formula = y ~ poly(x, 2),se = FALSE)
#tikz("iterations_chaises.tex",width=5,height=5)

ggplot(graph,aes(x=x,y=y,colour=factor(source)))+
  geom_point(data=subset(graph, source == "kerastest")) + geom_point(data=subset(graph, source=="kerastrain")) +
  geom_point(data=subset(graph, source == "pytorchtest")) + geom_point(data=subset(graph, source=="pytorchtrain")) +
  geom_smooth(data=subset(graph, source=="kerastest"), method='loess',formula=y~x,se=F) +
  geom_smooth(data=subset(graph, source=="kerastrain"), method='loess',formula=y~x,se=F) +
  geom_smooth(data=subset(graph, source=="pytorchtest"), method='loess',formula=y~x,se=F) +
  geom_smooth(data=subset(graph, source=="pytorchtrain"), method='loess',formula=y~x,se=F) 

