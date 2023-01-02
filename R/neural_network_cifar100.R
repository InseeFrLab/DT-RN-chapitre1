# ## Un exemple "bac à sable" : la reconnaissance d'images de meubles (lits et chaises)
# 
# Ce notebook se décompose de la façon suivante :
#   
#   [A. mise en forme des données de la base CIFAR-100](#donnees)
#     
#     Différentes méthodes de classification sont ensuite testées sur la base d'images de chaises et de lits :
# * Un réseau de neurones simple avec la librairie *neuralnet* [voir ci-dessous](#neuralnet)
# * Une méthode de *machine learning* : une régression pénalisée de type Lasso [voir](#lasso)
# * Un réseau de neurones avec la librairie *Keras* [voir](#keras)
# 
# Importation des librairies nécessaires
# 
# Une image docker contenant toutes les librairies nécessaires est disponible [ici](https://git.lab.sspcloud.fr/g6ginq/tests_docker)
# 
# Sinon, la librairie *magick* doit au préalable avoir été installé sous *conda* par *conda install r-magick*. Il peut aussi être nécessaire d'installer *tensorflow* et *Keras* par *pip install tensorflow*...
#     Le fichier *librairies_R.R* installe toutes les librairies utiles à ce notebook. Son exécution peut être assez longue.

# ### <a name="donnees">A. Importation et mise en forme des tables de données</a>
# 
# Importation des images de la base CIFAR-10 à partir de la librairie Keras


# importation des images
meubles <- dataset_cifar100()


# On retient les chaises et les lits. La fonction get_images :
#   * permet de sélectionner les images dans la base cifar-10 à l'aide du numéro de catégorie. 
# * Elle renvoie une table avec une ligne par image convetie en niveaux de gris.
# * Elle affiche les images pour vérification
# * echantillon doit être "train" ou "test"


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

# Vérification des tables
head(df_train,2)


normalise <- function(x){
  x = as.numeric(x)
  x = (x-0)/255
}


df_train[,1:(ncol(df_train)-1)] = apply(df_train[,1:(ncol(df_train)-1)],c(1,2),"normalise")
df_test[,1:(ncol(df_test)-1)] = apply(df_test[,1:(ncol(df_test)-1)],c(1,2),"normalise")


head(df_train[,c(1:10,ncol(df_train))],2)


#png("lits.png")
par(mfrow=c(2,3))
par(mar=c(0, 0, 0, 0), xaxs='i', yaxs='i')

atracer = df_train[df_train[,"type"]=="5",]

for (i in 1:6) { 
  img <- array_reshape(as.matrix(atracer[i,1:(ncol(atracer)-1)]),dim=c(32,32))
  image(1:32,1:32,img, col = gray((0:255)/255))
}
#dev.off()


# ### <A name="classifications"> B. Catégorisation des images par différentes méthodes</a>
# 
# -------
#   #### <a name="neuralnet"> 1. Un premier réseau de neurones avec la librairie *neuralnet* sous R </a>
#   -------
#   
#   On teste la librarie `neuralnet` sous R qui permet de réaliser des réseaux de neurones simples (des multilayers perceptrons). 
# 

library(neuralnet)

# Estimation d'un premier réseau sur la totalité de l'échantillon d'entrainement :
# 
# * type==20 ~ : on cherche à prédire le type 20 (les chaises) par les autres variables (intensité des pixels)
# * data : table où trouver les données
# * hidden : nombre de neurones de la couche cachée (il est possible de spécifier plusieurs couches cachées (voir la doc)).
# * threshold : critère d'arrêt 
# * act.fct : fonction d'activation à appliquer
# * linear.output : si False, la fonction d'activation est appliquée à la sortie. 
# * likelihood : les critères AIC et BIC sont affichés si l'entropie croisée est utilisée. 
# * err.fct : sse pour mse et ce pour l'entropie croisée
# * lifesign : un paramètre pour obliger le programme à décrire ce qu'il fait dans la log

# Binary classification
nn=neuralnet(type=="20" ~ . ,data=df_train, hidden=20,threshold = 0.0001,err.fct="ce",act.fct="logistic",
             linear.output = FALSE,likelihood = TRUE,lifesign="full")


## Prediction using neural network
Predict=predict(nn,df_test[,1:ncol(df_test)-1])
predict <- ifelse(Predict>0.5,1,0)
initial <- ifelse(df_test[,"type"]=="20",1,0)
table(initial,predict)

sum(as.integer(initial == predict))/length(df_test[,"type"])

# On cherche le nombre de neurones à utiliser dans la couche cachée. Pour cela, on va subdiviser l'échantillon d'entraînement afin de pouvoir tester les modèles évalués sur une grille de valeurs. Une évaluation par validation croisée aurait été préférable mais trop gourmande en temps de calcul.

# fonction de calcul de la précision
get_accuracy <- function(table){
  
  Predict=predict(nn,table[,1:ncol(table)-1])
  predict <- ifelse(Predict>0.5,1,0)
  initial <- ifelse(table[,"type"]=="20",1,0)
  return(sum(as.integer(initial == predict))/length(table[,"type"]))
  
}


# on melange les observations de la table initiale
rows <- sample(nrow(df_train))
df_train <- df_train[rows,]

# On separe la table en deux pour tester les valeurs des hyper-paramètres :
nbobs <- as.integer(nrow(df_train)*0.8)
df_train_e <- df_train[1:nbobs,]
df_train_t <- df_train[nbobs:nrow(df_train),]

# affichage des dimensions des deux tables
print("Taille de l'échantillon d'entrainement :")
print(dim(df_train_e))
print("Taille de l'échantillon test :")
print(dim(df_train_t))

accuracy_train = c()
accuracy_test = c()



for(k in seq(5,100,5)){
  
  print(k)
  
  # évaluation du réseau de neurones
  nn=neuralnet(type==20 ~ . ,data=df_train_e, hidden=k,threshold = 0.001,act.fct = "logistic",
               linear.output = FALSE,likelihood = FALSE,lifesign="full")
  
  # calcul de la précision sur l'échantillon test
  print(get_accuracy(df_train_e))
  
  accuracy_train <- c(accuracy_train,get_accuracy(df_train_e))
  accuracy_test <- c(accuracy_test,get_accuracy(df_train_t))
}


library(ggplot2)

a_list = list(accuracy_train, accuracy_test)
a_df = do.call("rbind", lapply(a_list, 
                               function(x) data.frame(value = x, 
                                                      count = seq(5,100,5))))
ID_options = c("train","test")
a_df$ID = rep(ID_options, sapply(a_list, length))

#png("accuracy_neuralnet.png") 
tikz("accuracy_neuralnet.tex",standAlone=TRUE,width=5,height=5)

ggplot(a_df, aes(x = count, y = value, color = ID)) + geom_point()

dev.off()

# Les performances du modèles sont réévaluées sur l'échantillon de test :
accuracy_test


# -----
#   #### <a name="lasso">2. Une régression pénalisée (Lasso)</a>
#   (ou comment résoudre la problème avec une technique de machine learning ?)
# 
# -----
#   
#   Une méthode de ML pour trier les images. On utilise pour cela la librairie `glmnet`.

library(tidyverse)
library(caret)
library(glmnet)

# mise en forme des variables explicatives
x <- model.matrix(type~., data.frame(df_train))[,-1]
# mise en forme de la variable à expliquer
y <- ifelse(df_train[,"type"] == "20", 1, 0)


# On estime une régression pénalisée avec les paramètres :
#   * family = "binomial" : la variable à expliquer est binaire
# 
# La fonction `cv.glmnet` permet de choisir le paramètre $\lambda$ par validation croisée. Le nombre de plis par défaut est égal à 10.
# 


# Build the model using the training set
reg <- cv.glmnet(x, y, family = "binomial",lambda = NULL,alpha=0.3)

# affichage du paramètre choisi
reg$lambda.min

# tracé de la fonction de coût
plot(reg)

# Calcul de la précision sur l'échantillon de test 
x.test <- model.matrix(type ~., data.frame(df_test))[,-1]
probabilities <- reg %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)

observed.classes <- ifelse(df_test[,"type"]=="20",1,0)
mean(predicted.classes == observed.classes)

# Export de la valeur des coefficients du Lasso :
#png("coefficients_lasso.png")

palette = brewer.pal(n=10,name="RdYlGn")

image(matrix(coef(reg)[2:length(coef(reg))],c(32,32)),col=rev(palette))
title("Valeurs des coefficients")

#dev.off()


# ----
#   
#   #### 3. <a name="keras"> Le tri des images par un perceptron multicouche
#   
#   1. Une première version du réseau avec la librairie *Keras*
#   Une version des programmes avec *pytorch* figure dans le notebook ().
# ----
library("tensorflow")
# le programme crée un graph à exécuter plus tard (utile pour le calcul futur de gradient)
tf$compat$v1$disable_eager_execution()

# 1. on crée un réseau de neurones avec `Keras` avec 20 neurones sur la couche cachée.
model <- keras_model_sequential()
model %>%
  # couche cachée avec 20 neurones et 1024 = 32x32 valeurs en entrée (la couche d'entrée est implicite)
  layer_dense(units = 150, activation = 'relu',input=c(1024)) %>%
  # couche de sortie (2 neurones car deux choix possibles = table ou lit)
  layer_dense(units = 2) %>%
  # une activation softmax = somme des probas normalisée à 1
  layer_activation('softmax')

# Avec la librairie `Keras`, le modèle doit être compilé.
model %>% compile(
  # algorithme pour l'optimisation de la fonction de coût
  optimizer = 'adam', 
  # fonction de coût (ou de perte) : ici, le résultat attendu est binaire => entropie croisée
  loss = 'categorical_crossentropy',
  # métrique pour l'évaluation du modèle (% de prédictions correctes)
  metrics = c('accuracy')
)

summary(model)

# Mise en forme des données (ellees doivent être au format numérique)
x_train = data.matrix(df_train[,1:(ncol(df_train)-1)])
x_test = data.matrix(df_test[,1:(ncol(df_test)-1)])

y_train = to_categorical(ifelse(df_train[,"type"]=="20",1,0))
y_test = to_categorical(ifelse(df_test[,"type"]=="20",1,0))

On estime le modèle :
  history <- model %>% fit(x_train, y_train, epochs = 100, verbose = 2)

# tracé des itérations 
plot(history)

Evaluation sur l'échantillon test :
model %>% evaluate(x_test, y_test)


png("sortie_keras.png")

class_names = c("lit","chaise")

# on mélange les observations de l'échantillon de test
rows <- sample(nrow(x_test))
x_test <- x_test[rows,]
y_test <- y_test[rows,]
prediction <- model %>% predict(x_test)

# on affiche les résultats sur les 25 premières images
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- array_reshape(x_test[i,],dim=c(32,32))
  #img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(prediction[i,])-1
  true_label <- y_test[i,2]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:32, 1:32, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label+1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}

dev.off()




# 2. Une adaptation de l'algorithme Grad-Cam à l'aide de la librairie Keras



# Un petit rappel de la structure du modèle :
#   Le modèle est constitué de trois couches. On cherche à afficher les activations en entrée de la première couche.

summary(model)

# On se propose ici de visualiser les valeurs des pixels de l'image d'entrée $x_i$ x $\frac{\partial{L_d}}{\partial x_i}$ (valeur du pixel x dérivée de l'activation de la couche de sortie (avant l'activation) par rapport à $x_i$).

# fonction d'écriture du fichier des activations sous forme d'image

write_heatmap <- function(heatmap, filename, width = 150, height = 150,
                          bg = "white", col = palette) {
  
  palette = brewer.pal(n=9,name="YlOrRd")
  
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  #rotate <- function(x) t(apply(x, 2, rev))
  image(heatmap, axes = FALSE, asp = 1, col = col)
}



get_heatmap_mlp <- function(no_input=1,no_output){
  library(keras)
  K <- backend()
  
  # selection de l'image no_input dans l echantillon test
  image = matrix(x_test[no_input,],c(1,1024))
  
  # selection de la couche de sortie (avant l'activation) => il s'agit d'un tenseur
  out <- model$layers[[2]]
  meuble_output <- out$output[, no_output]
  # on veut la dérivée de la réponse de sortie (no_output) par rapport aux valeurs d'entrée
  grads <- K$gradients(meuble_output, model$input)[[1]]
  
  # normalisation des gradients sur la couche d'entrée
  #pooled_grads <- K$mean(grads, axis = c(0L,1L))
  pooled_grads <- grads / ((k_sqrt(k_mean(k_square(grads)))) + 1e-5)
  
  
  # itération sur les valeurs de l'image en entrée afin d'obtenir les valeurs numériques du gradient associé
  iterate <- K$`function`(list(model$input), 
                          list(pooled_grads))
  
  
  c(pooled_grads_value) %<-% iterate(list(image))
  
  # on effectue la multiplication (valeur x gradient)
  mult <- image
  for (i in 1:1024) {
    mult[,i] <- 
      mult[,i] * pooled_grads_value[[i]] 
  }
  
  mult <- pmax(mult, 0) 
  #mult <- mult / max(mult)
  
  return(mult)
}


# Calcul sur une seule image :


i=3

# on calcul les predictions sur l'echantillon test 
prediction <- model %>% predict(x_test)

grays = brewer.pal(n = 9, name = "Greys")
write_heatmap(matrix(x_test[i,],c(32,32)),"original.png",col=rev(grays))
#write_heatmap(imgs,"original.png",col=rev(grays))

heatmap <- get_heatmap_mlp(no_input=i,no_output=1)

write_heatmap(matrix(heatmap,c(32,32)),"activations.png")

image <- image_read("original.png")
info <- image_info(image)
geometry <- sprintf("%dx%d!",info$width,info$height)


predicted_label <- which.max(prediction[i,])-1
true_label <- y_test[i,2]
if (predicted_label == true_label) {
  color <- '#008800' 
} else {
  color <- '#bb0000'
}

#png("chaise_zone2.png")

image_read("activations.png") %>% 
  image_resize(geometry,filter="quadratic") %>%
  image_composite(image,operator="blend",compose_args = "30") %>% 
  plot()
title(paste0(class_names[predicted_label+1], " (",
             class_names[true_label + 1], ")"),col.main=color)

#dev.off()

# Visualisation des $x_i$ x $\frac{\partial{L_d}}{\partial x_i}$ (par rapport à l'output sélectionné par le réseau) sur les 25 premières images :

#png("gradients.png")

par(mfcol=c(5,5))
par(mar=c(0, 0, 1, 0), xaxs='i', yaxs='i')

prediction <- model %>% predict(x_test)



for (i in 1:25) { 
  grays = brewer.pal(n = 9, name = "Greys")
  write_heatmap(matrix(x_test[i,],c(32,32)),"original.png",col=rev(grays))
  
  heatmap <- get_heatmap_mlp(no_input=i,no_output=which.max(prediction[i,]))
  
  write_heatmap(matrix(heatmap,c(32,32)),"heatmap.png")
  
  image <- image_read("original.png")
  info <- image_info(image)
  geometry <- sprintf("%dx%d!",info$width,info$height)
  
  
  predicted_label <- which.max(prediction[i,])-1
  true_label <- y_test[i,2]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  
  
  image_read("heatmap.png") %>% 
    #image_resize(geometry,filter="quadratic") %>%
    image_composite(image,operator="blend",compose_args = "30") %>% 
    plot()
  title(paste0(class_names[predicted_label+1], " (",
               class_names[true_label + 1], ")"),col.main=color)
}

#dev.off()

img <- array_reshape(1-x_test[1,],dim=c(32,32))
image(1:32, 1:32, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
      main = "inversion des couleurs")


model %>% evaluate(1-x_test, y_test)

# 
# ----
#   
#   #### Sélection du nombre de neurones de la couche cachée
#   
#   ----
#   <a name="nb_neurones"> </a>
#   
#   Préparation des données pour une recherche d'hyper-paramètre. On subdivise l'échantillon d'entraînement.

# on melange les observations de la table initiale
rows <- sample(nrow(df_train))
df_train <- df_train[rows,]

# On separe la table en deux pour tester les valeurs des hyper-paramètres :
nbobs <- as.integer(nrow(df_train)*0.8)
df_train_e <- df_train[1:nbobs,]
df_train_t <- df_train[nbobs:nrow(df_train),]

# affichage des dimensions des deux tables
print("Taille de l'échantillon d'entrainement :")
print(dim(df_train_e))
print("Taille de l'échantillon test :")
print(dim(df_train_t))

x_train_e = data.matrix(df_train_e[,1:(ncol(df_train_e)-1)])
x_train_t = data.matrix(df_train_t[,1:(ncol(df_train_t)-1)])

y_train_e = to_categorical(ifelse(df_train_e[,"type"]=="20",1,0))
y_train_t = to_categorical(ifelse(df_train_t[,"type"]=="20",1,0))


# Test des paramètres sur une grille de valeurs


# selection des parametres sur une grille
# on definit une fonction contenant le modèle avec pour parametre le nombre de neurones de la couche cachée
# en sortie : la précision sur les echnatillons test et entrainement

objective_grille <- function(x){
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = x, activation = 'relu',input=c(1024)) %>%
    layer_dense(units = 2, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam', 
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  history <- model %>% fit(x_train_e, y_train_e, epochs = 100, verbose = 2)
  acc_train <- model %>% evaluate(x_train_e, y_train_e)
  
  results <- model %>% evaluate(x_train_t, y_train_t)
  return(list(results$accuracy,acc_train$accuracy))
}



acc_test = c()
acc_train = c()

for(p in seq(10,200,10)){
  
  acc = objective_grille(p)
  print(paste(p,acc,sep=":"))
  
  write(paste(p,acc[1],sep=","),"test_keras_s.txt",append=TRUE)
  write(paste(p,acc[2],sep=","),"train_keras_s.txt",append=TRUE)
  
  acc_test = c(acc_test,acc[1])
  acc_train = c(acc_train,acc[2])
}

library("tikzDevice")

graph <- data.frame(cbind(c(rep("train",20),rep("test",20)),c(seq(10,200,10),seq(10,200,10)),c(unlist(acc_test),unlist(acc_train))))
colnames(graph) <- c("source","x","y")
graph[,2:ncol(graph)] <- apply(graph[,2:ncol(graph)],2,"as.numeric")



library(tidyverse)
library(ggplot2)
theme_set(theme_classic())


# Combine into single data frame and add interpolation column
#ggplot(graph, aes(x, y)) +
#  geom_point(aes(colour = source)) + geom_smooth(method = "lm", formula = y ~ poly(x, 2),se = FALSE)
tikz("iterations_chaises.tex",width=5,height=5)

ggplot(graph,aes(x=x,y=y,colour=factor(source)))+
  geom_point(data=subset(graph, source == "train")) + geom_point(data=subset(graph, source=="test")) +
  geom_smooth(data=subset(graph, source=="train"), method='loess',formula=y~x,se=F) +
  geom_smooth(data=subset(graph, source=="test"), method='loess',formula=y~x,se=F)

dev.off()


# Test du package `hyperopt` sous R => `hopticulate`


# chargement des packages
library(reticulate)
#remotes::install_github("njnmco/hopticulate",force=TRUE)
#reticulate::py_install("hyperopt")
library(hopticulate)



# definition de la& fonction objectif
# = le pramatre x est le nombre de neurones de la couche cachée
# la sortie est le nombre de mal classées = objectif à minimiser
objective <- function(x){
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = x, activation = 'relu',input=c(1024)) %>%
    #layer_dense(units = 10, activation = 'relu') %>%
    layer_dense(units = 2, activation = 'softmax')
  
  model %>% compile(
    optimizer = 'adam', 
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  history <- model %>% fit(x_train_e, y_train_e, epochs = 5, verbose = 2)
  
  results <- model %>% evaluate(x_train_t, y_train_t)
  return(1-results$accuracy)
}

# Define a search space.
space = hp.quniform("x", 10, 200, 10)
# recherche du meilleur paramètre
best = fmin(objective, space, algo=tpe.suggest, max_evals=15,verbose=1)


print(best)