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
data <- keras::dataset_cifar100()

# On retient les chaises et les lits. 
chairs_beds <- get_chairs_beds(data)
df_train <- chairs_beds$Train
df_test <- chairs_beds$Test

head(df_train[, c(1:10, ncol(df_train))], 2)


par(mfrow = c(2, 3))
par(mar = c(0, 0, 0, 0), xaxs = "i", yaxs = "i")

atracer <- df_train[df_train[, "type"] == "5", ]

for (i in 1:6) {
  img <- keras::array_reshape(as.matrix(atracer[i, 1:(ncol(atracer) - 1)]), dim = c(32, 32))
  image(1:32, 1:32, img, col = gray((0:255) / 255))
}


# ### <A name="classifications"> B. Catégorisation des images par différentes méthodes</a>
#
# -------
#   #### <a name="neuralnet"> 1. Un premier réseau de neurones avec la librairie *neuralnet* sous R </a>
#   -------
#
#   On teste la librarie `neuralnet` sous R qui permet de réaliser des réseaux de neurones simples (des multilayers perceptrons).
#

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
nn <- neuralnet::neuralnet(type == "20" ~ .,
  data = df_train, hidden = 20, threshold = 0.0001, err.fct = "ce", act.fct = "logistic",
  linear.output = FALSE, likelihood = TRUE, lifesign = "full"
)


## Prediction using neural network
Predict <- predict(nn, df_test[, 1:ncol(df_test) - 1])
predict <- ifelse(Predict > 0.5, 1, 0)
initial <- ifelse(df_test[, "type"] == "20", 1, 0)
table(initial, predict)

sum(as.integer(initial == predict)) / length(df_test[, "type"])

# On cherche le nombre de neurones à utiliser dans la couche cachée. Pour cela, 
# on va subdiviser l'échantillon d'entraînement afin de pouvoir tester les modèles 
# évalués sur une grille de valeurs. Une évaluation par validation croisée aurait 
# été préférable mais trop gourmande en temps de calcul.

# on melange les observations de la table initiale
rows <- sample(nrow(df_train))
df_train <- df_train[rows, ]

# On separe la table en deux pour tester les valeurs des hyper-paramètres :
nbobs <- as.integer(nrow(df_train) * 0.8)
df_train_e <- df_train[1:nbobs, ]
df_train_t <- df_train[nbobs:nrow(df_train), ]

# affichage des dimensions des deux tables
print("Taille de l'échantillon d'entrainement :", dim(df_train_e))
print("Taille de l'échantillon test :", dim(df_train_t))

accuracy_train <- c()
accuracy_test <- c()

grid_hidden <- seq(5, 100, 50)

for (k in grid_hidden) {
  print(k)

  # évaluation du réseau de neurones
  nn <- neuralnet::neuralnet(type == 20 ~ .,
    data = df_train_e, hidden = k, threshold = 0.001, act.fct = "logistic",
    linear.output = FALSE, likelihood = FALSE, lifesign = "full"
  )

  # calcul de la précision sur l'échantillon test
  print(get_accuracy(nn, df_train_e))

  accuracy_train <- c(accuracy_train, get_accuracy(nn, df_train_e))
  accuracy_test <- c(accuracy_test, get_accuracy(nn, df_train_t))
}

a_list <- list(accuracy_train, accuracy_test)
a_df <- do.call("rbind", lapply(
  a_list,
  function(x) {
    data.frame(
      value = x,
      count = grid_hidden
    )
  }
))
ID_options <- c("train", "test")
a_df$ID <- rep(ID_options, sapply(a_list, length))

ggplot2::ggplot(a_df, ggplot2::aes(x = count, y = value, color = ID)) +
  ggplot2::geom_point()


# Les performances du modèles sont réévaluées sur l'échantillon de test :
accuracy_test


# -----
#   #### <a name="lasso">2. Une régression pénalisée (Lasso)</a>
#   (ou comment résoudre la problème avec une technique de machine learning ?)
#
# -----
#
#   Une méthode de ML pour trier les images. On utilise pour cela la librairie `glmnet`.

# mise en forme des variables explicatives
x <- model.matrix(type ~ ., data.frame(df_train))[, -1]
# mise en forme de la variable à expliquer
y <- ifelse(df_train[, "type"] == "20", 1, 0)


# On estime une régression pénalisée avec les paramètres :
#   * family = "binomial" : la variable à expliquer est binaire
#
# La fonction `cv.glmnet` permet de choisir le paramètre $\lambda$ par validation croisée. Le nombre de plis par défaut est égal à 10.
#


# Build the model using the training set
reg <- glmnet::cv.glmnet(x, y, family = "binomial", lambda = NULL, alpha = 0.3)

# affichage du paramètre choisi
reg$lambda.min

# tracé de la fonction de coût
plot(reg)

# Calcul de la précision sur l'échantillon de test
x.test <- model.matrix(type ~ ., data.frame(df_test))[, -1]
probabilities <- reg |> predict(newx = x.test)
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)

observed.classes <- ifelse(df_test[, "type"] == "20", 1, 0)
mean(predicted.classes == observed.classes)

palette <- RColorBrewer::brewer.pal(n = 10, name = "RdYlGn")

image(matrix(coef(reg)[2:length(coef(reg))], c(32, 32)), col = rev(palette))
title("Valeurs des coefficients")


# ----
#
#   #### 3. <a name="keras"> Le tri des images par un perceptron multicouche
#
#   1. Une première version du réseau avec la librairie *Keras*
#   Une version des programmes avec *pytorch* figure dans le notebook ().
# ----

# le programme crée un graph à exécuter plus tard (utile pour le calcul futur de gradient)
tensorflow::tf$compat$v1$disable_eager_execution()

# 1. on crée un réseau de neurones avec `Keras` avec 20 neurones sur la couche cachée.
model <- keras::keras_model_sequential()
model |>
  # couche cachée avec 20 neurones et 1024 = 32x32 valeurs en entrée (la couche d'entrée est implicite)
  keras::layer_dense(units = 150, activation = "relu", input = c(1024)) |>
  # couche de sortie (2 neurones car deux choix possibles = table ou lit)
  keras::layer_dense(units = 2) |>
  # une activation softmax = somme des probas normalisée à 1
  keras::layer_activation("softmax")

# Avec la librairie `Keras`, le modèle doit être compilé.
model |> keras::compile(
  # algorithme pour l'optimisation de la fonction de coût
  optimizer = "adam",
  # fonction de coût (ou de perte) : ici, le résultat attendu est binaire => entropie croisée
  loss = "categorical_crossentropy",
  # métrique pour l'évaluation du modèle (% de prédictions correctes)
  metrics = c("accuracy")
)

summary(model)

# Mise en forme des données (ellees doivent être au format numérique)
x_train <- data.matrix(df_train[, 1:(ncol(df_train) - 1)])
x_test <- data.matrix(df_test[, 1:(ncol(df_test) - 1)])

y_train <- keras::to_categorical(ifelse(df_train[, "type"] == "20", 1, 0))
y_test <- keras::to_categorical(ifelse(df_test[, "type"] == "20", 1, 0))

# On estime le modèle :
history <- model |> keras::fit(x_train, y_train, epochs = 100, verbose = 2)

# tracé des itérations
plot(history)

# Evaluation sur l'échantillon test :
model |> keras::evaluate(x_test, y_test)

png("sortie_keras.png")

class_names <- c("lit", "chaise")

# on mélange les observations de l'échantillon de test
rows <- sample(nrow(x_test))
x_test <- x_test[rows, ]
y_test <- y_test[rows, ]
prediction <- model |> predict(x_test)

# on affiche les résultats sur les 25 premières images
par(mfcol = c(5, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = "i", yaxs = "i")
for (i in 1:25) {
  img <- keras::array_reshape(x_test[i, ], dim = c(32, 32))

    # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(prediction[i, ]) - 1
  true_label <- y_test[i, 2]
  if (predicted_label == true_label) {
    color <- "#008800"
  } else {
    color <- "#bb0000"
  }
  image(1:32, 1:32, img,
    col = gray((0:255) / 255), xaxt = "n", yaxt = "n",
    main = paste0(
      class_names[predicted_label + 1], " (",
      class_names[true_label + 1], ")"
    ),
    col.main = color
  )
}

# 2. Une adaptation de l'algorithme Grad-Cam à l'aide de la librairie Keras

# Un petit rappel de la structure du modèle :
#   Le modèle est constitué de trois couches. On cherche à afficher les activations en entrée de la première couche.

summary(model)

# On se propose ici de visualiser les valeurs des pixels de l'image d'entrée $x_i$ x $\frac{\partial{L_d}}{\partial x_i}$ (valeur du pixel x dérivée de l'activation de la couche de sortie (avant l'activation) par rapport à $x_i$).

par(mfcol = c(5, 5))
par(mar = c(0, 0, 1, 0), xaxs = "i", yaxs = "i")

prediction <- model |> predict(x_test)

for (i in 1:25) {
  grays <- RColorBrewer::brewer.pal(n = 9, name = "Greys")
  write_heatmap(matrix(x_test[i, ], c(32, 32)), "data/original.png", col = rev(grays))

  heatmap <- get_heatmap_mlp(model, x_test, no_input = i, no_output = which.max(prediction[i, ]))

  write_heatmap(matrix(heatmap, c(32, 32)), "data/heatmap.png")

  image <- magick::image_read("data/original.png")
  info <- magick::image_info(image)
  geometry <- sprintf("%dx%d!", info$width, info$height)


  predicted_label <- which.max(prediction[i, ]) - 1
  true_label <- y_test[i, 2]
  if (predicted_label == true_label) {
    color <- "#008800"
  } else {
    color <- "#bb0000"
  }

  magick::image_read("heatmap.png") |>
    magick::image_composite(image, operator = "blend", compose_args = "30") |>
    plot()
  
  title(paste0(
    class_names[predicted_label + 1], " (",
    class_names[true_label + 1], ")"
  ), col.main = color)
}

img <- keras::array_reshape(1 - x_test[1, ], dim = c(32, 32))
image(1:32, 1:32, img,
  col = gray((0:255) / 255), xaxt = "n", yaxt = "n",
  main = "inversion des couleurs"
)

model |> keras::evaluate(1 - x_test, y_test)

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
df_train <- df_train[rows, ]

# On separe la table en deux pour tester les valeurs des hyper-paramètres :
nbobs <- as.integer(nrow(df_train) * 0.8)
df_train_e <- df_train[1:nbobs, ]
df_train_t <- df_train[nbobs:nrow(df_train), ]

# affichage des dimensions des deux tables
print("Taille de l'échantillon d'entrainement :")
print(dim(df_train_e))
print("Taille de l'échantillon test :")
print(dim(df_train_t))

x_train_e <- data.matrix(df_train_e[, 1:(ncol(df_train_e) - 1)])
x_train_t <- data.matrix(df_train_t[, 1:(ncol(df_train_t) - 1)])

y_train_e <- keras::to_categorical(ifelse(df_train_e[, "type"] == "20", 1, 0))
y_train_t <- keras::to_categorical(ifelse(df_train_t[, "type"] == "20", 1, 0))


# Test des paramètres sur une grille de valeurs

# selection des parametres sur une grille
# on definit une fonction contenant le modèle avec pour parametre le nombre de neurones de la couche cachée
# en sortie : la précision sur les echnatillons test et entrainement


acc_test <- c()
acc_train <- c()

grid_param <- seq(10, 200, 10)

for (p in grid_param) {
  acc <- objective_grille(p)
  print(paste(p, acc, sep = ":"))

  write(paste(p, acc[1], sep = ","), "data/test_keras_s.txt", append = TRUE)
  write(paste(p, acc[2], sep = ","), "data/train_keras_s.txt", append = TRUE)

  acc_test <- c(acc_test, acc[1])
  acc_train <- c(acc_train, acc[2])
}

graph <- data.frame(cbind(c(rep("train", 20), rep("test", 20)), rep(grid_param, 2), c(unlist(acc_test), unlist(acc_train))))
colnames(graph) <- c("source", "x", "y")
graph[, 2:ncol(graph)] <- apply(graph[, 2:ncol(graph)], 2, "as.numeric")


# Combine into single data frame and add interpolation column
ggplot2::ggplot(graph, ggplot2::aes(x = x, y = y, colour = factor(source))) +
  ggplot2::geom_point(data = subset(graph, source == "train")) +
  ggplot2::geom_point(data = subset(graph, source == "test")) +
  ggplot2::geom_smooth(data = subset(graph, source == "train"), method = "loess", formula = y ~ x, se = F) +
  ggplot2::geom_smooth(data = subset(graph, source == "test"), method = "loess", formula = y ~ x, se = F) +
  ggplot2::theme_classic()



# Test du package `hyperopt` sous R => `hopticulate`

# Define a search space.
space <- hopticulate::hp.quniform("x", 10, 200, 10)
# recherche du meilleur paramètre
best <- hopticulate::fmin(objective, space, algo = hopticulate::tpe.suggest, max_evals = 15, verbose = 1)
print(best)
