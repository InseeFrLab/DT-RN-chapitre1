# Importation des fonctions utilisées
source("R/functions.R")

# Importation des images de la base CIFAR-10 à partir de la librairie Keras
data <- keras::dataset_cifar100()

# On retient les chaises et les lits. 
chairs_beds <- get_chairs_beds(data)
df_train <- chairs_beds$Train
df_test <- chairs_beds$Test

# On visualise nos données
head(df_train[, c(1:10, ncol(df_train))], 2)

# Le programme créer un graph à exécuter plus tard (utile pour le calcul futur de gradient)
tensorflow::tf$compat$v1$disable_eager_execution()

# Definition du modele
model <- keras::keras_model_sequential() |>
  keras::layer_conv_2d(
    filters = 5, kernel_size = c(3, 3), activation = "relu",
    input_shape = c(32, 32, 1)
  ) |>
  keras::layer_max_pooling_2d(pool_size = c(2, 2)) |>
  keras::layer_dropout(rate = 0.25) |>
  keras::layer_flatten() |>
  keras::layer_dense(units = 20, activation = "relu") |>
  keras::layer_dense(units = 2) |>
  keras::layer_activation("softmax")

# Compilation du modele
model |> keras::compile(
  # algorithme pour l'optimisation de la fonction de coût
  optimizer = "adam",
  # fonction de coût (ou de perte) : ici, le résultat attendu est binaire => entropie croisée
  loss = "categorical_crossentropy",
  # métrique pour l'évaluation du modèle (% de prédictions correctes)
  metrics = c("accuracy")
)

# On définit nos valeurs à prédire
y_train <- keras::to_categorical(ifelse(df_train[, "type"] == "20", 1, 0))
y_test <- keras::to_categorical(ifelse(df_test[, "type"] == "20", 1, 0))

# On définit nos features
x_train <- data.matrix(df_train[, 1:(ncol(df_train) - 1)])
x_test <- data.matrix(df_test[, 1:(ncol(df_test) - 1)])
x_train <- keras::array_reshape(x_train, dim = c(dim(x_train)[1], 32, 32, 1))
x_test <- keras::array_reshape(x_test, dim = c(dim(x_test)[1], 32, 32, 1))

# On entraine le modèle
history <- model |> 
  keras::fit(x_train, y_train, epochs = 100, verbose = 2)

# On peut représenter graphiquement le l'historique de l'entrainement
plot(history)

# On évalue le modèle sur nos données de test
model |> 
  keras::evaluate(x_test, y_test)


# Représentation graphique des résultats du modèle
K <- keras::backend()
palette <- RColorBrewer::brewer.pal(n = 9, name = "YlOrRd")

# On mélange les images de l'échantillon test :
index <- sample(nrow(x_test))
x_test <- x_test[index, , , ]
y_test <- y_test[index, ]

## On compare pour les 25 premières images la prédiction et la vraie valeur
## graphiquement
class_names = c("lit","chaise")

## Initialisation du graphique
par(mfcol=c(5,5))
par(mar=c(0, 0, 1, 0), xaxs='i', yaxs='i')

for (i in 1:25) {
  grays <- RColorBrewer::brewer.pal(n = 9, name = "Greys")
  write_heatmap(matrix(x_test[i, , ], c(32, 32)), "data/original.png", col = rev(grays))

  input_image_data <- array(x_test[i, , ], dim = c(1, 32, 32, 1))
  predicted_label <- model |> predict(input_image_data)
  predicted_label <- which.max(predicted_label) - 1

  heatmap <- get_heatmap(model, x_test, no_input = i, no_output = predicted_label + 1, out_layer = "dense", last_conv_layer = "conv2d")

  write_heatmap(matrix(heatmap, c(30, 30)), "data/ordi.png")

  image <- magick::image_read("data/original.png")
  info <- magick::image_info(image)
  geometry <- sprintf("%dx%d!", info$width, info$height)

  palette <- RColorBrewer::brewer.pal(n = 9, name = "YlOrRd")
  pal <- col2rgb(palette, alpha = TRUE)
  alpha <- floor(seq(255, 255, length = ncol(pal)))
  pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)

  true_label <- y_test[i, 2]
  if (predicted_label == true_label) {
    color <- "#008800"
  } else {
    color <- "#bb0000"
  }

  magick::image_read("data/ordi.png") |>
    magick::image_composite(image, operator = "blend", compose_args = "30") |>
    plot()
  title(paste0(
    class_names[predicted_label + 1], " (",
    class_names[true_label + 1], ")"
  ), col.main = color)
}
