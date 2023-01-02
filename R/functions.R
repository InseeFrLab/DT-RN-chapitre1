get_images <- function(data, categorie, ech = "train", verbose = 0) {
  # La fonction get_images :
  # * permet de sélectionner les images dans la base cifar-10 à l'aide du numéro de catégorie.
  # * Elle renvoie une table avec une ligne par image convetie en niveaux de gris.
  # * Elle affiche les images pour vérification
  # * echantillon doit être "train" ou "test"
  
  # selection des images
  if (ech == "train") {
    images <- data$train$x[data$train$y == categorie, , , ]
  } else {
    images <- data$test$x[data$test$y == categorie, , , ]
  }
  
  # conversion des images en niveaux de gris
  grayscale <- 0.298936 * images[, , , 1] + 0.587043 * images[, , , 2] + 0.114021 * images[, , , 3]
  grayscale <- apply(grayscale, c(1, 2, 3), "as.integer")
  
  if (verbose>0) {
    # affichage des images pour vérification
    par(mfcol = c(5, 5))
    par(mar = c(0, 0, 4, 1), xaxs = "i", yaxs = "i")
    
    for (i in 1:25) {
      image(1:32, 1:32, grayscale[i, , ], col = gray((0:255) / 255))
    }
  }

  # conversion en data.frame (1 ligne par image)
  grayscale <- data.frame(keras::array_reshape(grayscale, dim = c(dim(grayscale)[1], 32 * 32)))
  grayscale["type"] <- categorie
  
  return(grayscale)
}

normalise <- function(x) {
  x <- as.numeric(x)
  x <- (x - 0) / 255
}

write_heatmap <- function(heatmap, filename, width = 150, height = 150, bg = "white", col = palette) {
  png(filename, width = width, height = height, bg = bg)
  op <- par(mar = c(0, 0, 0, 0))
  on.exit(
    {
      par(op)
      dev.off()
    },
    add = TRUE
  )
  image(heatmap, axes = FALSE, asp = 1, col = col)
}

get_heatmap <- function(model, x_test, no_input = 1, no_output = 2, out_layer = "dense", last_conv_layer = "conv2d") {
  # selection de l'image no_input dans l echantillon test
  image <- matrix(x_test[no_input, , ], c(1, 32, 32, 1))
  
  # selection de la couche de sortie (avant l'activation)
  out <- model |> keras::get_layer(out_layer)
  meuble_output <- out$output[, no_output]
  # selection de la couche dont on veut obtenir les activations (ici la couche d'entrée)
  last_conv_layer <- model |> keras::get_layer(last_conv_layer)
  # on veut les gradients de la couche d'entrée qui maximise la réponse no_output
  grads <- K$gradients(meuble_output, last_conv_layer$output)[[1]]
  # moyenne sur les filtres de convolutions (inutile ici)
  pooled_grads <- K$mean(grads, axis = c(0L, 1L, 2L))
  
  # itération sur les valeurs de l'image et des gradients associés
  iterate <- K$`function`(
    list(model$input),
    list(pooled_grads, last_conv_layer$output[1, , , ])
  )
  
  input_image_data <- array(image, dim = c(1, 32, 32, 1))
  lists <- iterate(list(input_image_data))
  pooled_grads_value <- lists[[1]]
  conv_layer_output_value <- lists[[2]]
  
  # on effectue la multiplication (valeur x gradient)
  for (i in 1:5) {
    conv_layer_output_value[, , i] <-
      conv_layer_output_value[, , i] * pooled_grads_value[[i]]
  }
  
  heatmap <- apply(conv_layer_output_value, c(1, 2), mean)
  
  heatmap <- pmax(heatmap, 0)
  heatmap <- heatmap / max(heatmap)
  
  return(heatmap)
}

get_heatmap_mlp <- function(model, x_test, no_input = 1, no_output) {
  K <- keras::backend()
  
  # selection de l'image no_input dans l echantillon test
  image <- matrix(x_test[no_input, ], c(1, 1024))
  
  # selection de la couche de sortie (avant l'activation) => il s'agit d'un tenseur
  out <- model$layers[[2]]
  meuble_output <- out$output[, no_output]
  
  # on veut la dérivée de la réponse de sortie (no_output) par rapport aux valeurs d'entrée
  grads <- K$gradients(meuble_output, model$input)[[1]]
  
  # normalisation des gradients sur la couche d'entrée
  # pooled_grads <- K$mean(grads, axis = c(0L,1L))
  pooled_grads <- grads / ((keras::k_sqrt(keras::k_mean(keras::k_square(grads)))) + 1e-5)
  
  
  # itération sur les valeurs de l'image en entrée afin d'obtenir les valeurs numériques du gradient associé
  iterate <- K$`function`(
    list(model$input),
    list(pooled_grads)
  )
  
  
  pooled_grads_value <- iterate(list(image))[[1]]
  
  # on effectue la multiplication (valeur x gradient)
  mult <- image
  for (i in 1:1024) {
    mult[, i] <-
      mult[, i] * pooled_grads_value[[i]]
  }
  
  mult <- pmax(mult, 0)
  # mult <- mult / max(mult)
  
  return(mult)
}

# fonction de calcul de la précision
get_accuracy <- function(nn, table) {
  Predict <- predict(nn, table[, 1:ncol(table) - 1])
  predict <- ifelse(Predict > 0.5, 1, 0)
  initial <- ifelse(table[, "type"] == "20", 1, 0)
  return(sum(as.integer(initial == predict)) / length(table[, "type"]))
}

objective_grille <- function(x) {
  model <- keras::keras_model_sequential()
  model |>
    keras::layer_dense(units = x, activation = "relu", input = c(1024)) |>
    keras::layer_dense(units = 2, activation = "softmax")
  
  model |> keras::compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  history <- model |> keras::fit(x_train_e, y_train_e, epochs = 100, verbose = 2)
  acc_train <- model |> keras::evaluate(x_train_e, y_train_e)
  
  results <- model |> keras::evaluate(x_train_t, y_train_t)
  return(list(results$accuracy, acc_train$accuracy))
}

# definition de la fonction objectif
# = le pramatre x est le nombre de neurones de la couche cachée
# la sortie est le nombre de mal classées = objectif à minimiser
objective <- function(x) {
  model <- keras::keras_model_sequential()
  model |>
    keras::layer_dense(units = x, activation = "relu", input = c(1024)) |>
    keras::layer_dense(units = 2, activation = "softmax")
  
  model |> keras::compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  history <- model |> keras::fit(x_train_e, y_train_e, epochs = 5, verbose = 2)
  
  results <- model |> keras::evaluate(x_train_t, y_train_t)
  return(1 - results$accuracy)
}

get_chairs_beds <- function(data) {
  # importation de l'échantillon d'entrainement
  beds <- get_images(data, "5", ech = "train")
  lits <- get_images(data, "20", ech = "train")
  df_train <- rbind(beds, lits)
  
  # importation de l'échantillon de test
  beds <- get_images(data, "5", ech = "test")
  lits <- get_images(data, "20", ech = "test")
  df_test <- rbind(beds, lits)
  
  
  df_train[, 1:(ncol(df_train) - 1)] <- apply(df_train[, 1:(ncol(df_train) - 1)], c(1, 2), "normalise")
  df_test[, 1:(ncol(df_test) - 1)] <- apply(df_test[, 1:(ncol(df_test) - 1)], c(1, 2), "normalise")
  
  return(list("Train" = df_train, "Test" = df_test))
}
