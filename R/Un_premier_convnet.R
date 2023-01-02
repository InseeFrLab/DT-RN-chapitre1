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
  grayscale <- data.frame(keras::array_reshape(grayscale,dim=c(dim(grayscale)[1],32*32)))
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




# le programme crée un graph à exécuter plus tard (utile pour le calcul futur de gradient)
tensorflow::tf$compat$v1$disable_eager_execution()

# Define model
model <- keras::keras_model_sequential() |>
  keras::layer_conv_2d(filters = 5, kernel_size = c(3,3), activation = 'relu',
                input_shape = c(32,32,1)) |>
  keras::layer_max_pooling_2d(pool_size = c(2, 2)) |>
  keras::layer_dropout(rate = 0.25) |>
  keras::layer_flatten() |>
  keras::layer_dense(units = 20, activation = 'relu') |>
  keras::layer_dense(units = 2) |>
  keras::layer_activation('softmax')

model

model |> keras::compile(
  # algorithme pour l'optimisation de la fonction de coût
  optimizer = 'adam', 
  # fonction de coût (ou de perte) : ici, le résultat attendu est binaire => entropie croisée
  loss = 'categorical_crossentropy',
  # métrique pour l'évaluation du modèle (% de prédictions correctes)
  metrics = c('accuracy')
)



x_train = data.matrix(df_train[,1:(ncol(df_train)-1)])
x_test = data.matrix(df_test[,1:(ncol(df_test)-1)])

y_train = keras::to_categorical(ifelse(df_train[,"type"]=="20",1,0))
y_test = keras::to_categorical(ifelse(df_test[,"type"]=="20",1,0))



x_train = keras::array_reshape(x_train,dim=c(dim(x_train)[1],32,32,1))
x_test = keras::array_reshape(x_test,dim=c(dim(x_test)[1],32,32,1))


history <- model |> keras::fit(x_train, y_train, epochs = 100, verbose = 2)

# tracé des itérations 
plot(history)
model |> keras::evaluate(x_test, y_test)



K <- keras::backend()

palette = RColorBrewer::brewer.pal(n=9,name="YlOrRd")

write_heatmap <- function(heatmap, filename, width = 150, height = 150, bg = "white", col = palette) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  image(heatmap, axes = FALSE, asp = 1, col = col)
}

get_heatmap <- function(model, x_test, no_input=1,no_output=2, out_layer = "dense_3", last_conv_layer="conv2d_1"){
  
  # selection de l'image no_input dans l echantillon test
  image = matrix(x_test[no_input,,],c(1,32,32,1))
  
  # selection de la couche de sortie (avant l'activation)
  out <- model |> keras::get_layer(out_layer)
  meuble_output <- out$output[, no_output]
  # selection de la couche dont on veut obtenir les activations (ici la couche d'entrée)
  last_conv_layer <- model |> keras::get_layer(last_conv_layer)
  # on veut les gradients de la couche d'entrée qui maximise la réponse no_output
  grads <- K$gradients(meuble_output, last_conv_layer$output)[[1]]
  # moyenne sur les filtres de convolutions (inutile ici)
  pooled_grads <- K$mean(grads, axis = c(0L,1L,2L))
  
  # itération sur les valeurs de l'image et des gradients associés
  iterate <- K$`function`(list(model$input), 
                          list(pooled_grads, last_conv_layer$output[1,,,])) 
  
  input_image_data <- array(image,dim=c(1,32,32,1))
  lists <- iterate(list(input_image_data))
  pooled_grads_value <- lists[[1]]
  conv_layer_output_value <- lists[[2]]
  
  # on effectue la multiplication (valeur x gradient)
  for (i in 1:5) {
    conv_layer_output_value[,,i] <-
      conv_layer_output_value[,,i] * pooled_grads_value[[i]]
  }

  heatmap <- apply(conv_layer_output_value, c(1,2), mean)

  heatmap <- pmax(heatmap, 0)
  heatmap <- heatmap / max(heatmap)
  
  return(heatmap)
  
  
}


index <- sample(nrow(x_test))
x_test <- x_test[index,,]
y_test <- y_test[index,]



i=5
class_names = c("lit","chaise")

image = x_test[i,,]
input_image_data <- array(image,dim=c(1,32,32,1))

grays = RColorBrewer::brewer.pal(n = 9, name = "Greys")
write_heatmap(matrix(x_test[i,,],c(32,32)),"original.png",col=rev(grays))

heatmap <- get_heatmap(model, x_test, no_input=i,no_output=2, out_layer = "dense_3", last_conv_layer="conv2d_1")   
write_heatmap(matrix(heatmap,c(30,30)),"ordi.png")

image <- magick::image_read("original.png")
info <- magick::image_info(image)
geometry <- sprintf("%dx%d!",info$width,info$height)

palette = RColorBrewer::brewer.pal(n=9,name="YlOrRd")
pal <- col2rgb(palette,alpha=TRUE)
alpha <- floor(seq(255,255,length=ncol(pal)))
pal_col <- rgb(t(pal),alpha=alpha,maxColorValue = 255)
#write_heatmap(matrix(heatmap,c(30,30)),"original_overlay.png",width = 14,height = 14,bg = NA,col=pal_col)


predicted_label <- prediction <- model |>predict(input_image_data)
predicted_label <- which.max(predicted_label)-1

true_label <- y_test[i,2]
if (predicted_label == true_label) {
  color <- '#008800' 
} else {
  color <- '#bb0000'
}

#png("chaise_zone2.png")

magick::image_read("ordi.png") |>
  magick::image_resize(geometry,filter="quadratic") |>
  magick::image_composite(image,operator="blend",compose_args = "30") |>
  plot()
title(paste0(class_names[predicted_label+1], " (",
             class_names[true_label + 1], ")"),col.main=color)



par(mfcol=c(5,5))
par(mar=c(0, 0, 1, 0), xaxs='i', yaxs='i')


for (i in 1:25) { 
  grays = RColorBrewer::brewer.pal(n = 9, name = "Greys")
  write_heatmap(matrix(x_test[i,,],c(32,32)),"original.png",col=rev(grays))
  
  input_image_data <- array(x_test[i,,],dim=c(1,32,32,1))
  predicted_label <- model |>predict(input_image_data)
  predicted_label <- which.max(predicted_label)-1
  
  heatmap <- get_heatmap(model, x_test, no_input=i,no_output=predicted_label+1, out_layer = "dense_3", last_conv_layer="conv2d_1")
  
  write_heatmap(matrix(heatmap,c(30,30)),"ordi.png")
  
  image <- magick::image_read("original.png")
  info <- magick::image_info(image)
  geometry <- sprintf("%dx%d!",info$width,info$height)
  
  palette = RColorBrewer::brewer.pal(n=9,name="YlOrRd")
  pal <- col2rgb(palette,alpha=TRUE)
  alpha <- floor(seq(255,255,length=ncol(pal)))
  pal_col <- rgb(t(pal),alpha=alpha,maxColorValue = 255)

  true_label <- y_test[i,2]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  
  
  magick::image_read("ordi.png") |>
    magick::image_composite(image,operator="blend",compose_args = "30") |>
    plot()
  title(paste0(class_names[predicted_label+1], " (",
               class_names[true_label + 1], ")"),col.main=color)
}
