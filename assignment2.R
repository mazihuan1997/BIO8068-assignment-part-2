library(rinat)
library(sf)

source("download_images.R") 
gb_ll <- readRDS("gb_simple.RDS")


Dandelion_recs <-  get_inat_obs(taxon_name  = "taraxacum officinale",
                                #bounds = gb_ll,
                                quality = "research",
                                
                                maxresults = 600)
download_images(spp_recs = Dandelion_recs, spp_folder = "Dandelion")

wisteria_recs <-  get_inat_obs(taxon_name  = "Wisteria sinensis",
                                     #bounds = gb_ll,
                                     quality = "research",
                                     
                                     maxresults = 600)
download_images(spp_recs = wisteria_recs, spp_folder = "wisteria")

Whiterampingfumitory_recs <-  get_inat_obs(taxon_name  = "Fumaria capreolata",
                                       #bounds = gb_ll,
                                       quality = "research",
                                       
                                       maxresults = 600)
download_images(spp_recs = Whiterampingfumitory_recs, spp_folder = "Whiterampingfumitory")

# path to folder with photos
image_files_path <- "images"

spp_list <- dir(image_files_path)

# number of spp classes
output_n <- length(spp_list)

# Create test, and species sub-folders
for(folder in 1:output_n){
  dir.create(paste("test", spp_list[folder], sep="/"), recursive=TRUE)
}

# Now copy over spp_501.jpg to spp_600.jpg using two loops, deleting the photos
# from the original images folder after the copy
for(folder in 1:output_n){
  for(image in 501:600){
    src_image  <- paste0("images/", spp_list[folder], "/spp_", image, ".jpg")
    dest_image <- paste0("test/"  , spp_list[folder], "/spp_", image, ".jpg")
    file.copy(src_image, dest_image)
    file.remove(src_image)
  }
}

library(keras)

# image size to scale down to (original images vary but about 400 x 500 px)
img_width <- 150
img_height <- 150
target_size <- c(img_width, img_height)

# Full-colour Red Green Blue = 3 channels
channels <- 3

# Rescale from 255 to between zero and 1
train_data_gen = image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)


# training images
train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "training",
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "validation",
                                                    seed = 42)
# Check that things seem to have been read in OK
cat("Number of images per class:")
table(factor(train_image_array_gen$classes))
cat("Class labels vs index mapping")
train_image_array_gen$class_indices
plot(as.raster(train_image_array_gen[[1]][[1]][8,,,]))


# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32
epochs <- 10

# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 
print(model)

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# Train the model with fit_generator
history <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)

plot(history)

detach("package:imager", unload = TRUE)
save.image("assignment2.RData")


#test
path_test <- "test"

test_data_gen <- image_data_generator(rescale = 1/255)

test_image_array_gen <- flow_images_from_directory(path_test,
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = "categorical",
                                                   classes = spp_list,
                                                   shuffle = FALSE, # do not shuffle the images around
                                                   batch_size = 1,  # Only 1 image at a time
                                                   seed = 123)

model %>% evaluate_generator(test_image_array_gen, 
                             steps = test_image_array_gen$n)

#Making a prediction for a single image
test_image_plt <- imager::load.image("test/wisteria/spp_518.jpg")
plot(test_image_plt)

# Need to import slightly differently resizing etc. for Keras
test_image <- image_load("test/wisteria/spp_518.jpg",
                         target_size = target_size)

test_image <- image_to_array(test_image)
test_image <- array_reshape(test_image, c(1, dim(test_image)))
test_image <- test_image/255

# Now make the prediction
pred <- model %>% predict(test_image)
pred <- data.frame("Species" = spp_list, "Probability" = t(pred))
pred <- pred[order(pred$Probability, decreasing=T),][1:3,]
pred$Probability <- paste(round(100*pred$Probability,2),"%")
pred

predictions <- model %>% 
  predict_generator(
    generator = test_image_array_gen,
    steps = test_image_array_gen$n
  ) %>% as.data.frame
colnames(predictions) <- spp_list

# Create 3 x 3 table to store data
confusion <- data.frame(matrix(0, nrow=3, ncol=3), row.names=spp_list)
colnames(confusion) <- spp_list

obs_values <- factor(c(rep(spp_list[1],100),
                       rep(spp_list[2], 100),
                       rep(spp_list[3], 100)))
pred_values <- factor(colnames(predictions)[apply(predictions, 1, which.max)])

library(caret)
conf_mat <- confusionMatrix(data = pred_values, reference = obs_values)
conf_mat

detach("package:imager", unload = TRUE)
save.image("assignment2.1.RData")
