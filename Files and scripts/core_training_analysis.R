library(dplyr)
library(ggplot2)
library(tidyr)
library(ggforce)

mob_training_data <- read.csv("Files and scripts/mobilenet_model_history_log.csv")
inc_training_data <- read.csv("Files and scripts/inception_resenetv2_model_history_log.csv")
res_training_data <- read.csv("Files and scripts/resnext_model_history_log.csv")

comb_data <- mob_training_data %>% mutate(Model = 'MobileNetV2') %>%
  rbind(inc_training_data %>% mutate(Model = 'Inception-ResnetV2')) %>%
  rbind(res_training_data %>% mutate(Model = 'ResNeXt50'))


data_to_plot <- comb_data %>% 
  rename(Train_accuracy = accuracy, 
         Train_loss = loss, 
         Train_sparse_top_3_categorical_accuracy = sparse_top_k_categorical_accuracy,
         val_sparse_top_3_categorical_accuracy = val_sparse_top_k_categorical_accuracy) %>%
  pivot_longer(Train_accuracy:val_sparse_top_3_categorical_accuracy, names_to = "temp") %>%
  separate(temp, c("data_type", "measure"), "_", extra = 'merge') %>%
  mutate(data_type = if_else(data_type == 'Train', 'Training', 'Validation'),
         measure = case_when(measure == 'accuracy' ~ 'Accuracy',
                             measure == 'loss' ~ 'Categorical Cross Entropy',
                             measure == 'sparse_top_3_categorical_accuracy' ~ 'Top 3 Accuracy')) %>%
  group_by(Model, data_type, measure) %>%
  mutate(best_val_acc = if_else(data_type == 'Validation' & measure == 'Accuracy' & value == max(value), 1, 0)) %>%
  ungroup()

data_to_plot %>%
  ggplot(aes(x = epoch, y= value, group = Model)) + 
  geom_line(aes(col = Model), size = 1.25) + 
  geom_mark_ellipse(aes(filter = best_val_acc == 1), expand = unit(2, "mm")) +
  facet_grid(measure~data_type, scales = 'free_y') +
  theme_bw() +
  labs(x = 'Epoch',
       y = 'Measure Value',
       caption = 'Circled validation accuracies are the best for each model and were used as the final model for testing') +
  scale_color_brewer(palette="Set1") +
  theme(legend.position = 'bottom',
        plot.caption = element_text(face = 'italic'))
