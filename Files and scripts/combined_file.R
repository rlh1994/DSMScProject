# Libraries ---------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(tidyr)
library(ggforce)


# Functions ---------------------------------------------------------------
get_avg_pred_cols <- function(df){
  df %>%
    pivot_longer(`Arts & Photography`:Travel, names_to = 'Predicted_Class', values_to = 'Prediction_Perc') %>%
    group_by(Filename) %>%
    filter(Prediction_Perc == max(Prediction_Perc)) %>%
    left_join(test_data) %>%
    group_by(Predicted_Class) %>% 
    summarise(Pred_Avg_red = mean(Avg_Red)/255, 
              Pred_Avg_green = mean(Avg_Green)/255,
              Pred_Avg_blue = mean(Avg_Blue)/255) %>%
    return()
}

get_class_confidence <- function(df){
  df %>%
    pivot_longer(`Arts & Photography`:Travel, names_to = 'Predicted_Class', values_to = 'Prediction_Perc') %>%
    group_by(Filename) %>%
    arrange(desc(Prediction_Perc)) %>%
    slice(1:2) %>%
    summarise(uncertainty = min(Prediction_Perc)/max(Prediction_Perc)) %>%
    left_join(test_data) %>%
    group_by(Category) %>% 
    summarise(avg_uncertainty = mean(uncertainty)) %>%
    return()
}

get_class_overlap <- function(df){
  temp_df <- df %>%
    pivot_longer(`Arts & Photography`:Travel, names_to = 'Predicted_Class', values_to = 'Prediction_Perc') 
  
  left_join(temp_df, temp_df, by = c('Filename' = 'Filename')) %>%
    mutate(class_overlap = Prediction_Perc.x/Prediction_Perc.y,
           class_overlap = if_else(class_overlap > 1, 1/class_overlap, class_overlap)) %>%
    filter(Predicted_Class.x >= Predicted_Class.y)  %>% # deal with double on all but equal joins
    
    group_by(Predicted_Class.x, Predicted_Class.y) %>% 
    summarise(avg_overlap = mean(class_overlap)) %>%
    return()
}


# Reference ---------------------------------------------------------------

c_names <- c('Arts & Photography',
             'Biographies & Memoirs',
             'Business & Money',
             'Calendars',
             "Children's Books",
             'Comics & Graphic Novels',
             'Computers & Technology',
             'Cookbooks, Food & Wine',
             'Crafts, Hobbies & Home',
             'Christian Books & Bibles',
             'Engineering & Transportation',
             'Health, Fitness & Dieting',
             'History',
             'Humor & Entertainment',
             'Law',
             'Literature & Fiction',
             'Medical Books',
             'Mystery, Thriller & Suspense',
             'Parenting & Relationships',
             'Politics & Social Sciences',
             'Reference',
             'Religion & Spirituality',
             'Romance',
             'Science & Math',
             'Science Fiction & Fantasy',
             'Self-Help',
             'Sports & Outdoors',
             'Teen & Young Adult',
             'Test Preparation',
             'Travel',
             'Filename')

classes <- c("0" = 'Arts & Photography',
             "1" = 'Biographies & Memoirs',
             "2" = 'Business & Money',
             "3" = 'Calendars',
             "4" = "Children's Books",
             "5" = 'Comics & Graphic Novels',
             "6" = 'Computers & Technology',
             "7" = 'Cookbooks, Food & Wine',
             "8" = 'Crafts, Hobbies & Home',
             "9" = 'Christian Books & Bibles',
             "10" = 'Engineering & Transportation',
             "11" = 'Health, Fitness & Dieting',
             "12" = 'History',
             "13" = 'Humor & Entertainment',
             "14" = 'Law',
             "15" = 'Literature & Fiction',
             "16" = 'Medical Books',
             "17" = 'Mystery, Thriller & Suspense',
             "18" = 'Parenting & Relationships',
             "19" = 'Politics & Social Sciences',
             "20" = 'Reference',
             "21" = 'Religion & Spirituality',
             "22" = 'Romance',
             "23" = 'Science & Math',
             "24" = 'Science Fiction & Fantasy',
             "25" = 'Self-Help',
             "26" = 'Sports & Outdoors',
             "27" = 'Teen & Young Adult',
             "28" = 'Test Preparation',
             "29" = 'Travel')


# Load Data ---------------------------------------------------------------

mob_training_data <- read.csv("Files and scripts/mobilenet_model_history_log.csv")
inc_training_data <- read.csv("Files and scripts/inception_resenetv2_model_history_log.csv")
res_training_data <- read.csv("Files and scripts/resnext_model_history_log.csv")

noprep_training_data <- read.csv("Files and scripts/noprep_model_history_log.csv")
padded_training_data <- read.csv("Files and scripts/padded_model_history_log.csv")
cropped_training_data <- read.csv("Files and scripts/cropped_model_history_log.csv")

mob_results <- read.csv("Files and scripts/mobilenetV2_preds.csv",
                        col.names = c_names, check.names = FALSE)
inc_results <- read.csv("Files and scripts/InceptionResnetV2_preds.csv",
                        col.names = c_names, check.names = FALSE)
res_results <- read.csv("Files and scripts/Resnext_preds.csv",
                        col.names = c_names, check.names = FALSE)

train_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_train.csv")
valid_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_valid.csv")
test_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_test.csv")

results <- read.csv("Files and scripts/all_top_n_results.csv")


# Clean Data --------------------------------------------------------------

comb_data_core <- mob_training_data %>% mutate(Model = 'MobileNetV2') %>%
  rbind(inc_training_data %>% mutate(Model = 'Inception-ResnetV2')) %>%
  rbind(res_training_data %>% mutate(Model = 'ResNeXt50'))


data_to_plot_core <- comb_data_core %>% 
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

comb_data_pre <- noprep_training_data %>% mutate(Model = 'No Prep') %>%
  rbind(padded_training_data %>% mutate(Model = 'Padded')) %>%
  rbind(cropped_training_data %>% mutate(Model = 'Cropped'))

data_to_plot_pre <- comb_data_pre %>% 
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

results_clean <- results %>% filter(Class != 'Total', N == 1) %>%
  left_join(data.frame(classes) %>% mutate(Class = row.names(.)) )

true_class_colours <- test_data %>% 
  group_by(Category) %>% 
  summarise(Avg_red = mean(Avg_Red)/255, 
            Avg_green = mean(Avg_Green)/255,
            Avg_blue = mean(Avg_Blue)/255)


mob_avg_cols <- get_avg_pred_cols(mob_results)
inc_avg_cols <- get_avg_pred_cols(inc_results)
res_avg_cols <- get_avg_pred_cols(res_results)

comb_col_data <- mob_avg_cols %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_avg_cols %>% mutate(model = 'InceptionResnetV2')) %>%
  rbind(res_avg_cols %>% mutate(model = 'ResneXt50')) %>%
  left_join(true_class_colours, by = c('Predicted_Class' = 'Category'))

data_to_plot_col <- comb_col_data %>% 
  mutate(col_dist = sqrt((Avg_red - Pred_Avg_red)^2 + 
                           (Avg_green - Pred_Avg_green)^2 + 
                           (Avg_blue - Pred_Avg_blue)^2),
         pred_col = rgb(Pred_Avg_red, Pred_Avg_green, Pred_Avg_blue),
         true_col = rgb(Avg_red, Avg_green, Avg_blue)) %>%
  left_join(results_clean, by = c('Predicted_Class' = 'classes', "model" = 'Model')) %>%
  select(Predicted_Class, model, col_dist, pred_col, true_col, accuracy = Result)

comb_data_dist <- train_data %>% 
  rbind(valid_data, test_data)

pivot_data <- comb_data_dist %>% mutate(Aspect_Ratio = Height/Width) %>%
  pivot_longer(Height:Aspect_Ratio, names_to = 'measure')


results_clean_res <- results %>% filter(N %in% c(1, 3, 5)) %>%
  left_join(data.frame(classes) %>% mutate(Class = row.names(.)) ) %>%
  mutate(classes = if_else(is.na(classes), 'All', classes),
         N = paste('Top', N)) %>%
  select(-Class)


mob_uncert <- get_class_confidence(mob_results)
inc_uncert <- get_class_confidence(inc_results)
res_uncert <- get_class_confidence(res_results)


comb_uncert_data <- mob_uncert %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_uncert %>% mutate(model = 'InceptionResnetV2')) %>%
  rbind(res_uncert %>% mutate(model = 'ResneXt50')) %>%
  left_join(results_clean_res %>% filter(classes != 'All', N == 'Top 1'), by = c('Category' = 'classes'))



mob_overlap <- get_class_overlap(mob_results)
inc_overlap <- get_class_overlap(inc_results)
res_overlap <- get_class_overlap(res_results)


comb_overlap_data <- mob_overlap %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_overlap %>% mutate(model = 'InceptionResnetV2')) %>%
  rbind(res_overlap %>% mutate(model = 'ResneXt50')) 




# Plots -------------------------------------------------------------------
data_to_plot_core %>%
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



data_to_plot_pre %>%
  ggplot(aes(x = epoch, y= value, group = Model)) + 
  geom_line(aes(col = Model), size = 1.25) + 
  geom_mark_ellipse(aes(filter = best_val_acc == 1), expand = unit(2, "mm")) +
  facet_grid(measure~data_type, scales = 'free_y') +
  theme_bw() +
  labs(x = 'Epoch',
       y = 'Measure Value',
       caption = 'Circled validation accuracies are the best for each model and were used as the final model for testing') +
  scale_color_brewer(palette="Set1", name = 'Pre-Processing Method') +
  theme(legend.position = 'bottom',
        plot.caption = element_text(face = 'italic'))


data_to_plot_col %>%
  mutate(class2 = case_when(
    Predicted_Class %in% c('Test Preparation',
                           'Comics & Graphic Novels',
                           'Travel',
                           'Computers & Technology',
                           'Romance')~ Predicted_Class,
    TRUE ~ 'Other Classes'
  )) %>%
  ggplot(aes(x = accuracy, y = col_dist, col = class2)) + 
  geom_point(size = 2) + 
  scale_color_brewer(palette="Set2", name = 'Class Callout') +
  facet_wrap(~model) +
  theme_bw() +
  scale_y_log10() +
  labs(x = 'Test Accuracy',
       y = 'Colour Distance (Euclidian) (Log Scale)',
       caption = 'Top 5 classes by Average Accuracy across all models called out specifically') +
  theme(legend.position = 'bottom',
        plot.caption = element_text(face = 'italic')) +
  scale_x_continuous(labels = scales::percent_format())

pivot_data %>% 
  filter(measure != 'Height', 
         ((measure == 'Aspect_Ratio' & value < 10) | measure != 'Aspect_Ratio'),
         Dataset == 'Train') %>%
  
  ggplot(aes(x = reorder(Category, desc(Category)),  group = Category, fill = measure)) + 
  geom_boxplot(aes(y= value)) +
  facet_wrap(~measure, scales = 'free_x') +
  coord_flip() +
  theme_bw() +
  labs(y = 'Measure Value', x = 'Category') +
  scale_fill_manual(values = c('white', 'blue', 'green', 'red', 'white')) +
  theme(legend.position = 'none')


comb_data_dist %>%
  group_by(Category) %>%
  mutate(Avg_Color = rgb(mean(Avg_Red)/255, mean(Avg_Green)/255, mean(Avg_Blue)/255)) %>% 
  ungroup() %>%
  select(Category, Avg_Color) %>% 
  distinct() %>%
  ggplot(aes(x = 1, y = reorder(Category, desc(Category)), fill = Avg_Color)) + 
  geom_col() +
  scale_fill_identity() +
  theme_minimal() + 
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank()) +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = 'Category',
       title = 'Average Cover Colour by Category')

results_clean_res %>%
  ggplot(aes(x = Result, y = reorder(classes, desc(classes)), group = Model, col = Model)) + 
  geom_point(shape = 15, alpha = 0.7, size = 2.5) + 
  scale_color_brewer(palette="Set1", name = 'Model') +
  facet_wrap(~N, scales = 'free_x') +
  theme_bw() +
  labs(x = 'Test Accuracy',
       y = 'True Class',
       caption = 'Top 5 classes by Average Accuracy across all models called out specifically') +
  theme(legend.position = 'bottom',
        plot.caption = element_text(face = 'italic'),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_line(colour = 'grey')) +
  scale_x_continuous(labels = scales::percent_format())

plot(comb_uncert_data$avg_uncertainty, comb_uncert_data$Result)
# no correlation really


comb_overlap_data %>% 
  rbind(comb_overlap_data %>% rename(Predicted_Class.x = Predicted_Class.y, Predicted_Class.y = Predicted_Class.x)) %>%
  distinct() %>%
  filter(avg_overlap > 0.5) %>%
  ggplot(aes(x = Predicted_Class.x, y =reorder(Predicted_Class.y, desc(Predicted_Class.y)), fill = avg_overlap)) + 
  geom_tile() +
  facet_wrap(~model, ncol = 1) +
  viridis::scale_fill_viridis(name = 'Average Overlap') +
  theme_bw() +
  #scale_y_discrete(guide = guide_axis(n.dodge=2)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        #legend.position = 'top',
        plot.caption = element_text(face = 'italic')) +
  labs(y = 'Predicted Class',
       x = 'Predicted Class',
       caption = 'Only overlaps of 0.5 or greater are shown'
  )


# Analysis ----------------------------------------------------------------
#class overlap
comb_overlap_data %>% 
  filter(avg_overlap < 1) %>%
  group_by(model) %>% 
  arrange(desc(avg_overlap)) %>%
  slice(1:5) %>%
  ungroup() %>%
  select(Predicted_Class.x, Predicted_Class.y) %>% distinct()

#precision
mob_results %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_results %>% mutate(model = 'InceptionResnetV2')) %>%
  rbind(res_results %>% mutate(model = 'ResneXt50')) %>%
  pivot_longer(`Arts & Photography`:Travel, names_to = 'Predicted_Class', values_to = 'Prediction_Perc') %>%
  group_by(Filename, model) %>%
  arrange(desc(Prediction_Perc)) %>%
  slice(1) %>% 
  left_join(test_data) %>%
  select(Filename, Category, Predicted_Class, model) %>%
  mutate(correct = if_else(Predicted_Class == Category, 1, 0)) %>%
  group_by(Predicted_Class, model) %>%
  summarise(precision = sum(correct)/n()) %>%
  pivot_wider(names_from = model, values_from = precision) %>%
  clipr::write_clip()

mob_results %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_results %>% mutate(model = 'InceptionResnetV2')) %>%
  rbind(res_results %>% mutate(model = 'ResneXt50')) %>%
  pivot_longer(`Arts & Photography`:Travel, names_to = 'Predicted_Class', values_to = 'Prediction_Perc') %>%
  group_by(Filename, model) %>%
  arrange(desc(Prediction_Perc)) %>%
  slice(1) %>% 
  left_join(test_data) %>%
  select(Filename, Category, Predicted_Class, model) %>%
  mutate(correct = if_else(Predicted_Class == Category, 1, 0)) %>%
  group_by(model) %>%
  summarise(precision = sum(correct)/n()) %>%
  pivot_wider(names_from = model, values_from = precision) 
