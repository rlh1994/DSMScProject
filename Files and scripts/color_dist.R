library(dplyr)
library(ggplot2)
library(tidyr)

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


mob_results <- read.csv("Files and scripts/mobilenetV2_preds.csv",
                        col.names = c_names, check.names = FALSE)
inc_results <- read.csv("Files and scripts/InceptionResnetV2_preds.csv",
                        col.names = c_names, check.names = FALSE)
res_results <- read.csv("Files and scripts/Resnext_preds.csv",
                        col.names = c_names, check.names = FALSE)


test_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_test.csv")
results <- read.csv("Files and scripts/all_top_n_results.csv")

results_clean <- results %>% filter(Class != 'Total', N == 1) %>%
  left_join(data.frame(classes) %>% mutate(Class = row.names(.)) )



true_class_colours <- test_data %>% 
  group_by(Category) %>% 
  summarise(Avg_red = mean(Avg_Red)/255, 
            Avg_green = mean(Avg_Green)/255,
            Avg_blue = mean(Avg_Blue)/255)

# Get average colour by predicted class 
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

mob_avg_cols <- get_avg_pred_cols(mob_results)
inc_avg_cols <- get_avg_pred_cols(inc_results)
res_avg_cols <- get_avg_pred_cols(res_results)

comb_col_data <- mob_avg_cols %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_avg_cols %>% mutate(model = 'InceptionResnetV2')) %>%
    rbind(res_avg_cols %>% mutate(model = 'ResneXt50')) %>%
    left_join(true_class_colours, by = c('Predicted_Class' = 'Category'))

data_to_plot <- comb_col_data %>% 
  mutate(col_dist = sqrt((Avg_red - Pred_Avg_red)^2 + 
                           (Avg_green - Pred_Avg_green)^2 + 
                           (Avg_blue - Pred_Avg_blue)^2),
         pred_col = rgb(Pred_Avg_red, Pred_Avg_green, Pred_Avg_blue),
         true_col = rgb(Avg_red, Avg_green, Avg_blue)) %>%
  left_join(results_clean, by = c('Predicted_Class' = 'classes', "model" = 'Model')) %>%
  select(Predicted_Class, model, col_dist, pred_col, true_col, accuracy = Result)


data_to_plot %>%
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
