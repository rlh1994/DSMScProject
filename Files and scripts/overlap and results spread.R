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

results_clean <- results %>% filter(N %in% c(1, 3, 5)) %>%
  left_join(data.frame(classes) %>% mutate(Class = row.names(.)) ) %>%
  mutate(classes = if_else(is.na(classes), 'All', classes),
         N = paste('Top', N)) %>%
  select(-Class)



results_clean %>%
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



# Uncertainty by class ----------------------------------------------------
# get prediction "confidence" by class

# Get average colour by predicted class 
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

mob_uncert <- get_class_confidence(mob_results)
inc_uncert <- get_class_confidence(inc_results)
res_uncert <- get_class_confidence(res_results)


comb_uncert_data <- mob_uncert %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_uncert %>% mutate(model = 'InceptionResnetV2')) %>%
  rbind(res_uncert %>% mutate(model = 'ResneXt50')) %>%
  left_join(results_clean %>% filter(classes != 'All', N == 'Top 1'), by = c('Category' = 'classes'))


plot(comb_uncert_data$avg_uncertainty, comb_uncert_data$Result)
# no correlation really



# Overlap of topics -------------------------------------------------------

# Get average colour by predicted class 
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


mob_overlap <- get_class_overlap(mob_results)
inc_overlap <- get_class_overlap(inc_results)
res_overlap <- get_class_overlap(res_results)


comb_overlap_data <- mob_overlap %>% mutate(model = 'MobileNetV2') %>%
  rbind(inc_overlap %>% mutate(model = 'InceptionResnetV2')) %>%
  rbind(res_overlap %>% mutate(model = 'ResneXt50')) 


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

comb_overlap_data %>% 
  filter(avg_overlap < 1) %>%
  group_by(model) %>% 
  arrange(desc(avg_overlap)) %>%
  slice(1:5) %>%
  ungroup() %>%
  select(Predicted_Class.x, Predicted_Class.y) %>% distinct()
