# Libraries ---------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(tidyr)


# Load Data ----------------------------------------------------------------
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
res_results <- read.csv("Files and scripts/Resnext_preds.csv",
                        col.names = c_names, check.names = FALSE)
test_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_test.csv")



# Calc Precision ----------------------------------------------------------

res_results %>%
  pivot_longer(`Arts & Photography`:Travel, names_to = 'Predicted_Class', values_to = 'Prediction_Perc') %>%
  group_by(Filename) %>%
  arrange(desc(Prediction_Perc)) %>%
  slice(1) %>% 
  left_join(test_data) %>%
  select(Filename:Category, -X) %>%
  mutate(correct = if_else(Predicted_Class == Category, 1, 0)) %>%
  group_by(Predicted_Class) %>%
  summarise(precision = sum(correct)/n())
