library(dplyr)
library(ggplot2)
library(tidyr)

train_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_train.csv")
valid_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_valid.csv")
test_data <- read.csv("Files and scripts/dims_and_colours_to_analyse_test.csv")

comb_data <- train_data %>% 
  rbind(valid_data, test_data)

pivot_data <- mutate(Aspect_Ratio = Height/Width) %>%
  pivot_longer(Height:Aspect_Ratio, names_to = 'measure')

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


comb_data %>%
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
