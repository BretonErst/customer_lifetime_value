# libraries ---- 
library(tidyverse)
library(tidymodels)
library(styleBreton)


# data acquisition ----
dt_00 <- 
  read_csv("data/cdnow.csv") |> 
  select(-1)


dt_00 |> 
  map_df(\(x) sum(is.na(x)))

dt_00 |> 
  distinct(customer_id) |> 
  nrow()

# cohort analysis ----
# customers that joined at the specific business day

# first purchase of each customer
first_purchase <- 
  dt_00 |> 
  group_by(customer_id) |> 
  arrange(customer_id, date) |> 
  slice_head(n = 1) |> 
  ungroup()

# date of first purchase  
first_purchase |> 
  (\(x) min(x$date))()

# date of last first purchase  
first_purchase |> 
  (\(x) max(x$date))()


# month visualization
dt_00 |> 
  group_by(month = floor_date(date, "month")) |> 
  summarize(monthly_sale = sum(price)) |> 
  ggplot(aes(x = month, y = monthly_sale)) +
  geom_line(linewidth = 1,
            color = "steelblue") +
  labs(title = "Sales by Month",
       subtitle = "Aggregated amount in thounsands",
       y = "Amount (000)",
       x = "Date",
       caption = "Juan L.Bretón, PMP") +
  theme_breton() +
  scale_y_continuous(labels = scales::dollar_format(scale = 1/1000)) +
  scale_x_date(date_breaks = "1 month",
               date_labels = "%m\n%y")


# purchasing behavior

# sample of customers
set.seed(3342)
cust_samp <- 
  # sample(dt_00$customer_id, 10)
  c(1:10)

dt_00 |> 
  filter(customer_id %in% cust_samp) |> 
  group_by(customer_id) |> 
  arrange(customer_id, date) |> 
  ggplot(aes(x = date, y = price, group = customer_id)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue") +
  facet_wrap(~ customer_id) +
  theme_breton() +
  labs(title = "Purchasing Behavior",
       subtitle = "Purchases by customer",
       y = "Amount",
       x = "Date",
       caption = "Juan L. Bretón, PMP") +
  scale_y_continuous(labels = scales::dollar_format()) +
  scale_x_date(date_breaks = "3 month",
               date_labels = "%m\n%y")


# feature engineering ----

# purchasing period
num_days <- 90

# cutoff day
cutoff <- 
  dt_00 |> 
  (\(x) max(x$date))() - 90

# in-dataset
dt_in <- 
  dt_00 |> 
  filter(date <= cutoff)

# after-dataset
dt_out <- 
  dt_00 |> 
  filter(date > cutoff)


# feature engineering

# target
dt_targets <- 
  dt_out |> 
  group_by(customer_id) |> 
  summarize(spend_90_total = sum(price)) |> 
  mutate(spend_90_flag = 1)

# recency
dt_recency <- 
  dt_in |> 
  group_by(customer_id) |> 
  slice_max(order_by = date, n = 1, with_ties = FALSE) |> 
  ungroup() |> 
  mutate(recency = as.numeric(date - dt_00 |> (\(x) max(x$date))())) |> 
  select(customer_id, recency)

# frequency
dt_frequency <- 
  dt_in |> 
  group_by(customer_id) |> 
  count() |> 
  ungroup() |> 
  select(customer_id, frequency = n)

# price
dt_price <- 
  dt_in |> 
  group_by(customer_id) |> 
  summarize(price_sum = sum(price),
            price_mean = mean(price),
            .groups = "drop")

# dataset of features
dt_features <- 
  dt_recency |> 
  add_column(dt_frequency |> select(frequency)) |> 
  add_column(dt_price |> select(starts_with("price")))


# join to targets
dt_01 <- 
  dt_features |> 
  left_join(dt_targets, 
            by = join_by(customer_id)) |> 
  replace_na(list(spend_90_total = 0,
                  spend_90_flag = 0)) 


# regression ----
# what will customers spend in the next 90-day period? --regression

## train / test split
set.seed(443)
clv_spend_split <- 
  initial_split(dt_01, strata = spend_90_total)

clv_spend_train <- 
  clv_spend_split |> 
  training()

clv_spend_test <- 
  clv_spend_split |> 
  testing()

# recipe
clv_spend_recipe <- 
  recipe(spend_90_total ~ ., data = clv_spend_train) |> 
  update_role(customer_id, new_role = "id") |> 
  update_role(spend_90_flag, new_role = "intent")

# cross validation
set.seed(332)
clv_spend_folds <- 
  vfold_cv(clv_spend_train, 
           strata = spend_90_total)


# xgboost spec
clv_spend_xgb_spec <- 
  boost_tree(trees = tune(),
             min_n = tune(),
             mtry = tune(),
             learn_rate = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

# workflow
clv_spend_wf <- 
  workflow() |> 
  add_recipe(clv_spend_recipe) |> 
  add_model(clv_spend_xgb_spec)


# tunning
clv_spend_tuned <- 
  tune_grid(object = clv_spend_wf,
            resamples = clv_spend_folds,
            grid = 10)

# best model
show_best(clv_spend_tuned, metric = "rmse")

# final fit
clv_spend_fit <- 
  clv_spend_wf |> 
  finalize_workflow(select_best(clv_spend_tuned, "rmse")) |> 
  last_fit(clv_spend_split)

clv_spend_fit |> 
  collect_metrics()

# extract trained model
clv_spend_model <- 
  extract_workflow(clv_spend_fit)

# prediction
new <- 
  dt_01 |> 
  slice_sample(n = 5)

predict(clv_spend_model, new_data = new)

# importance
clv_spend_fit |> 
  extract_fit_parsnip() |> 
  vip::vip()



# classification ----
# probability of a customer to make a purchase in the next 90-d period

# convert target to factor
dt_02 <- 
  dt_01 |> 
  mutate(spend_90_flag = as_factor(spend_90_flag))


## train / test split
set.seed(423)
clv_class_split <- 
  initial_split(dt_02, 
                strata = spend_90_flag)

clv_class_train <- 
  clv_class_split |> 
  training()

clv_class_test <- 
  clv_class_split |> 
  testing()


# recipe
clv_class_recipe <- 
  recipe(spend_90_flag ~ ., 
         data = clv_class_train) |> 
  update_role(customer_id, new_role = "id") |> 
  update_role(spend_90_total, new_role = "intent")


# cross validation
set.seed(232)
clv_class_folds <- 
  vfold_cv(clv_class_train, 
           strata = spend_90_flag)


# xgboost spec
clv_class_xgb_spec <- 
  boost_tree(trees = tune(),
             min_n = tune(),
             mtry = tune(),
             learn_rate = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

# workflow
clv_class_wf <- 
  workflow() |> 
  add_recipe(clv_class_recipe) |> 
  add_model(clv_class_xgb_spec)


# tunning
clv_class_tuned <- 
  tune_grid(object = clv_class_wf,
            resamples = clv_class_folds,
            grid = 10)

# best model
show_best(clv_class_tuned, metric = "roc_auc")

# final fit
clv_class_fit <- 
  clv_class_wf |> 
  finalize_workflow(select_best(clv_class_tuned, "roc_auc")) |> 
  last_fit(clv_class_split)

clv_class_fit |> 
  collect_metrics()

# extract trained model
clv_class_model <- 
  extract_workflow(clv_class_fit)

# prediction
new <- 
  dt_01 |> 
  slice_sample(n = 5)

predict(clv_class_model, new_data = new, type = "prob")


# importance
clv_class_fit |> 
  extract_fit_parsnip() |> 
  vip::vip()



# prep for production ----

# prediction
dt_predictions <- 
  predict(clv_spend_model, 
        new_data = dt_01) |> 
  add_column(predict(clv_class_model, 
                     new_data = dt_01, 
                     type = "prob")) |> 
  add_column(dt_01) |> 
  relocate(customer_id, 
           .before = everything()) |> 
  select(customer_id, 
         pred_amount = .pred,
         prob_purchase = .pred_1,
         recency, frequency, price_sum, price_mean, 
         spend_90_total, spend_90_flag)


dt_predictions |> 
  ggplot(aes(x = frequency, y = prob_purchase)) +
  geom_point(alpha = 0.25)


# customers with highest probability to spend in the next 90-day period
dt_predictions |> 
  arrange(desc(prob_purchase))


# customers that recently purchased but unlikely to buy
dt_predictions |> 
  filter(recency > -120 & prob_purchase < 0.4) |> 
  arrange(prob_purchase) 

dt_predictions |> 
  summary(prob_purchase)

# missed oportunities
dt_predictions |> 
  filter(spend_90_total == 0) |> 
  arrange(-prob_purchase)




























  
























