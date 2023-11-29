# Librerías ---- 
library(tidyverse)
library(tidymodels)
library(styleBreton)


# Adquisición de Datos ----
dt_00 <- 
  read_csv("data/cdnow.csv") |> 
  select(-1)

# verificación de NAs
dt_00 |> 
  map_df(\(x) sum(is.na(x)))

# número de clientes únicos
dt_00 |> 
  distinct(customer_id) |> 
  nrow()


# Análisis de compra ----
# customers that joined at the specific business day

# primera compra de cada cliente
first_purchase <- 
  dt_00 |> 
  arrange(customer_id, date) |> 
  group_by(customer_id) |>
  slice_head(n = 1) |> 
  ungroup()

# fecha de primera compra 
first_purchase |> 
  (\(x) min(x$date))()

# fecha de la primera compra más tardía  
first_purchase |> 
  (\(x) max(x$date))()


# visualización de compras por mes
dt_00 |> 
  group_by(month = floor_date(date, "month")) |> 
  summarize(monthly_sale = sum(price)) |> 
  ggplot(aes(x = month, y = monthly_sale)) +
  geom_line(linewidth = 1,
            color = "steelblue") +
  labs(title = "Compras Por Mes",
       subtitle = "Cantidad agregada (en miles)",
       y = "Cantidad (000)",
       x = "Mes",
       caption = "Fuente: CDNOW data <br>
       Modelaje y visualización: Juan L.Bretón, PMP") +
  theme_breton() +
  scale_y_continuous(labels = scales::dollar_format(scale = 1/1000)) +
  scale_x_date(date_breaks = "1 month",
               date_labels = "%m\n%y")


# comportamiento individual de compra
# muestra de clientes
set.seed(3352)
cust_samp <- 
  sample(dt_00$customer_id, 12)
  # c(1:10)

# visualización del comportamiento individual
# acumula precio por cliente y fecha
dt_00 |> 
  filter(customer_id %in% cust_samp) |> 
  group_by(customer_id, date) |> 
  summarize(across(.cols = c(quantity, price),
                   .fns = sum,
                   .names = "acum_{.col}"),
            .groups = "drop") |>
  ggplot(aes(x = date, y = acum_price, group = customer_id)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue") +
  facet_wrap(~ customer_id) +
  theme_breton() +
  labs(title = "Comportamiento Individual de Compra",
       subtitle = "Número de compras por cliente",
       y = "Cantidad gastada por día",
       x = "Fecha",
       caption = "Fuente: CDNOW data <br>
       Modelaje y visualización: Juan L.Bretón, PMP") +
  scale_y_continuous(labels = scales::dollar_format()) +
  scale_x_date(date_breaks = "3 month",
               date_labels = "%m\n%y")



# Ingeniería de Características ----
# los últimos 90 días serán el periodo de análisis

# longitud del perido de análisis
num_days <- 90

# fecha de corte
cutoff <- 
  dt_00 |> 
  (\(x) max(x$date))() - num_days

# data en periodo
dt_in <- 
  dt_00 |> 
  filter(date <= cutoff)

# data después de periodo
dt_out <- 
  dt_00 |> 
  filter(date > cutoff)


# generación de características

# respuestas
# clientes que hicieron una compra en los últimos 90 días
# solo queda una proporción pequeña de todos los clientes

# dataset de objetivos
dt_targets <- 
  dt_out |> 
  group_by(customer_id) |> 
  summarize(spend_90_total = sum(price),
            .groups = "drop") |> 
  mutate(spend_90_flag = 1)


# predictores
# clientes que no hicieron compra en los últimos 90 días

# dataset de compra reciente
# hace cuántos días fue su compra más reciente
dt_recency <- 
  dt_in |> 
  group_by(customer_id) |> 
  slice_max(order_by = date, n = 1, with_ties = FALSE) |> 
  ungroup() |> 
  mutate(recency = as.numeric(date - dt_in |> (\(x) max(x$date))())) |> 
  select(customer_id, recency)


# dataset de frecuencia de compras
# cuenta de compras por cliente
dt_frequency <- 
  dt_in |> 
  group_by(customer_id) |> 
  count() |> 
  ungroup() |> 
  select(customer_id, frequency = n)


# dataset de cantidad gastada
# price
dt_price <- 
  dt_in |> 
  group_by(customer_id) |> 
  summarize(across(.cols = price,
                   .fns = list(sum = sum, 
                               mean = mean),
                   .names = "{.col}_{.fn}"))

# integración de dataset de features
dt_features <- 
  dt_recency |> 
  add_column(dt_frequency |> select(frequency)) |> 
  add_column(dt_price |> select(starts_with("price")))


# integración con dataset de targets
dt_01 <- 
  dt_features |> 
  left_join(dt_targets, 
            by = join_by(customer_id)) |> 
  replace_na(list(spend_90_total = 0,
                  spend_90_flag = 0)) 


# Regresión Machine Learning ----
# cuánto gastarán los clientes en el siguiente periodo de 90 días?

## train / test split
set.seed(443)
clv_spend_split <- 
  initial_split(dt_01, 
                strata = spend_90_total)

# dataset de entrenamiento
clv_spend_train <- 
  clv_spend_split |> 
  training()

# dataset de prueba
clv_spend_test <- 
  clv_spend_split |> 
  testing()

# receta de preprocesamiento
clv_spend_recipe <- 
  recipe(spend_90_total ~ ., data = clv_spend_train) |> 
  update_role(customer_id, new_role = "id") |> 
  update_role(spend_90_flag, new_role = "intent") |> 
  step_normalize(all_numeric_predictors())


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

# metricas del final fit
clv_spend_fit |> 
  collect_metrics()

# extracción del modelo entrenado
clv_spend_model <- 
  extract_workflow(clv_spend_fit)

# predicción con nuevos datos
new <- 
  dt_01 |> 
  slice_sample(n = 5)

predict(clv_spend_model, new_data = new)


# importancia de variables
clv_spend_fit |> 
  extract_fit_parsnip() |> 
  vip::vi() |> 
  ggplot(aes(x = Importance, 
             y = fct_reorder(Variable, Importance))) +
  geom_col(alpha = 0.65, 
           fill = "darkgreen") +
  labs(title = "Variables Relacionadas con la Cantidad Gastada",
       subtitle = "Modelo XG Boost Regresión",
       x = "Importancia",
       y = NULL,
       caption = "Fuente: CDNOW data <br>
       Modelaje y visualización: Juan L.Bretón, PMP") +
  theme_breton()



# Clasificación Machine Learning ----
# probabilidad de que un cliente haga una compra en el siguiente periodo

# convertir la variable de respuesta a factor
dt_02 <- 
  dt_01 |> 
  mutate(spend_90_flag = as_factor(spend_90_flag))


## train / test split
set.seed(423)
clv_class_split <- 
  initial_split(dt_02, 
                strata = spend_90_flag)

# dataset de entrenamiento
clv_class_train <- 
  clv_class_split |> 
  training()

# dataset de prueba
clv_class_test <- 
  clv_class_split |> 
  testing()


# receta de preprocesamiento
clv_class_recipe <- 
  recipe(spend_90_flag ~ ., 
         data = clv_class_train) |> 
  update_role(customer_id, new_role = "id") |> 
  update_role(spend_90_total, new_role = "intent") |> 
  step_normalize(all_numeric_predictors())


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

# métricas del final fit
clv_class_fit |> 
  collect_metrics()

# extracción del modelo entrenado
clv_class_model <- 
  extract_workflow(clv_class_fit)

# predicción en nuevos datos
new <- 
  dt_01 |> 
  slice_sample(n = 5)

predict(clv_class_model, new_data = new, type = "prob")


# importance
clv_class_fit |> 
  extract_fit_parsnip() |> 
  vip::vi() |> 
  ggplot(aes(x = Importance, 
             y = fct_reorder(Variable, Importance))) +
  geom_col(alpha = 0.65, 
           fill = "darkgreen") +
  labs(title = "Variables Relacionadas con la Probabilidad de Comprar",
       subtitle = "Modelo XG Boost Clasificación",
       x = "Importancia",
       y = NULL,
       caption = "Fuente: CDNOW data <br>
       Modelaje y visualización: Juan L.Bretón, PMP") +
  theme_breton()



# Preparación para Producción ----

# predicción en todos los clientes
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
         spend_90_total, spend_90_flag) |> 
  mutate(difer = spend_90_total - pred_amount)


# visualización de la predicción de todos los clientes
dt_predictions |> 
  ggplot(aes(x = frequency, 
             y = prob_purchase, 
             color = difer)) +
  geom_point(alpha = 0.25)


# clientes con la más alta probabilidad de comprar en los
# siguientes 90 días
dt_predictions |> 
  arrange(desc(prob_purchase))


# clientes que han comprado recientemente pero que tienen
# pocas probabilidades de volver a comprar
dt_predictions |> 
  filter(recency > -100 & prob_purchase < 0.4) |> 
  arrange(prob_purchase) 

dt_predictions |> 
  summary(prob_purchase)

# clientes con alta cantidad pero que no han hecho compra
dt_predictions |> 
  filter(spend_90_total == 0) |> 
  arrange(-pred_amount)




























  
























