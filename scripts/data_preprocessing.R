library(dplyr)
library(lubridate)
library(zoo)

cat("Loading raw data...\n")

df <- read.csv("C:/Users/palak/OneDrive/Desktop/Stock-analysis/data/sp500_processed.csv")
df$date <- as.Date(df$date)

df <- df %>%
  arrange(date)

cat(paste("Rows loaded  :", nrow(df), "\n"))
cat(paste("Date range   :", min(df$date), "→", max(df$date), "\n"))
cat(paste("Missing (raw):", sum(is.na(df$price)), "\n"))


cat("\n--- Step 1: Handling Missing Values ---\n")

df$price <- na.locf(df$price, na.rm = FALSE) # forward fill
df <- df %>% filter(!is.na(price))

cat(paste("Missing after fill:", sum(is.na(df$price)), "\n"))
cat(paste("Rows after cleaning:", nrow(df), "\n"))


cat("\n--- Step 2: Time Features ---\n")

df <- df %>%
  mutate(
    year = year(date),
    month = month(date),
    month_name = format(date, "%b"),
    quarter = quarter(date),
    week = isoweek(date),
    weekday = weekdays(date),
    decade = (year %/% 10) * 10
  )


cat("--- Step 3: Daily Returns ---\n")

df <- df %>%
  mutate(
    daily_return = (price / lag(price) - 1) * 100,
    log_return = log(price / lag(price)) * 100,
    price_change = price - lag(price),
    direction = ifelse(daily_return >= 0, "Up", "Down")
  )


cat("--- Step 4: Moving Averages ---\n")

df <- df %>%
  mutate(
    ma_7 = rollmean(price, 7, fill = NA, align = "right"),
    ma_30 = rollmean(price, 30, fill = NA, align = "right"),
    ma_90 = rollmean(price, 90, fill = NA, align = "right"),
    ma_200 = rollmean(price, 200, fill = NA, align = "right")
  )

# Handle min_periods = 1 equivalent
df$ma_7[is.na(df$ma_7)] <- df$price[is.na(df$ma_7)]
df$ma_30[is.na(df$ma_30)] <- df$price[is.na(df$ma_30)]
df$ma_90[is.na(df$ma_90)] <- df$price[is.na(df$ma_90)]
df$ma_200[is.na(df$ma_200)] <- df$price[is.na(df$ma_200)]

df$golden_cross <- ifelse(df$ma_30 > df$ma_200, 1, 0)


cat("--- Step 5: Volatility ---\n")

df <- df %>%
  mutate(
    volatility_7 = rollapply(daily_return, 7, sd, fill = NA, align = "right"),
    volatility_30 = rollapply(daily_return, 30, sd, fill = NA, align = "right"),
    volatility_90 = rollapply(daily_return, 90, sd, fill = NA, align = "right")
  )


cat("--- Step 6: Bollinger Bands ---\n")

df <- df %>%
  mutate(
    bb_mid = rollmean(price, 20, fill = NA, align = "right"),
    bb_std = rollapply(price, 20, sd, fill = NA, align = "right"),
    bb_upper = bb_mid + 2 * bb_std,
    bb_lower = bb_mid - 2 * bb_std,
    bb_width = bb_upper - bb_lower
  )

df$cumulative_return <- (df$price / df$price[1] - 1) * 100


returns <- na.omit(df$daily_return)

cat("\n--- Summary Statistics ---\n")
cat(paste("Mean daily return   :", round(mean(returns), 4), "%\n"))
cat(paste("Median daily return :", round(median(returns), 4), "%\n"))
cat(paste("Std dev of returns  :", round(sd(returns), 4), "%\n"))
cat(paste("Best single day     :", round(max(returns), 4), "%\n"))
cat(paste("Worst single day    :", round(min(returns), 4), "%\n"))
cat(paste("Positive days       :", sum(returns > 0), "\n"))
cat(paste("Negative days       :", sum(returns < 0), "\n"))

cat(paste("\nFull DataFrame shape:", dim(df)[1], dim(df)[2], "\n"))
cat(paste("Columns:", paste(colnames(df), collapse = ", "), "\n"))


write.csv(df, "C:/Users/palak/OneDrive/Desktop/Stock-analysis/data/sp500_processed.csv", row.names = FALSE)

cat("\nProcessed data saved to: data/sp500_processed.csv\n")
cat("Preprocessing complete. Run 03_eda.R next.\n")
