import math, pylab
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# create a data frame object from csv file
df = pd.read_csv("/Users/salah/Documents/ecommerce-linear-regression/data/ecomerce.csv")

# first 5 rows of data
df.head()

# since no null data, dataset did not need to be cleaned
# summary stats about the data
df.describe()
df.info()

# scatterplot each numerical predictor against each other
sns.pairplot(data=df, kind="scatter", plot_kws={"alpha": 0.5})

# fit linear regression line
sns.lmplot(x="Length of Membership", y="Yearly Amount Spent", data=df, scatter_kws={"alpha": 0.5})

# create sub-data frames that includes the predictors and the response for the model
X = df[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = df["Yearly Amount Spent"]

# split data in train and test sets
# random state indicates that particular shuffle of split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train linear model
lm = LinearRegression()
lm.fit(X_train, y_train)

# create dataframe to neatly show coef of predictors
coef_df = pd.DataFrame(index=X.columns, columns=["Coef"], data=lm.coef_)
print("Intercept", lm.intercept_)
print(coef_df)

# plot actual vs predicted
predictions = lm.predict(X_test)

sns.scatterplot(x=predictions, y=y_test, alpha=0.5)
plt.xlabel("Predictions")

# strength of regression
mab = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)

print("MAB", mab)
print("MSE", mse)
print("RMSE", rmse)

# residual analysis
residuals = y_test - predictions
sns.displot(residuals, kde=True)

# QQ plot
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()