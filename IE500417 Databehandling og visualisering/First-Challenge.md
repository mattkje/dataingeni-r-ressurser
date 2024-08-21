# First Challenge
Author: Matti Kjellstadli

## Question
In order to make related policies, the Norwegian government asks you to predict how many COVID-19 infected cases in the next two weeks (14 days - daily) in Norway.

## Prerequisites


### What data do we need?

Before we can start creating a solution for this problem, we first have to find the necessary data to be able to predict how many COVID-19 cases there will be.
Example of data we might need:

- How many was infected the last two weeks.
- How infectious is the decease.
- Looking at other countries´ situation may also prove a benefit for our solution.

### How do we get this data?

The amount of cases is can be collected through hospitals, and also "Smitteappen" where users themselves submit their own cases.
However, this data is not clean and may contain errors. It is important to clean the data before using it in a model.
This includes eliminating outliers, missing values, and other errors.

## How should we predict?

In order to predict how many COVID-19 cases that will occur daily the next two weeks we have to rely on what we have already got.
Plotting previous data into a tool such as Python, we can create a scatter plot and add a regression line.


An example of such program provided we have a file called "theRawData.csv" with the necessary data:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
data_root = "/theRawData.csv"
casesperday = pd.read_csv(data_root)
X = casesperday[["Day"]].values
y = casesperday[["Cases"]].values

# Visualize the data
casesperday.plot(kind='scatter', grid=True, x="Day", y="Cases")
plt.show()

# Select and fit a linear model
model = LinearRegression()
model.fit(X, y)

# Make predictions for the next 14 days
future_days = np.array([[i] for i in range(X[-1][0] + 1, X[-1][0] + 15)])
predictions = model.predict(future_days)

# Plot the regression line and predictions
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.scatter(future_days, predictions, color='green')
plt.xlabel("Day")
plt.ylabel("Cases")
plt.title("COVID-19 Cases Prediction")
plt.grid(True)
plt.show()
```

### Conclusion

This solution is a simple linear regression model, and may not be the best solution for this problem.
It only takes into account the previous data, and does not consider other factors that may affect the number of cases.
However, it is a good starting point for further development.

