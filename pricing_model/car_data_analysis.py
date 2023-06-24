import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment', None)

car_data = pd.read_csv('car_prices.csv', on_bad_lines='skip').head(50000)

index = 'vin'
price = 'sellingprice'

car_data = car_data.dropna(how='any')
car_data = car_data.drop_duplicates()
car_data = car_data[car_data['year'] > 2000]

prices = car_data[price].astype(np.float64)
avg_price = car_data[price].mean()
std_price = car_data[price].std()

'''
Method that reveals whether or not the type of a column is numeric
True = numeric (int64 or float64)
False = not numeric
'''
def is_numeric(df, i):
    return df.iloc[0:1].applymap(np.isreal).iloc[0][i]

'''
Returns standardized matrix for car_data, which will provide numeric representations for categorical variables
'''
def standardize_training_set(df):
    standardized_car_data = car_data.copy()
    for column in df.drop(columns=[price, index]):
        if not is_numeric(df, column):
            df[column] = df[column].str.strip()
            df[column] = df[column].str.lower()
            unique = df[column].unique()
            for i in unique:
                projection_i = df[[column, price]][df[column] == i]
                standardized_car_data[column][df[column] == i] = (projection_i[price].mean() - avg_price) / std_price
        else:
            standardized_car_data[column] = (df[column] - df[column].mean()) / df[column].std()

    return standardized_car_data

'''
Returns the standardized matrix accounting for x_0, which is used in normal equation for multivariate linear regression
'''
def get_X(standardized_car_data):
    X = standardized_car_data.copy()
    X = X.drop(columns=[price, index])
    x_bias = np.ones((car_data.shape[0], 1))
    X.insert(0, "ones", x_bias)
    X = X.astype(np.float64)
    return X

def get_theta(X, Y):
    print(X)
    print(X.T)
    print(Y)
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

'''
Standardizes an input vector using the same method that was used on the training data
'''
def standardize(series, df, standardized_df):
    series.index = df.drop(columns=[index, price]).columns
    standardized_series = series.copy()
    for i in series.index:
        if(series[i] == None):
            standardized_series[i] = standardized_df[i].mean()
            continue
        if is_numeric(df, i):
            standardized_series[i] = (series[i] - df[i].mean()) / df[i].std()
        else:
            series[i] = series[i].lower()
            selection_i = df[df[i] == series[i]]
            if(selection_i.empty):
                standardized_series[i] = standardized_df[i].mean()
            else:
                df_index = selection_i[index].iloc[0]
                standardized_value = standardized_df[standardized_df[index] == df_index][i].iloc[0]
                standardized_series.loc[i] = standardized_value
    return np.insert(standardized_series, 0, 1)

'''
Predicts the price of a given car
'''
def predict_price(series, df, standardized_df, theta):
    return np.dot(theta.T, standardize(series, df, standardized_df))

trained_data = standardize_training_set(car_data)
X = get_X(trained_data)
theta = get_theta(X, prices)

print(predict_price(pd.Series([2010,"Nissan","Altima",None,"Sedan",None,
                       "ca",3.9,100000, "black", "black", None ,None , None]),
                       car_data, trained_data, theta))


print(theta)