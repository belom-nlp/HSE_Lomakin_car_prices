# HSE_Lomakin_car_prices
For HSE summer 2023 bootcamp
Here I use some basic machine learning methods to predict car prices by their features. Dataset EDA cars is provided by Higher School of Economics, you can find its description in the code available.
Basically, the work was organized as follows:
1) Data preprocessing: a) year was transformed into age, thus making calculations simpler; b) milage was tranformed from string into float and unified (kmpl was turned into km/kg); c) engine and mileage were transformed from string into float; d) torque was divided into three seperate columns: torque itself and minimum and maximum rpm.
2) Data normalization: I used MinMaxScaler and StandardScaler in order to find out which one will work better.
3) Model seceltion: I imported several models from Scikit learn and, using mean squared error metric, found out that ExtraTreesRegressor with StandardScaler works best.
4) I used Optuna to fine-tune the model.

 All these steps are described in detail (in Russian) in the code page.
