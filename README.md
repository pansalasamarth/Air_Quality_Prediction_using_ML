
# Air Quality Prediction Using Machine Learning Models

This repository contains the Python code implementation of the research paper: **"Air Quality Prediction by Machine Learning Models: A Predictive Study on the Indian Coastal City of Visakhapatnam"** by Gokulan Ravindiran et al. (2023). The study uses advanced machine learning techniques to predict the Air Quality Index (AQI) based on historical data.

## Overview

Air pollution is a significant global challenge, impacting both human health and the environment. This project leverages machine learning models to predict AQI levels using air pollutant and meteorological data. The implementation includes:

-   Data preprocessing and transformation
-   Exploratory Data Analysis (EDA)
-   Model training and evaluation using:
    -   LightGBM
    -   Random Forest
    -   CatBoost
    -   AdaBoost
    -   XGBoost
-   Visualization of model performance and feature importance

## Features

-   Handles missing and non-numeric values
-   Performs data transformation for skewness and kurtosis normalization
-   Predicts AQI with high accuracy using optimized machine learning models
-   Generates feature importance and comparison metrics for the models
-   Visualizes AQI trends and pollutant contributions

## Dataset

The dataset used in this implementation is based on the Central Pollution Control Board (CPCB) data from July 2017 to September 2022. It includes:

-   **12 Air Pollutants:** PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, Toluene, Xylene
-   **10 Meteorological Factors:** Temperature, Relative Humidity, Wind Speed, Wind Direction, Solar Radiation, Air Pressure, Ambient Temperature, Rainfall, and Total Rainfall

## Requirements

The implementation requires the following Python libraries:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `seaborn`
-   `scikit-learn`
-   `lightgbm`
-   `xgboost`
-   `catboost`

Install all dependencies using:


```bash
pip install -r requirements.txt 
```
## Code Structure

-   **Data Preprocessing:** Handles missing and non-numeric values, normalizes skewed data, and prepares features for modeling.
-   **EDA:** Analyzes correlations between pollutants and AQI, visualizes monthly and annual pollutant variations.
-   **Model Training and Evaluation:** Implements and compares the performance of five machine learning models.
-   **Prediction:** Uses trained models to predict AQI and categorize its health impact.

## Results

### Model Performance Comparison

<table id="T_67279">
  <caption>Machine learning Models with their performance factors in prediction of AQI</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_67279_level0_col0" class="col_heading level0 col0" >Model_Training</th>
      <th id="T_67279_level0_col1" class="col_heading level0 col1" >MAE_Training</th>
      <th id="T_67279_level0_col2" class="col_heading level0 col2" >MSE_Training</th>
      <th id="T_67279_level0_col3" class="col_heading level0 col3" >RMSE_Training</th>
      <th id="T_67279_level0_col4" class="col_heading level0 col4" >R2_Training</th>
      <th id="T_67279_level0_col5" class="col_heading level0 col5" >MAE_Testing</th>
      <th id="T_67279_level0_col6" class="col_heading level0 col6" >MSE_Testing</th>
      <th id="T_67279_level0_col7" class="col_heading level0 col7" >RMSE_Testing</th>
      <th id="T_67279_level0_col8" class="col_heading level0 col8" >R2_Testing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_67279_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_67279_row0_col0" class="data row0 col0" >LightGBM</td>
      <td id="T_67279_row0_col1" class="data row0 col1" >1.373889</td>
      <td id="T_67279_row0_col2" class="data row0 col2" >15.846370</td>
      <td id="T_67279_row0_col3" class="data row0 col3" >3.980750</td>
      <td id="T_67279_row0_col4" class="data row0 col4" >0.995221</td>
      <td id="T_67279_row0_col5" class="data row0 col5" >1.811602</td>
      <td id="T_67279_row0_col6" class="data row0 col6" >19.478235</td>
      <td id="T_67279_row0_col7" class="data row0 col7" >4.413415</td>
      <td id="T_67279_row0_col8" class="data row0 col8" >0.992536</td>
    </tr>
    <tr>
      <th id="T_67279_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_67279_row1_col0" class="data row1 col0" >RandomForest</td>
      <td id="T_67279_row1_col1" class="data row1 col1" >0.444116</td>
      <td id="T_67279_row1_col2" class="data row1 col2" >3.171609</td>
      <td id="T_67279_row1_col3" class="data row1 col3" >1.780901</td>
      <td id="T_67279_row1_col4" class="data row1 col4" >0.999043</td>
      <td id="T_67279_row1_col5" class="data row1 col5" >1.279939</td>
      <td id="T_67279_row1_col6" class="data row1 col6" >20.688324</td>
      <td id="T_67279_row1_col7" class="data row1 col7" >4.548442</td>
      <td id="T_67279_row1_col8" class="data row1 col8" >0.992072</td>
    </tr>
    <tr>
      <th id="T_67279_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_67279_row2_col0" class="data row2 col0" >CatBoost</td>
      <td id="T_67279_row2_col1" class="data row2 col1" >1.373889</td>
      <td id="T_67279_row2_col2" class="data row2 col2" >15.846370</td>
      <td id="T_67279_row2_col3" class="data row2 col3" >3.980750</td>
      <td id="T_67279_row2_col4" class="data row2 col4" >0.995221</td>
      <td id="T_67279_row2_col5" class="data row2 col5" >1.811602</td>
      <td id="T_67279_row2_col6" class="data row2 col6" >19.478235</td>
      <td id="T_67279_row2_col7" class="data row2 col7" >4.413415</td>
      <td id="T_67279_row2_col8" class="data row2 col8" >0.992536</td>
    </tr>
    <tr>
      <th id="T_67279_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_67279_row3_col0" class="data row3 col0" >AdaBoost</td>
      <td id="T_67279_row3_col1" class="data row3 col1" >1.373889</td>
      <td id="T_67279_row3_col2" class="data row3 col2" >15.846370</td>
      <td id="T_67279_row3_col3" class="data row3 col3" >3.980750</td>
      <td id="T_67279_row3_col4" class="data row3 col4" >0.995221</td>
      <td id="T_67279_row3_col5" class="data row3 col5" >1.811602</td>
      <td id="T_67279_row3_col6" class="data row3 col6" >19.478235</td>
      <td id="T_67279_row3_col7" class="data row3 col7" >4.413415</td>
      <td id="T_67279_row3_col8" class="data row3 col8" >0.992536</td>
    </tr>
    <tr>
      <th id="T_67279_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_67279_row4_col0" class="data row4 col0" >XGBoost</td>
      <td id="T_67279_row4_col1" class="data row4 col1" >0.439370</td>
      <td id="T_67279_row4_col2" class="data row4 col2" >0.635832</td>
      <td id="T_67279_row4_col3" class="data row4 col3" >0.797391</td>
      <td id="T_67279_row4_col4" class="data row4 col4" >0.999808</td>
      <td id="T_67279_row4_col5" class="data row4 col5" >1.623464</td>
      <td id="T_67279_row4_col6" class="data row4 col6" >19.362135</td>
      <td id="T_67279_row4_col7" class="data row4 col7" >4.400243</td>
      <td id="T_67279_row4_col8" class="data row4 col8" >0.992580</td>
    </tr>
  </tbody>
</table>

The **CatBoost** model achieved the highest accuracy with an R² of 0.9998.

### Feature Importance

Key contributors to AQI prediction:

-   **PM2.5**
-   **PM10**
-   **NO2**
-   **CO**
-   **NOx**

## Visualization

The repository includes scripts to visualize:

-   Correlation matrices
-   Feature importance
-   AQI trends (monthly and annual)

## References

1.  Gokulan Ravindiran, Gasim Hayder, Karthick Kanagarathinam, Avinash Alagumalai, Christian Sonne. "Air Quality Prediction by Machine Learning Models: A Predictive Study on the Indian Coastal City of Visakhapatnam." _Chemosphere_, 2023. [DOI: 10.1016/j.chemosphere.2023.139518] (https://doi.org/10.1016/j.chemosphere.2023.139518)
    
2.  Central Pollution Control Board (CPCB), India.

## Acknowledgments

Special thanks to the authors of the research paper and the organizations involved for providing the foundational dataset and methodologies.
