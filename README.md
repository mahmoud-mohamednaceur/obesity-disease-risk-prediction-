# Obesity Disease Risk Prediction Web App

![2024-11-0618-28-01-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/a721f312-c09a-4021-847a-918bd2d783d7)


For the **Obesity Disease Risk Prediction Web App**, I utilized two robust machine learning models—**Random Forest** and **XGBoost**—to achieve accurate predictions of obesity risk. These models were selected for their strong performance in classification tasks, especially in scenarios involving complex, non-linear relationships between features, which is typical in health-related datasets.

### Model Development and Training

1. **Random Forest Model**:
   - **Random Forest** is an ensemble learning method that constructs multiple decision trees and aggregates their predictions, enhancing accuracy and reducing overfitting.
   - During training, the Random Forest model was fine-tuned by adjusting parameters such as the number of trees, maximum depth, and minimum samples required for a split.
   - The **Random Forest** model performed well in identifying key obesity risk factors (such as BMI, physical activity, and dietary habits) due to its capacity to handle a mix of categorical and numerical data and its robustness against overfitting.

2. **XGBoost Model**:
   - **XGBoost (Extreme Gradient Boosting)** is another ensemble technique, widely known for its efficiency and predictive power in handling large and complex datasets.
   - The XGBoost model was optimized with a grid search to adjust parameters like learning rate, max depth, and the number of estimators. Its regularization features also helped improve generalization.
   - XGBoost’s strength lies in its ability to capture complex relationships and interactions between features, making it effective for health prediction scenarios where multiple factors contribute to obesity risk.

### Model Selection and Evaluation

Both **Random Forest** and **XGBoost** were evaluated on metrics such as **accuracy, precision, recall,** and **F1-score**. Each model was trained using cross-validation to ensure reliability, and the following performance insights were observed:

- The **Random Forest** model demonstrated high interpretability, allowing easy identification of feature importance and key predictors of obesity risk. It proved to be effective in identifying moderate-risk cases with a balanced trade-off between precision and recall.
  
- The **XGBoost** model excelled in capturing subtle patterns within the data, leading to higher accuracy, especially for cases in the high-risk category. Its ability to handle complex patterns made it valuable for the final prediction stage.

Given their complementary strengths, the final deployment includes both models, with **XGBoost** used as the primary predictor due to its slightly better performance, and **Random Forest** serving as a secondary option to cross-check predictions for high-risk cases.

### Implementation in the Web App

With the **Streamlit** web app, users can input personal health information and lifestyle factors, and the app will utilize the selected model to provide an obesity risk prediction. Depending on the input, users will receive a risk level—*low*, *moderate*, or *high*—based on the prediction from the XGBoost model, with Random Forest available as a backup for cross-validation.

### Conclusion

The combination of **Random Forest** and **XGBoost** in the Obesity Disease Risk Prediction Web App provides a well-rounded, highly accurate approach to predicting obesity risk. By leveraging the interpretability of Random Forest and the precision of XGBoost, the app delivers a powerful tool for users to better understand and manage their health risks related to obesity.
