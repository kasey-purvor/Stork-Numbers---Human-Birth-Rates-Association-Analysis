## Stork Numbers & Population Analysis
[See Full Report](report.md)

This repository contains a Jupyter Notebook exploring the relationship between stork populations and human birth rates across various European countries. The analysis critically examines the often-cited correlation between these variables, using statistical modeling and the bootstrap technique to evaluate different predictors of birth rates and assess the validity of the "storks deliver babies" hypothesis.

**Project Objectives:**

* Construct and compare two regression models to explain variability in birth rates:
    * **Model 1:**  Predicting birth rates based solely on stork populations (as in the original paper).
    * **Model 2:** Predicting birth rates based on population size, GDP per capita, and population density. 
* Utilise the bootstrap technique to estimate confidence intervals for model performance metrics, specifically Root Mean Squared Error (RMSE). 
* Evaluate model fit, predictive accuracy, and residual characteristics through visual and statistical analysis. 
* Discuss the implications of the findings, emphasizing the distinction between correlation and causation in statistical analysis.

**Key Findings:**

* Model 2, incorporating population size, GDP per capita, and population density as predictors, significantly outperforms Model 1 in explaining birth rate variability.
* Model 2 exhibits a substantially higher R-squared value (0.77) and a lower RMSE (189.26) compared to Model 1 (R-squared = 0.38, RMSE = 312.03).
* Bootstrap analysis reveals narrower confidence intervals for Model 2's RMSE, indicating greater predictive reliability.
* Residual analysis suggests that while both models exhibit deviations from normality and some heteroscedasticity, these issues are less pronounced in Model 2.
* Although Model 2 identifies a strong relationship between the chosen predictors and birth rates, the study reinforces the crucial distinction between correlation and causation. The findings highlight the need for cautious interpretation of statistical results and consideration of potential confounding factors.

**Repository Contents:**

* `Notebook.ipynb`: The Jupyter Notebook containing the complete analysis, including data loading, model building, bootstrap implementation, results visualization, and discussion. 
* `README.md`: An overview of the project, its objectives, and key findings.
* `report.md`: The full report in markdown format for viewing in github.

**Instructions for Running the Notebook:**

1. Ensure you have the necessary Python libraries installed, including pandas, NumPy, scikit-learn, matplotlib, and seaborn.
2. Clone the repository to your local machine.
3. Open the `ECS7024P_Coursework_3.ipynb` file in a Jupyter Notebook environment.
4. Execute the cells sequentially to reproduce the analysis and visualizations.

This project serves as a demonstration of statistical modeling techniques and critical evaluation in data analysis, emphasizing the importance of considering alternative explanations and the limitations of correlational findings. 

