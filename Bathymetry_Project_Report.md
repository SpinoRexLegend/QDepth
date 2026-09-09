# Satellite-Derived Shallow-Water Bathymetry Prediction Using Classical and Quantum Machine Learning

## 1. Title & Metadata

**Project Title:** Satellite-Derived Shallow-Water Bathymetry Prediction Using Classical and Quantum Machine Learning  
**Author(s):** [Your Name / Team Members]  
**Institution / Course:** [Institution / Course / Department]  
**Date:** [Submission Date]

## 2. Problem Statement & Objectives

### 2.1 Problem Statement

Bathymetric mapping measures the depth and shape of underwater terrain. Traditional surveys are accurate but expensive, time-consuming, and difficult to apply over large or remote coastal areas.

This project investigates satellite-derived optical features for estimating shallow-water depth. It focuses on depths between 0 and 30 metres, where visible spectral information can penetrate the water column. Classical regression algorithms are compared with variational quantum regression methods.

### 2.2 Project Objectives

The objectives are to:

- Predict shallow-water depth from satellite-derived spectral features.
- Compare linear, ensemble, neural-network, kernel, boosting, and quantum regression models.
- Evaluate models using MAE, RMSE, and R-squared.
- Investigate whether separate regional models improve prediction accuracy.
- Examine the practicality of quantum regression for bathymetric prediction.

### 2.3 Task Formulation

This is a supervised multivariate regression problem. The six model input features are `blue_band`, `green_band`, `log_blue`, `log_green`, `bg_ratio`, and `stumpf_ratio`. The target variable is `depth`, measured in metres.

Latitude and longitude are used for regional partitioning but are not included as direct model features. The regions are the Arabian Sea / Gulf, Strait of Hormuz, and Bay of Bengal.

## 3. Dataset Description

### 3.1 Dataset Source

The primary file is `Dataset/final_sdb_dataset_clean.csv`. The notebooks do not document the original public dataset, satellite mission, bathymetric survey, or collection methodology. This information should be added before final submission.

An alternate file, `final_sdb_dataset_clean(2).csv`, contains 8,000 rows and five columns but is not used by the modeling notebooks. The primary nine-column dataset is used for the experiments.

### 3.2 Dataset Properties

The primary dataset contains 11,676 rows and 9 columns. All stored CSV cells are populated, and the variables are numerical. The columns are:

1. `latitude`
2. `longitude`
3. `blue_band`
4. `green_band`
5. `depth`
6. `stumpf_ratio`
7. `log_blue`
8. `log_green`
9. `bg_ratio`

The target variable is `depth`. The six model features are the engineered spectral variables listed in Section 2.3.

After restricting the target to 0-30 metres, 4,548 rows remained. The geographic distribution was:

| Region | Samples |
|---|---:|
| Arabian Sea / Gulf | 4,067 |
| Strait of Hormuz | 54 |
| Bay of Bengal | 427 |
| **Total** | **4,548** |

### 3.3 Train/Validation/Test Split

The notebooks use an 80% training and 20% testing split with `random_state=42`:

- Training set: 3,638 samples
- Test set: 910 samples
- Validation set: None
- Stratification: Not used

The same strategy is applied independently to regional subsets. Because the split is random rather than spatial, nearby observations may occur in both training and testing sets.

## 4. Exploratory Data Analysis

### 4.1 Descriptive Statistics

Reported shallow-water statistics are:

| Variable | Mean | Standard Deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Latitude | 25.5682 | 3.8950 | 6.0438 | 30.4396 |
| Longitude | 54.3719 | 8.5611 | 48.0021 | 89.9604 |
| Blue band | 0.0702 | 0.0355 | Not reported | Not reported |
| Green band | 0.0714 | 0.0475 | Not reported | Not reported |
| Depth | 13.6906 | 8.8719 | 1 | 30 |
| Stumpf ratio | 0.9911 | 0.0443 | 0.8321 | 1.1743 |

Median and complete quartile statistics were not consistently recorded.

### 4.2 Data Distribution Analysis

The filtered target ranges from 1 to 30 metres, with a mean of approximately 13.69 metres and a standard deviation of approximately 8.87 metres.

The dataset is geographically unbalanced. The Arabian Sea / Gulf contains most observations, whereas the Strait of Hormuz contains only 54. This makes regional comparisons, especially for the Strait of Hormuz, less statistically reliable.

### 4.3 Missing Values & Outliers

The stored primary CSV contains no blank cells. However, one parser output showed only 482 non-null values for `stumpf_ratio`, while the quantum encoding notebook explicitly replaces infinite values with `NaN` and drops incomplete rows. The classical notebooks do not clearly document equivalent missing-value handling.

No formal outlier method, such as the interquartile-range rule or robust z-scores, was implemented.

### 4.4 Correlation & Relationships

A complete correlation matrix and numerical feature-to-target correlation analysis were not included. The engineered spectral ratios and logarithmic features are intended to capture the relationship between water depth and optical reflectance.

Regional scatter plots show three geographically separated areas. Differences in water clarity, bottom type, environmental conditions, and sensor characteristics may explain why model accuracy varies by region.

### 4.5 Visualizations

The notebooks include:

- Regional longitude-latitude scatter plots.
- Actual-versus-predicted depth plots.
- Ideal 1:1 prediction lines.
- A training-loss plot for the second quantum model.
- A test-set actual-versus-predicted plot.

A correlation heatmap, feature histograms, missing-value plot, and systematic outlier plots should be added in future work.

## 5. Data Cleaning & Preprocessing

### 5.1 Missing Value & Outlier Handling

The data was filtered using:

$$0 \leq depth \leq 30$$

This reduced the dataset from 11,676 rows to 4,548 shallow-water observations.

The quantum pipeline additionally replaces positive and negative infinity with missing values and removes rows missing any selected feature or the target. The classical notebooks do not clearly document equivalent missing-value removal.

### 5.2 Feature Encoding & Transformation

The engineered variables include logarithmic transformations of the blue and green bands, a blue-to-green band ratio, and the Stumpf ratio:

$$
\text{stumpf\_ratio} =
\frac{\log(1000 \times \text{blue\_band})}
{\log(1000 \times \text{green\_band})}
$$

MLP and SVR use `StandardScaler`.

The quantum pipeline uses `MinMaxScaler(feature_range=(0, pi))` to convert each feature to a rotation angle:

$$
x' = \pi\frac{x-x_{min}}{x_{max}-x_{min}}
$$

### 5.3 Feature Engineering & Selection

No automated feature-selection or dimensionality-reduction method was used. Six spectral features were manually selected based on their expected relationship with water depth. Latitude and longitude were used for regional grouping rather than direct prediction.

### 5.4 Preprocessing Rationale

Scaling is important for SVR and MLP because both algorithms are sensitive to differences in feature magnitude. Quantum circuits also require normalized values that can be represented as rotation angles. Tree-based models are less sensitive to feature scale, but the same feature set was used to maintain consistency.

## 6. Methodology & Model Development

### 6.1 Model Selection Rationale

The project evaluates seven models:

1. **Linear Regression:** A simple baseline for measuring linear relationships.
2. **Random Forest Regression:** Captures nonlinear relationships using an ensemble of decision trees.
3. **Multilayer Perceptron Regression:** Learns nonlinear relationships through neural-network layers.
4. **Support Vector Regression:** Uses an RBF kernel for nonlinear regression.
5. **XGBoost Regression:** Uses gradient-boosted decision trees for nonlinear tabular data.
6. **Variational Quantum Regression Method 1:** Tests a Qiskit parameterized quantum circuit.
7. **Variational Quantum Regression Method 2:** Tests a PennyLane variational circuit with deeper layers and a trainable bias.

### 6.2 Model Architectures & Hyperparameters

**Linear Regression**

- Default `LinearRegression()` configuration.

**Random Forest**

- `n_estimators=100`
- `max_depth=15`
- `random_state=42`
- `n_jobs=-1`

**MLP**

- Hidden layers: `(64, 32)`
- Activation: ReLU
- Solver: Adam
- Maximum iterations: 500
- Early stopping: Enabled
- `random_state=42`
- Standardized input features

**SVR**

- RBF kernel
- $C=10.0$
- $\epsilon=0.1$
- Gamma: `scale`
- Standardized input features

**XGBoost**

- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=6`
- `random_state=42`
- `n_jobs=-1`
- No early stopping

**Quantum Method 1**

- Six qubits
- One $RY$ rotation per feature
- Two ansatz layers
- Trainable $RY$ and $RZ$ rotations
- Linear CNOT connections
- 24 trainable parameters
- Observable: `ZZZZZZ`
- COBYLA optimizer with 100 maximum iterations

**Quantum Method 2**

- Six qubits
- Six ansatz layers
- $RX$ feature embedding
- Trainable $RY$ and $RZ$ rotations
- Circular CNOT entanglement
- 72 trainable circuit parameters plus a trainable bias
- Adam optimizer with step size 0.3
- 50 epochs
- PennyLane `lightning.qubit` simulator

### 6.3 Training & Experimental Setup

The notebooks use pandas, NumPy, scikit-learn, Matplotlib, Seaborn, XGBoost, Qiskit, Qiskit Machine Learning, PennyLane, and PennyLane Lightning.

The operating system, processor, GPU, memory, Python version, and package versions are not recorded. No pinned environment file is included.

## 7. Evaluation Metrics & Justification

### 7.1 Metric Definitions

Mean Absolute Error is:

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

MAE measures the average absolute prediction error in metres.

Root Mean Squared Error is:

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

RMSE penalizes large errors more strongly than MAE.

The coefficient of determination is:

$$
R^2 = 1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

An $R^2$ value closer to 1 indicates stronger explanatory performance. A negative value indicates performance worse than predicting the mean target.

### 7.2 Justification

MAE is directly interpretable as an average depth error. RMSE highlights large prediction errors, which are important in bathymetric mapping. $R^2$ indicates how much variation in depth is explained by the model. Together, these metrics provide a balanced evaluation.

## 8. Results & Model Comparison

### 8.1 Numerical Results Table

Global test-set performance was:

| Model | MAE (m) | RMSE (m) | R-squared |
|---|---:|---:|---:|
| Linear Regression | 5.3373 | 6.6804 | 0.4334 |
| Random Forest | 5.0608 | 6.6561 | 0.4375 |
| MLP | 4.9935 | 6.3775 | **0.4836** |
| SVR | **4.7868** | 6.6097 | 0.4453 |
| XGBoost | 4.9535 | **6.4577** | 0.4706 |
| Quantum Method 1 | 12.8481 | 15.5530 | -2.0711 |
| Quantum Method 2 | 7.6518 | 8.7164 | 0.0354 |

The MLP achieved the highest global R-squared. SVR achieved the lowest MAE. XGBoost achieved the lowest RMSE among the classical models.

### 8.2 Regional Results

| Model | Region | MAE (m) | RMSE (m) | R-squared |
|---|---|---:|---:|---:|
| Linear Regression | Arabian Sea / Gulf | 4.9398 | 6.2977 | 0.4780 |
| Linear Regression | Strait of Hormuz | 3.8646 | 5.1043 | 0.3284 |
| Linear Regression | Bay of Bengal | 6.0030 | 7.4287 | 0.2501 |
| Random Forest | Arabian Sea / Gulf | 4.7710 | 6.4024 | 0.4605 |
| Random Forest | Strait of Hormuz | 1.9618 | 2.6047 | **0.8251** |
| Random Forest | Bay of Bengal | 5.9502 | 7.4412 | 0.2476 |
| MLP | Arabian Sea / Gulf | 4.6114 | 6.0592 | 0.5168 |
| MLP | Strait of Hormuz | 4.0939 | 5.0220 | 0.3499 |
| MLP | Bay of Bengal | 5.6702 | 7.1076 | 0.3135 |
| SVR | Arabian Sea / Gulf | 4.4972 | 6.2283 | 0.4895 |
| SVR | Strait of Hormuz | 2.9251 | 4.5723 | 0.4611 |
| SVR | Bay of Bengal | 4.9596 | 6.5862 | **0.4106** |
| XGBoost | Arabian Sea / Gulf | 4.7388 | 6.3162 | 0.4750 |
| XGBoost | Strait of Hormuz | 3.2836 | 4.1254 | 0.5613 |
| XGBoost | Bay of Bengal | 6.6185 | 8.5530 | 0.0060 |

### 8.3 Key Observations

- MLP produced the best global R-squared: 0.4836.
- SVR produced the lowest global MAE: 4.7868 metres.
- XGBoost produced the lowest classical RMSE: 6.4577 metres.
- Random Forest performed very well in the Strait of Hormuz, but that region contains only 54 samples.
- SVR performed best in the Bay of Bengal.
- Both quantum methods underperformed all classical models.
- Quantum Method 1 produced a negative R-squared, indicating very poor predictive performance.

## 9. Discussion & Critical Analysis

### 9.1 Analysis of Best and Worst Performers

The MLP achieved the strongest global R-squared, indicating that nonlinear relationships exist between engineered spectral features and depth. Standardization and early stopping likely helped the neural network train effectively.

SVR achieved the lowest MAE, suggesting that the RBF kernel captured useful nonlinear structure while limiting the impact of large errors.

Linear Regression performed worst among the global classical models, suggesting that the relationship is not purely linear.

Quantum Method 1 was the weakest model. Its predictions were concentrated near approximately 0.59-0.98 metres, even though the target extended to 30 metres. This indicates that the circuit output was not appropriately scaled to the target range.

### 9.2 Impact of Preprocessing

Standardization was important for MLP and SVR. Quantum feature scaling converted input values into usable rotation angles, but the target depth was not normalized. Quantum expectation values are naturally bounded near $[-1,1]$, whereas the target ranges from approximately 1 to 30 metres. This mismatch likely contributed to the poor quantum performance.

### 9.3 Overfitting and Generalization

The MLP uses early stopping, but complete training and validation curves are not reported. For Quantum Method 2, training MSE decreased from 265.97241 at epoch 1 to 76.50291 at epoch 50.

No repeated cross-validation, confidence intervals, or independent validation set was used. Random train/test splitting may also produce optimistic results when nearby spatial observations appear in both sets.

### 9.4 Computational and Practical Trade-offs

Classical models are currently more practical because they are easier to train, reproduce, and deploy. They also provide substantially better predictions on this dataset.

The quantum models require specialized libraries and circuit simulation while producing lower accuracy. Their value in this project is primarily exploratory rather than practical.

## 10. Conclusion, Limitations & Future Work

### 10.1 Conclusion

This project evaluated shallow-water depth prediction using satellite-derived spectral features. Five classical regression models and two variational quantum regression models were tested.

The classical models substantially outperformed the quantum models. MLP achieved the best global R-squared of 0.4836, SVR achieved the lowest MAE of 4.7868 metres, and XGBoost achieved the lowest classical RMSE of 6.4577 metres.

SVR or MLP is recommended for future development, depending on whether minimizing average absolute error or maximizing explained variance is the primary objective.

### 10.2 Project Limitations

The main limitations are:

- The original dataset source is not documented.
- The satellite platform and physical meaning of the bands are not specified.
- Only one random train/test split is used.
- No validation set or cross-validation is included.
- Spatial leakage may produce optimistic test results.
- The Strait of Hormuz region contains only 54 samples.
- No formal outlier analysis is provided.
- Missing-value handling is inconsistent between notebooks.
- Training times and hardware requirements are not reported.
- Package versions are not pinned.
- No saved model files or reproducible training scripts are included.
- Quantum target scaling was not implemented.
- Regional boundary definitions exclude some longitude ranges and exact boundary values.

### 10.3 Future Work

Future work should:

- Document the original satellite and bathymetric data sources.
- Add complete correlation, distribution, and outlier analyses.
- Use spatial cross-validation or spatial holdout testing.
- Repeat experiments over multiple random seeds.
- Apply systematic hyperparameter optimization.
- Test target transformations and prediction intervals.
- Apply consistent missing-value handling.
- Normalize the quantum target and inverse-transform predictions.
- Compare against a constant-depth baseline.
- Report training and inference times.
- Save trained models and preprocessing pipelines.
- Deploy the best model as an API or web application.
- Investigate larger and more balanced regional datasets.

## 11. References & Declarations

### 11.1 References

The final report should cite:

- Scikit-learn documentation for Linear Regression, Random Forest, MLPRegressor, SVR, preprocessing, and evaluation metrics.
- XGBoost documentation for `XGBRegressor`.
- Qiskit and Qiskit Machine Learning documentation for VQR and quantum circuits.
- PennyLane documentation for variational circuits and quantum simulation.
- The original satellite-derived bathymetry dataset, once its source is confirmed.
- Academic publications describing the Stumpf ratio and satellite-derived bathymetry.

### 11.2 Synthetic Data / Generative AI Declaration

The project files do not document the use of synthetic data or generative AI. Complete the following declaration according to the actual project history:

> No synthetic data was intentionally used in this project. The dataset was obtained from [insert verified source]. Generative AI tools were [not used / used] for [describe purpose]. Any generated code or text was reviewed, tested, and validated against the project data and notebook outputs.
