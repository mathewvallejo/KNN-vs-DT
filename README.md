# KNN-vs-DT
K-Nearest Neighbor and Decision Tree Comparison on BCW and NHANES Datasets
# Analysis of NHANES and BREAST CANCER Datasets
*Bonaventure Dossou, Mat Vallejo, Minjae Kim*  
*January 31, 2024*

## Abstract
This report examines the performance of two machine learning models, K-Nearest Neighbor (KNN) and Decision Tree (DT), on two distinct datasets: NHANES and BREAST CANCER. The evaluation was conducted by exploring various aspects of model tuning such as hyperparameters, distance functions (KNN), and cost functions (DT), across both datasets and comparing the results. The results show strong capabilities of both models to accurately predict target values, with both models outperforming each other, under certain conditions. Notably, the DT model was able to achieve perfect accuracy on the NHANES dataset due to the nature of the input variables.

## 1. Introduction
The purpose of this assignment is to implement K-Nearest Neighbor (KNN) and Decision Tree (DT) machine learning models across two distinct datasets, NHANES and BREAST CANCER. The tasks include importing and cleaning the datasets, splitting the datasets into train/validation/test sets, establishing accuracy across all input variables as well as specific subsets of highly relevant input features. Additionally, we have to build the models, and experiment them with various hyperparameters, in order to access their efficiency and accuracy in predicting target values. Hyperparameters include different values of k and different distance functions in the KNN model and cost functions in the DT model as well as K-fold cross validation.

## 2. Methods
### KNN Method
The KNN method is widely used in the machine learning field for classification problems, and involves measuring test data against training data that has been fit to the model by use of distance functions (see more in KNN Experiments). The test data is evaluated by taking the test input variable(s) ’X’ and finding the closest k-number of training target values ’y’. By calculating the distance between these points, the model can infer the most likely label for the target value.

### DT Method
The DT method can be used for classification, regression, or a combination of both. It functions by passing input values ’X’ through a combination of conditional nodes that determine whether or not a value should be passed either ”left” or ”right” by implementing certain cost functions using the training datam (see more in DT Experiments). When the input variable reaches either a pre-designated maximum ”depth” or the last node in the set, known as a leaf node, the decision ’y’ is made.

## 3. Dataset Description and Statistics
The sub-dataset of the NHANES, managed by the CDC, selectively extracts features such as physiological data, lifestyle choices, and biochemical markers from the broader dataset to predict the age of a diverse U.S. population, addressing its usual breadth for more specific analysis. The Breast Cancer dataset is the original Wisconsin Breast Cancer Database.
NHANES contains no missing values and has no duplicate values. Breast Cancer has one variable called Bare nuclei which has 16 missing values, representing 0.023% of its total number of samples. Consequently, we decided just to dynamically drop those values. Additionally, this dataset also has 234 duplicate values (out of 683), that have been dropped to avoid contamination.

## 4. Results
### 4.1 Feature Importance and Impact on Performance
Based on the results in Table 3, it is overall obvious that using top-3 features gives the best results, on both models. Moving on next, we will stick to those 3 features. Those features were selected as follows:
• Compute the square difference between both classes for all features and select the top ones. 
• Compute the Spearman Rank Correlation between all features and the target variable.
• The final set of features used is the intersection of features of the two steps described above.
### 4.2 KNN Experiments
#### Evaluation of different values of k hyperparameter for KNN
We decided to stick to AUC to select the optimal k. This is because AUC is better when the problem is imbalanced (which is the case here). Given this, AUC is also more robust to changes in the threshold and can capture the model’s performance across the entire range of probabilities. Finally, AUC is also useful when the cost of false positives and false negatives are different and need to be balanced. Overall, AUC will allow us to select models that achieve false positive and true positive rates that are significantly above random chance, which is not guaranteed for accuracy. Consequently, based on the results of Table 4, for NHANES and BREAST CANCER datasets, the chosen best k values are respectively k = 10 and k = 30. To compute the AUROC, since we need the probabilities of the positive class, we have implemented and used the predict proba in both models’ classes. These metrics are computed using the Euclidean distance function. In the KNN class, we have also implemented the cosine similarity and the Manhattan distance.
#### KNN Comparison of Several Distance Functions with the Best Hyperparameters

Following the results in Table 5, we can see that in the case of NHANES, the Manhattan distance works better than the Euclidean distance; as opposed to the BREAST Cancer dataset. The following might be possible reasons:

• When the data is high-dimensional, Manhattan distance can better capture the similarity between points than Euclidean distance (Source).
• When the data has different scales or units, Manhattan distance can be more robust to outliers or irrelevant features than Euclidean distance (Source).
• When the data is discrete or categorical, Manhattan distance can be more appropriate than Euclidean distance, which assumes a continuous space (Source).
• Manhattan distance uses the L1 Norm which encourages sparsity while Euclidean Distance uses the L2 norm. Manhattan distance works better input variables are not similar in type (e.g. in NHANES they are Glucose, Oral, and Age which are highly likely unrelated vs in BREAST they are Bare nuclei, Unifor- mity of cell size, and Uniformity of cell shape which are very likely to be related). Moreover, due to the curse of dimensionality, Euclidean distance becomes a poor choice as the number of dimensions increases. (MIT Lesson).
For the remaining KNN experiments, we have used the Manhattan and Euclidean distances, respectively for the NHANES and BREAST CANCER datasets.
#### Exploring Impact of K-Fold Cross Validation
We implemented the k-fold cross validation technique and leveraged it important on the test set. Below are the results:

Comparing Tables 6 and 5 (experiments without cross validation), we can notice that K-Fold cross-validation helps. However, compared to the best hyperparameters results, these new results are statistically insignificant (in the case of the BREAST CANCER dataset) and not better in the case of the NHANES dataset.

### 4.3 Evaluation of DT model
The evaluation of the decision tree model was measured for a range of maximum depth k for k = [1,20] using three separate training costs and finding the best depth for each. The range of k depth values was chosen to give general context to the results after preliminary tests showed significant diminishing returns in accuracy variance after a max depth of k ≈ 7. The training costs implemented were chosen as misclassification rate, entropy, and gini index. The implementation of the same decision tree iteration using various training cost calculations highlights the strengths, weaknesses, and similarities of each in the results. General attributes of each training cost are as follows:
• Misclassification measures the total proportion of misclassified samples and has the potential to be insen- sitive as a training cost as a result (Source).
• Entropy refers to the amount of uncertainty present at a given node and is calculated by evaluating the potential split from each input variable across the possible outcomes. The lower the entropy, the more information is gained increasing decision accuracy (Source).
• The gini index is commonly used as the default cost function for decision tree architectures (such as in sklearn) and calculates how likely a variable is to be misclassified. In this case, low values also equate to
increased decision accuracy (Source).

Considering results in Table 7, in the case of the NHANES dataset, the model performed optimally under all conditions. We see subtle shifts in accuracy in the BREAST CANCER implementation, notably when using entropy as the training cost, which resulted in a slight decrease in accuracy measurements but a slight increase in AUCROC. Overall, the accuracy of the model is functionally consistent across all hyperparameters and cost functions.


## 5. Conclusion
Experiments conducted for the NHANES and BREAST CANCER datasets reflect the varying results that can be achieved when implementing KNN and Decision Tree machine learning models. As shown in Figure 1 (see Appendix), these experiments yielded high accuracy across both models and datasets, achieving perfect accuracy for all implementations except for the implementation of DT on the BREAST CANCER dataset. Notably, both models perform comparably well under most circumstances, and there is often no perfect model choice for every scenario. As a result, fine-tuning the respective models using various cost/distance functions and other tools such as value-weighting proves critical for determining the relative performance of a given model under certain conditions and preparing for optimal model selection.

## 6. Statement of Contributions
Data cleaning, KNN implementation, Report - Bonaventure Dossou | 
Data cleaning, DT implementation, Report - Mat Vallejo | 
Review and Report - Minjae Kim
## 7. Appendix
![Figure 1: Comparison of AUCROC scores KNN vs DT cross both datasets](link_to_image)
