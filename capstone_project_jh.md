# Machine Learning Engineer Nanodegree
## Capstone Project
Joseph Heenan  
September 19, 2017

## I. Definition
_(approx. 1-2 pages)_

### Project Overview

Computational chemistry is the study of chemical structures and reactions using computer methods. This project focuses on applications of machine learning algorithms, in particular deep learning algorithms, to
the problem of 'virtual drug screening' in computational chemistry. This is the challenge of developing an algorithm that will give useful predictions as to which small molecule may interact meaningfully with a biological target relevant to human medicine.

Bioassays produce much of the data used by computational chemists; they are a form of high-throughput experiment in which many small molecule compounds are tested against a target biological system to determine which of them exhibits ‘bioactivity’ in that system. The definition of ‘bioactivity’ may vary from assay to assay but often means increased or decreased expression of a particular gene or protein.

There is a wealth of bioassay information hosted by the US National Institutes of Health at their [PubChem](https://pubchem.ncbi.nlm.nih.gov) site. As of the time this report was written, the Bioassay database at PubChem "...held over one million records holding 230,000,000 bioactivity outcomes deposited by over 80 organizations" (Wang, Yanli et al. 2017). Clearly there is a wealth of well-labelled data available at this site. 

For this particular project, we use the data that is available for Bioassay 1030, which looks at inhibitors for the gene ALDH1A1, a gene that is implicated in metabolic disorders. The source data is available [here](https://pubchem.ncbi.nlm.nih.gov/bioassay/1030). We utilize here the assay data table as well as a table with canonical module representations (referred to as SMILES format) as the base data for this project.



### Problem Statement

At a high level the problem we want to solve is that of drug discovery; that is, helping to discover new small-module medicines. More specifically, we want to create a binary classifier that, when given a sample module in an appropriately featurized format, will predict whether or not it will be actice in inhibiting ALDH1A1. This predictor should be both sensitive and specific; we want to neither miss potential new medicies, nor do we want to spend time lab testing 'false positive' predictions that turn out not to be bioactive. This binary classifier will allow us to implement a type of 'virtual drug screening' where we can test a large library of modules which we may not yet be able to physically produce and test, in order to focus and steer future confirmatory laboratory testing.

The strategy I will use to attempt to find a sensitive and specific binary classifier is to experiment with several well-known machine learning algorithms from the packages of Scikit-Learn and Keras. I plan to take the following steps:


1. Ensure source data is properly merged loaded, using the Pandas library
2. Ensure that source data is balanced (as per reviewer suggestions). Because over 90% of the bioactivity results are 'False' (not bioactive) a naive classifier could simply preduct 'False' and achieve high accuracy. I plan to use the imbalanced-learn package for sklearn to create a balanced dataset to help avoid this problem.
3. Ensure that proper mevaluation metrics are selected; I intend to use the binary accuracy and F1 score metrics to compare different classifiers because this will help to ensure that sensitivity and specificity are both considered during evaluation
4. Experiment with several different classifiers, specifically
	a) Decision Trees (sklearn)
	b) Random Forests (sklearn)
	c) Fully-Connected Deep Neural Networks (keras)
5. Experiment with several different types of featurization for the above classifiers, such as Morgan fingerprints of different sizes, as well as trying to learn on the SMILES representation directly.
6. Experiment with optimization of model hyperparameters using a validation set

Because the bioassay 1030 presents a fully labelled dataset of 220,402 compounds tested, we will hold out a section of test data in order to evaluate the performance of the binary classifier. 

I anticipate that the best solution to this problem may be found in the application of a deep learning algorithm, as these have shown best-in-class performance on several binary classification problems in recent years, e.g., the Cats vs Dogs problem on Kaggle linked [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).

In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section.

### Metrics

I plan to examine the average F1 score across the two prediction class (True, e.g. predicted bioactive, and False, e.g. predicted not bioactive) for each of the models. Using this will provide an objective way to compare the performance of different models, while appropriately penalizing both "missed predictions" and "wrong predictions".


## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration

We conduct the majority of our exploratory analysis in the "pubchem_exploratory_analysis.ipynb" notebook.

We begin by exploring the characteristics of the class that we will use for training.

Here we note that there are 220,365 distinct observations with associated outcomes, and that the most frequent outcome is 'inactive', with 148,299 of the observations being inactive. The unique outcomes are "Active", "Inactive", and "Inconclusive". We note that the data are distributed as follows:

```python
Active: 16112, Inactive: 148299, Inconclusive: 55954
```
That is, we have 3 unique values for activity in the dataset, of which 'Inactive' is by far the most common with approx 148k of the values; there are approx 55k inconclusive outcomes and approx 16k active outcomes. We are now presented with two challenges:

1.  How do we engineer these three outcomes to fit in the desired binary classification scheme. Do we wish to drop the inconclusive results, or do we instead count them as 'inactive'? And:
2.  How do we handle the fact that the outcomes are highly imbalanced, e.g. the 'inactive' class is over 7 times larger than the 'active' class?

Our tentative plan is to coalese the 'Inactive' and 'Inconclusive' values, and to under-sample the 'inactive' class to obtain even representation of 'active' and 'inactive' when training, but we will return to this decision in the "featurization" step of our planned strategy later to determine whether this is optimal.

In terms of the fingerprints themselves, there are several domain-specific parameters which can be used to tune them; specifically the size of the bit array as well as the specific molecular features that should be used can be considered can be specified (such as chiral form and other domain-specific features). Given that prediction of activity against ALDH1A1 is still only partially-understood, we hope that tuning these parameters may result in a more robust model for future studies.

One other point of note is that the labelled dataset does contain activity scores for each compound in the form of a PUBCHEM_ACTIVITY_SCORE column. Active compounds are those with scores between 40 and 100, Inconclusive have scores between 40 and 1, and Inactive compounds have scores of 0. We also considered rephrasing the problem from one of binary classification (Active vs Inactive) to one of Linear Regression (predicting an activity score, and from that deriving a classification).

Regarding related datasets, the MoleculeNet benchmark [paper](https://arxiv.org/abs/1703.00564) deserves particular note, as it includes this bioassay as part of its list of 128 PubChem assay datasets that it analyzes. However the results are not directly comparable, as MoleculeNet looks at models that learn joint features via making a multi-task/multi-class prediction (e.g., for 1 SMILES string, it will predict bioactive or not against 128 different assay targets). Nevertheless it is important to reference this paper, as well as the 128-class [dataset](https://github.com/deepchem/deepchem/blob/master/datasets/pcba.csv.gz), as providing an important reference point for this work.


### Exploratory Visualization

As discussed earlier, the root dataset is highly imbalanced. Below is a histogram of the activity scores of each compound. Compounds with an activity score of 0 were classified as inactive in the original assay; scores from 0-40 were ranked as inconclusive; and scores from 40 and above were ranked as active. We can see that the great majority of the scores are under 40:

<a href="" target="_blank"><img src="FrequencyOfActivityDistribution.png"/></a>

In the above diagram, the X axis shows the activity score, and the y axis shows the number of compounds falling without this bucket of e.g., 0-10 activity score, 10-20 activity score, etc.

We also construct visualizations of the first two compounds which were tested in the assay, in order to show how the SMILES string allows for reconstruction of a chemically-relevant entity:

<a href="" target="_blank"><img src="MoleculeVisualizations.png"/></a>

In this diagram, the title shows two example SMILES string; the left molecule represents the molecule with the SMILES string for "Example 1" and the right molecule represents the molecule with the SMILES string for "Example 2".

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

I intend to look at a range of algorithms for solving this problem; specifically decision trees, random forests as well as both shallow (via MLPClassifier) as well as deep (via KerasClassifier) neural networks to see how well the algorithms are able to classify the data. I avoid the use of support vector machine-based classifiers given the large size of the unsampled dataset. 

One particular challenge of this task is that it involves unbalanced learning, as postive results occurr in a ratio of approximately 1 positive result for every 9 negative results. If required I will undersample the dataset to achieve a balanced dataset for training.

### Benchmark

Because this is a novel analysis, there is no directly comparable benchmark available. For this reason I will look directly at F1 score metrics obtained, examining them on a class-by-class basis to ensure the model is not simply predicting 'false' to obtain a high F1 score.

One partially-comparable benchmark is the MoleculeNet benchmark [paper](https://arxiv.org/abs/1703.00564), which notes an area under curve for the receiver operating characteristic (AUC-ROC) of .781 for the test set of their model, which given a SMILES string simultaneously predicts bioactivity on each of the 128 assays the model was trained on (this is what is meant by 'multi-task' in this context).

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

The unbalanced dataset abnormality needed to be corrected during preprocessing, as well as one module that failed featurization was removed.

Additionally we coalese the "inactive" and "inconclusive" activities into both "inactive" to make analysis and engineering effort less. We conduct some experimental regression analysis to see if modelling based on the raw activity scores provides much of a boost to the model, however we determine that it does not.

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

The implementation was carried out using scikit-learn and Keras. There were a number of challenges encountered during the implementation. Initially as a first step I created a Decision Tree classifier based on the raw dataset (without sampling) and was surprised to find a high cross-class F1-score of around .9. However when analyzed via the classification_report feature of scikit-learn, I realized this was due to excellent performance in nearly always predicting false for the label but with an F1 score for True class around .3; this meant little value however for a model that we want to have precision in finding new medicines. Initially I attempted to use several different values for the class_weight parameter available in scikit-learn classifiers however this made little improvement in the F1-score for the positive class. For this reason I focused the rest of my analysis on using an undersampled dataset which contained an even number of True and False class labels.

Another challenge was the size of the dataset and computational complexity of training deep neural networks; I found myself needing to use a GPU-equipped machine to train the Keras-based classifiers in a reasonable amount of time.

An additional challenge was implementing the regression model. A basic model was implemented in pubchem_regression_notsampled [report]("pubchem_regression_notsampled.html") and [notebook]("pubchem_regression_notsampled.ipynb")

A final challenge was the large amount of memory required during the featurization process; many times over 64 GB of RAM was used and models failed to train on the development virtual machine. For this reason as well, an undersampled dataset was generally used, except for in the experimental section on regression.

### Refinement

The best initial solution that was found was a RandomForestClassifier; this was both relatively easy to implement and efficient to evaluate on the undersampled dataset. This classifier achieved an average F1-score for the positive class of .70 under 2 folds of testing:

             precision    recall  f1-score   support

          0       0.70      0.72      0.71      8055
          1       0.72      0.69      0.70      8055

avg / total       0.71      0.71      0.71     16110

The F1-score for the negative class was also a respectable .71.

During the refinement process, I made sure to use k-fold cross-validation, so that I could be sure that my results were robust an not due to a particularly lucky or unluckly train-test split. I additionally used an open-source package, hyperopt-sklearn, to see if an algorithm could do a more efficient job of finding an optimal model than my manual analysis could. I did mark this section as experimental in case future evaluators have trouble reproducing it; it did not change my selected model.

The process of refining the initial model is documented in the pubchem_bioassay_sklearn notebook [report]("pubchem_bioassay_sklearn.html") and can be reproduced using the pubchem_bioassay_sklearn [notebook]("pubchem_bioassay_sklearn.ipynb").

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

I had expected the final classification model to come in the form of a deep neural network, per my proposed methodology, but was surprised that using a deep neural network on the classification dataset failed to yield better results than the Random Forest documented in the Refinement section. I believe this may be due to the fact that the effective size of the dataset was significantly reduced due to the need to create balanced classes for training; I went for a dataset of approximately 220,000 datapoints pre-processing, down to about 16,000 datapoints during the training process. I experimented with multiple deep neural network configurations using the Keras library but failed to obtain a average F1-score better than .68 in any configuration. A representative analysis is shown in the pubchem_dnn notebook [report]("pubchem_dnn.html") and can be reproduced using the pubchem_dnn [notebook]("pubchem_dnn.ipynb").

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

### Citations

Wang, Yanli et al. “PubChem BioAssay: 2017 Update.” Nucleic Acids Research 45.Database issue (2017): D955–D963. PMC. Web. 19 Sept. 2017.

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
