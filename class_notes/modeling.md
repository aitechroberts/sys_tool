# Performance
ROC (Receiver Operating Characteristic) Curve
- plots True Positive Rate (TP/(TP+FN)) against False Positive Rate (FN/(TP+FN)) normalized from 0 to 1

```python
# make the ROC curve, all for Logistic Regression, but all the same regardless
lr_pred_prob = lr_predictions.select("probability")
to_array = F.udf(lambda v: v.toArray().tolist(), T.ArrayType(T.FloatType()))
lr_pred_prob = lr_pred_prob.withColumn('probability', to_array('probability'))
lr_pred_prob = lr_pred_prob.toPandas()
lr_pred_prob_nparray = np.array(lr_pred_prob['probability'].values.tolist())

lr_fpr, lr_tpr, lr_thresholds = roc_curve(outcome_true, lr_pred_prob_nparray[:,1])
# first input is the column vector of true label, second input is column vector of probability of attack
```
Evaluate the performance of ROC curve at inflection point

## Evaluate Area Under Curve
In Classification Evaluator classes in pyspark.ml.evaluation, raw = prediction vector column, labelCol is the true label column. Then evaluator.evaluate(predictions)

## Hyper-Parameter Tuning
In the Classification class regressions, you can specify other arguments beyond the two columns, including `maxIter`,`regParam` aka regularization parameter, `elasticNetParam`, actually control the fitting (training) behavior and may impact the performance of the fitted model. These arguments are called **Hyper-Parameters** and the goal of cross validation is to change the hyper-parameters with the goal of **finding a combination of hyper-parameters that produce a fitted model with the best performance**, where the performance metric we use today is AUC. 
