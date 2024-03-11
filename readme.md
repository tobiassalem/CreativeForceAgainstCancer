### CreativeForceAgainstCancer
A machine learning classifier, built with Python and the SciPi libraries
- scipy
- numpy
- matplotlib
- pandas
- scikit-learn


### Dataset Information
What do the instances in this dataset represent?
Each instance corresponds to a patient with a tumour. These all have a set of the measured parameters, 
like the mean radius, mean perimeter, the texture and compactness. 
The diagnosis class indicates 1 for malign, and 0 for benign.


### Workflow
1) Setup environment, check Python library versions etc.
2) Look at the data, check statistics, and visualize it.
3) Perform any pre-processing, like data scaling, cleaning the data from duplicates, etc.
4) Build and evaluate the models (e.g. with K-fold cross validation). 
5) Make predictions.

### Troubleshooting
When using tensorflow keras, if you get an error on the following form:  

```
ValueError: Input 0 of layer "sequential_1" is incompatible with the layer: expected shape=(None, 455, 30), 
found shape=(None, 30)
```
then replace the 1st `model.add` line segment. Instead of 
```
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape, activation='sigmoid'))
```
use
```
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
```
Most importantly recompile all segments, from start to finish after making this change.

#### Background:
The Tensorflow model expects the first dimension of the input to be the batch size.
In the model declaration however, they set the input shape to be the same shape as the input. 

To fix this you can change the input shape of the model to be the number of features in the dataset.
The number of rows in the .csv files will be the number of samples in your dataset. 
Since you're not using batches, the model will evaluate the whole dataset at once every epoch.

Ref. https://stackoverflow.com/questions/70067588/valueerror-input-0-of-layer-sequential-is-incompatible-with-the-layer-expect

### References
* Python - @See https://www.python.org/
* SciPy - @See https://scipy.org/
* Pandas DataFrame - @See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
* Data Scaling - @See https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
* K-fold cross validation - @See https://machinelearningmastery.com/k-fold-cross-validation/
* Cancer markers - @See https://www.cancer.gov/about-cancer/diagnosis-staging/diagnosis/tumor-markers-fact-sheet


