# Skoltech, Intro to CV, final project, Artem Galliamov, Max Komatovskiy

The main idea with this repository is to play around with FER 2013 dataset.

NB the overall record of f1-score on Kaggle seems to be gained by a quite complex and heavy neural net, which we simplified and trained using Colab GPU's for the bonus part (in the end).

# A short brief of what happens in the scripts

The first problem, we ran into was the number of badly-posed images, such as black and white squares, photos of heads from such angles, that the emotion can't be distinguished even by human brain and so on.

We performed thresholding, based on the sum of intensities histogram (cut off the almost black and almost white samples).

We then perforemed template matching (eye-search) in the specific pixel area, do that only straightforward faces are kept.

Let us note, that such preprocessing, despite reducing the dataset, has a severely good impact on the metrics of all applied approaches.

Further we tested some teqniques of a multilabel classification (7 emotions, including neutral).

As the pictures in the dataset a of quite a low resolution, we need to perform some appropriate feature engineering, which is a bit tough, because typical methods, such as binarization and thresholding wouldn't work due to the various light conditions of the images even in a preprocessed dataset. Canny edge detection and gradients are also not so informative due to their thickness, they keep only the most general information and lack the usefull details. What should we do?

The most naive technique is based on the cosine similarity and eigen-based feature extraction.

We work with a matrix, formed columnwise of vectorized (len = 2304) images. We centralize this matrix, subtracting the mean column out of each column in the matrix.

By this we gain somewhat of a residual matrix, the dispersion of which must be explained.

We approximate this residual matrix, performing rank-compressing SVD with a hyperparameter R.

NB! The whole dataset may not be decomposed on a typical CPU, cause for some cases it may not fit into memory, so random subsampling of subdatasets and ansembling several models were used to improve the model.

The columns of U now form a strong basis in the space of all the faces presented in the columnwise matrix. Tho core information about any image can be deduced from the coefficients of it's decomposition in this basis. That is what we do further.

Each image from the vectorized testset we project, using the properties of columnwise unitary U to get the coefficients of linear combination of basis columns of U, corresponding to the test image. The coefficients for train images are contained as S@V* (the product of the last two factors).

We now want to use these features as an input for the classification problem.


Model 1. Naive approach, based on 1NN cosine similarity.

We project the test image onto the span of columns of U and find the closest in terms of cosine similarity train image (coefficients vector used as features),
so the amount of features used may be changed by changing the rank-reducing R.

We need to perform the compreddion for the sake of robustness, avoiding the overfitting and reducing the computational burden of the model fits further.

The predicted label for the test image is considered to be the same as the label of the closest training one


Model 2. Stronger classifier, same features.

If we train somewhat of a strong model, e.g. gradient boosting on the same features, the results i,prove rapidly. We used out-of-box LgbmClassifier, but we are sure that some improvements may be gained by tuning the hyperparameters.


Model 3. + bonus part and out-of-dataset-testing.

We reimplied the CNN, we found in Kaggle Kernels, they all looked quite the same from the prospect of their architecture and trained it. We also wrote a script, taking a shot using the computers front camera, preprocessing the image in the way to fit the FER2013 dataset input and outputting prediction of the current emotion into the .txt file. The whole process takes around 2.3 seconds, one may play around with it, it is fun, at least for a couple of minutes.

The f1-score of the classification, performed by the network is close to the maximally-attained one (around 0.71). Of course, due to the class diabalance, the average = 'weighted' was used.




