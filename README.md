# MLNeuralNetworksRepository
Beginning a public repository for all the requisities for implementing baseline Neural Network &amp; other ML Implementations

Henceforth, all meandering ML Implementations will be pushed to this Repository.






#### For the moment, this repository will act as a collection of various Neural Network Implementations through multiple platforms. These include:
    Python Packages: Scikit Learn, PyBrain and bits of Theano
    
    Java libraries: DeepLearning4j , Neuroph, JavaNet
    
    C Libraries: Torch, FNN
    
## DeepLearning4J 
This package is a great way to get started with multi-layer nets and allows configuration of a variety of hyperparameters easily and robustly. DeepLearning4j uses Canova for vectorization of Data which is a rather convenient implementation, similar to *Python statistical libraries* (very useful). 

Finally, and most importantly, this library sufficiently allows for ConvNet and deep-learning implementations (as the name suggests) from the get go with GPU and CUDA support.
Also, Note: Maven is heavily utilized here. So for someone new to it, it'll take some getting used to.

#### Regarding Files Updated: 
    1. Several Multi-Layer Perceptron Implementations for Sample datasets. 
    
    2. Logs and paramter/hyperparameters saved in bin and jsons for future reference.
    
    3. .docx files to log the F1-Scores for each model

## Java ML Library

One of the better ML Libraries offered by the Java Community, offers a variety of Model Implementations, though the documentation on it is rather poor. (There is only one tutorial PDF to get you started)

As far as the implementation goes, the procedure is rather straightforward, and the models require little or no parameter tuning. 
The library is weak in terms of the variety and efficiency of implementations of models offered. 

Further, as far as I could find, Java_ML lacks a native Plotting Library/module for ROC curves/graphs and uses Weka Library externally( of which some modules have to be downloaded seperately).

#### Regarding Files Updated:
    1. Trained KNearestNeightbours model with varying 'K' and attempted to optimize its value through test-set performance instead of Validation sets 
    
    2. RandomForest,Linear SVM models have also been added. (Trained with Sample RGB DataSet)
    
    3. A weird issue with Serializability of the Naive Bayes model, but besides that, its performing quite well.
    
    4. Added Serializability to the rest of the ML models. The models can now be saved, their structures and hyperparameters logged and loaded through the _tester source for re-use.
    
    5. Some DataProcessing source code has been added, that uses java.ml.core libaries, hence updated here.
    
