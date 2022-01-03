`[__Update 01/22__: This is an old repository. These collect my notes and implementations of Machine Learning models back in 2015, as I explored the early landscape of libraries across Java and Python. Java was infact the overwhelmingly preferred language for Machine Learning back then. With Theano, PyBrain and Lasange gaining momentum. ]`

#### For the moment, this repository will act as a collection of various Machine Learning Models implemented through multiple platforms. These include:
    Python Packages: Scikit(Sk) Learn, PyBrain and Theano
    
    Java libraries: DeepLearning4j , Neuroph, JavaML, Weka
    
    C Libraries: OpenCV, FNN
    
    Lua Library: Torch
    
# **Java Libraries:** 
    
Java clearly doesn't offer the most popular ecosystem for Machine Learning implementations. It has lagged behind R and Python, and even Lua (courtesy of Torch) in both depth and breadth of the models offered. The following are the set of Libraries offered by the Java ecosystem, that make a convincing argument that Java might finally be catching on, and offers advantages of its own. 
    
## *DeepLearning4J*

This package is a great way to get started with multi-layer nets and allows configuration of a variety of hyperparameters easily and robustly. DeepLearning4j uses Canova for vectorization of Data which is a rather convenient implementation, similar to the *Python statistical libraries* (very useful). 

Finally, and most importantly, this library natively allows for **Distributed implementations** of Deep Nets, with a modified implementation of Map-Reduce called Iterative-Reduce. DeepLearning4j allows deploying & training models using CUDA, Amazon EC2, Spark & Hadoop. 
Also, Note: Maven is heavily utilized here. So for someone new to it, it'll take some getting used to.

#### Regarding Files Updated:

        1. Multi-Layer Perceptron Implementations for MultiClass Classification Problems.
    
        2. Several iterations of hyperparametric changes to the Network to understand their pros and cons;
           From L1 & L2 Regularization, Drop-offs (which is rather convenient in Dl4j) to varying weight-initializers, Error Updaters, and large variations in network structures, needless to say. 
    
        3. Models have all been serialized and stored along with respective Logs and paramter/hyperparameters saved in bin and jsons
            for future reference.

Deeplearning4j also allows integration with Hadoop, Spark and porting pre-trained models from Caffe; Interesting. Hope to try it out on CPU Clusters soon.

## *Java ML Library*

One of the better ML Libraries offered by the Java Community, offers a variety of Model Implementations, though the documentation on it is rather poor. (There is only one tutorial PDF to get you started)


As far as the implementation goes, the procedure is rather straightforward, and the models require little or no parameter tuning. 
The library is weak in terms of the variety and efficiency of implementations of models offered and it is clear from the offset, the library wasn't designed for scaling. 

Further, as far as I could find, Java_ML lacks a native Plotting module for ROC curves and uses Weka Library externally( of which some modules have to be downloaded seperately).


*Another major drawback of the library, which is precisely where Weka shines, is its inconvenience of implementation; In much of the routinely needed functions for model optimization and comparison, JavaML provides no support, leave alone for GPUs or Cluster computing; It simply isn't meant for scaling, or even for routine research (its lacking too many things to be used for it). For example, K-Fold validation on Java ML is verbose and RBF SVMs non-existant; Code for ROCs and AUCs have to be written from scratch, and that is for perhaps the de-facto reference for model evaluation.*

All in all, Java ML is a decent library, perhaps best suited for basic implementations without a need for scaling to distributed computing/multiple GPUs and for suitors to a Java library offering a range of implementable models at a high level of abstraction.

#### Regarding Files Updated:

        1. Trained KNearestNeightbours model with varying 'K' and attempted to optimize its value through test-set performance instead of Validation sets. 
            (No K-Fold Validation available **Update**: Wrote KFoldValidation functions from scratch and implemented the same for 70:30 train set split) 
    
        2. RandomForest,Linear SVM models have also been added. (Trained with Sample RGB DataSet)
    
        3. A weird issue with Serializability of the Naive Bayes model, but besides that, its performing quite well.
    
        4. Added Serializability to the rest of the ML models. The models can now be saved, their structures and hyperparameters logged and loaded through the _tester source for re-use.
    
        5. Some DataProcessing source code has been added, that uses java.ml.core libaries, hence updated here.
    
## *Weka* 

Weka is offered both as a software application for Researchers, and as a Java module to other Java libraries. Weka provides optimized models with easy, highly abstracted implementations (similar to JavaML) and is well-rounded with functions for a range of methodologies. In contrast to other libraries, Weka has a rather comprehensive GUI support in various areas; It allows "Auto-building" in MultiLayer Perceptrons, which is essentially code for "We'll display a sample model, you can modify it by clicking and changing its structure, parameters and hyper-parameters as you like". Convenient. 


#### Weka implementations: 

        1. KnearestNeighbours model with 10-fold validation on Medical dataset.
        
        2. Random Forest, LibSVM's implementation with Weka along with ROC curves for all models.
        
        3. MultiLayer perceptron network with varying structures.
        
        4. Wrote a script to convert .csv files to .arff files.

        
# **Python Libraries**

Python has powerful libraries for Machine Learning & Neural Networks in Scikit-Learn, Theano & PyBrain and the various others built on these (Lasagne is a good example). Python uses the powerful Numpy library to its advantage with optimized Matrix multiplications, and is one of the most efficient, scalable platforms for ML implementations. 
Further, it has perhaps the most intuitive, easy-to-pickup implementations.
Really, Scikit-Learn takes a few minutes to get familiar with and has a variety of Machine Learning models & Data Pre-processing functions. 

For the time being, only the Theano & Scikit-Learn implementations have been added, but I hope to try out the Lasange & PyBrain libraries a little more in depth. 

#### Regarding files updated:
    
        1. Model comparison of Linear SVM, Random Forest, K Nearest Neighbors, Naive Bayes, RBF SVM using F1 scores & ROC curves.
        
        2. Comparison of Kernel variants of PCA on sample dataset using Plots, ROC curves on sample Classifier.
        
