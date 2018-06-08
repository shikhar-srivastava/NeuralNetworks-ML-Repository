package sys.Mark2;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.DataOutputStream;
import java.io.File;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;

/**
 * @ author- Shikhar Srivastava
 * @ credit due to : Relevant Authors of DeepLearning4J Library
 *
 */
public class MLPClassifier_Mark1 {


    public static void main(String[] args) throws Exception {
        String type="Mark2_ver0.1";
        int seed = 123;
        double learningRate = 0.005;
        int nEpochs =9001;
        int iterations= 1;
        int numInputs = 23;
        int numOutputs = 6;
        int numHiddenNodes = 20;

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/Feature_test_ann.csv")));
        DataSetIterator testIter =(DataSetIterator)  new RecordReaderDataSetIterator(rrTest,269,0,6);
        DataSet test_set=testIter.next();

        RecordReader rrTrain= new CSVRecordReader();
        rrTrain.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/Feature_train_ann.csv")));
        DataSetIterator trainIter =(DataSetIterator)  new RecordReaderDataSetIterator(rrTrain,414,0,6);
        DataSet train_set= trainIter.next();

        train_set.shuffle();
        train_set.normalizeZeroMeanZeroUnitVariance();
        test_set.shuffle();
        test_set.normalizeZeroMeanZeroUnitVariance();


        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
            .learningRate(learningRate)
            //.l1(1e-1).regularization(true).l2(2e-4)
            .updater(Updater.NESTEROVS)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes*2)
                .name("FirstHiddenLayer")
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(1,new DenseLayer.Builder()
                .name("SecondHiddenLayer")
                .nIn(numHiddenNodes*2).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(2,new DenseLayer.Builder()
                .name("ThirdHiddenLayer")
                .nIn(numHiddenNodes).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
        .layer(3,new DenseLayer.Builder()
            .name("ThirdHiddenLayer")
            .nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .momentum(0.9)
            .build())
        .layer(4,new DenseLayer.Builder()
            .name("ThirdHiddenLayer")
            .nIn(numHiddenNodes).nOut(numHiddenNodes/2)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .momentum(0.9)
            .build())
        .layer(5,new DenseLayer.Builder()
            .name("ThirdHiddenLayer")
            .nIn(numHiddenNodes/2).nOut(numHiddenNodes/2)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .momentum(0.9)
            .build())
        .layer(6,new DenseLayer.Builder()
            .name("ThirdHiddenLayer")
            .nIn(numHiddenNodes/2).nOut(numHiddenNodes/2)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .momentum(0.9)
            .build())
        .layer(7,new DenseLayer.Builder()
            .name("ThirdHiddenLayer")
            .nIn(numHiddenNodes/2).nOut(numHiddenNodes/4)
            .weightInit(WeightInit.XAVIER)
            .activation("relu")
            .momentum(0.9)
            .build())
            .layer(8, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("OutputLayer")
                .weightInit(WeightInit.XAVIER)
                .momentum(0.9)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes/4).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        for ( int n = 0; n < nEpochs; n++) {
            model.fit(train_set);
        }


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);

        INDArray features = test_set.getFeatureMatrix();
        INDArray lables = test_set.getLabels();
        INDArray predicted = model.output(features,false);

        eval.eval(lables, predicted);

        //Print the evaluation statistics
        System.out.println(eval.stats());
        System.out.println("Confusion Matrix: "+eval.getConfusionMatrix());

        // **********Training is done. Code that follows is to save and print the model parameters***************

        OutputStream fos = Files.newOutputStream(Paths.get("ann23.bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File("ann23.json"), model.getLayerWiseConfigurations().toJson());

    }
}
