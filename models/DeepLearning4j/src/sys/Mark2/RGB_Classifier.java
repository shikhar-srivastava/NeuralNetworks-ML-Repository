package Mark2;

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
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.DataOutputStream;
import java.io.File;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;

/**
 * @ author- Shikhar Srivastava
 * @ credit due to : Relevant Authors of DeepLearning4J Library
 */
public class RGB_Classifier {


    public static void main(String[] args) throws Exception {
        int seed = 231;
            double learningRate = 0.005;
        int nEpochs = 9501;
        int iterations= 1;
        int numInputs = 3;
        int numOutputs = 3;
        int numHiddenNodes = 20;

        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_test.csv")));
        DataSetIterator testIter =(DataSetIterator)  new RecordReaderDataSetIterator(rrTest,118,0,3);
        DataSet test_set=testIter.next();

        RecordReader rrTrain= new CSVRecordReader();
        rrTrain.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_train.csv")));
        DataSetIterator trainIter =(DataSetIterator)  new RecordReaderDataSetIterator(rrTrain,482,0,3);
        DataSet train_set= trainIter.next();

       // train_set.shuffle();
      // train_set.normalizeZeroMeanZeroUnitVariance();
     //   test_set.shuffle();
      // test_set.normalizeZeroMeanZeroUnitVariance();


        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
            .learningRate(learningRate)
            // .l1(1e-1).regularization(true).l2(2e-4)
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
                .name("SecondHiddenLayer")
                .nIn(numHiddenNodes).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(3,new DenseLayer.Builder()
                .name("SecondHiddenLayer")
                .nIn(numHiddenNodes).nOut(numHiddenNodes/2)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("OutputLayer")
                .weightInit(WeightInit.XAVIER)
                .momentum(0.9)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes/2).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        for ( int n = 0; n < nEpochs; n++) {
            model.fit( train_set);
        }


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);

        INDArray features = test_set.getFeatureMatrix();
        INDArray lables = test_set.getLabels();
        INDArray predicted = model.output(features,false);

        eval.eval(lables, predicted);

        //Print the evaluation statistics
        System.out.println(eval.stats());

        // **********Training is done. Code that follows is to save and print the model parameters***************

        OutputStream fos = Files.newOutputStream(Paths.get("ann3.bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File("ann3.json"), model.getLayerWiseConfigurations().toJson());

    }
}
