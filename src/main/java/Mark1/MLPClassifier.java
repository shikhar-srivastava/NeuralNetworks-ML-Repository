package Mark1;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;

import org.apache.commons.io.FileUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.feedforward.classification.PlotUtil;
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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifier {


    public static void main(String[] args) throws Exception {
        int type=2;
        int seed = 123;
        double learningRate = 0.005;
        int batchSize = 290;
        int nEpochs = 9001;
        int splitTrainNum = (int) (batchSize * .8);
        int iterations= 5;
        int numInputs = 3;
        int numOutputs = 4;
        int numHiddenNodes = 5;

        //Load the training data:
       /* RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/train_file_norm.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,4);


        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/test_file_norm.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,4);*/
        RecordReader traintest= new CSVRecordReader();
        traintest.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/dataset_1.csv")));
        DataSetIterator traintestIter = new RecordReaderDataSetIterator(traintest,batchSize,0,4);
        DataSet next= traintestIter.next();
        next.shuffle();
        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum,new java.util.Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .updater(Updater.NESTEROVS)
            .list(2)
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(500)));
        for ( int n = 0; n < nEpochs; n++) {
            model.fit( train);
        }


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);

            INDArray features = test.getFeatureMatrix();
            INDArray lables = test.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        //Print the evaluation statistics
        System.out.println(eval.stats());

        // **********Training is done. Code that follows is to save and print the model parameters***************

        OutputStream fos = Files.newOutputStream(Paths.get("coefficients_MLP_org"+type+"_"+nEpochs+".bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File("conf_MLP_org"+type+"_"+nEpochs+".json"), model.getLayerWiseConfigurations().toJson());

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf_MLP_org"+type+"_"+nEpochs+".json")));
        DataInputStream dis = new DataInputStream(new FileInputStream("coefficients_MLP_org"+type+"_"+nEpochs+".bin"));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParams(newParams);
        System.out.println("Original network params: " + model.params());
        System.out.println("Saved network params: " + savedNetwork.params());

    }
}
