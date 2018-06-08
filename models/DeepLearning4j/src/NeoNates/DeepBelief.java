package NeoNates;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Collections;


public class DeepBelief {


    static String mainCallCount="DeepBelief_";
    public static void main(String[] args) throws Exception {
        //String type="2Hidden";
        int seed = 123;
        // double learningRate = 0.001;
        // int trainbatchSize = 300;
        // int testbatchSize=50;
        int batchSize=364;
        //int nEpochs = 5;  //5001
        int splitTrainNum = (int) (batchSize * .8);
        int iterations= 1001;  // 5
        int numInputs = 22;
        int numOutputs = 2;

        //double regularizationParam=0.001;
        //mainCallCount+=Double.toString(regularizationParam);
        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("C:\\Users\\MAHE\\Desktop\\scientia_sit_potentia\\coursera-notebook\\IITKGP\\apnea_data_2_allfeaturesunedited.csv")));
        DataSetIterator traintestIter = new RecordReaderDataSetIterator(rr,batchSize,numInputs,numOutputs);


        DataSet next= traintestIter.next();
        next.normalize();


        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum,new java.util.Random(seed));
        DataSet train = testAndTrain.getTrain();

        DataSet test = testAndTrain.getTest();

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
            .list()
            .layer(0, new RBM.Builder().nIn(numInputs).nOut(numInputs*4).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(1, new RBM.Builder().nIn(numInputs*4).nOut(numInputs*2).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(2, new RBM.Builder().nIn(numInputs*2).nOut(numInputs).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(3, new RBM.Builder().nIn(numInputs).nOut(numInputs/2).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(4, new RBM.Builder().nIn(numInputs/2).nOut(numInputs/4).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //encoding stops
            .layer(5, new RBM.Builder().nIn(numInputs/4).nOut(numInputs/2).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //decoding starts
            .layer(6, new RBM.Builder().nIn(numInputs/2).nOut(numInputs).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(7, new RBM.Builder().nIn(numInputs).nOut(numInputs*2).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(8, new RBM.Builder().nIn(numInputs*2).nOut(numInputs*4).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(numInputs*4).nOut(2).build())
            .pretrain(true).backprop(true)
            .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        Evaluation eval = new Evaluation(numOutputs);
        INDArray features = test.getFeatureMatrix();
        INDArray lables = test.getLabels();
        INDArray predicted;
        int max_n=0;
        double max_f1_score=-1;
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(100)));
        long startTime = System.nanoTime();
        int step=0;
        int len=0;

        model.fit(train);
        predicted = model.output(features,false);
        Evaluation eval2 = new Evaluation(numOutputs);
        eval2.eval(lables, predicted);
        System.out.println("Confusion Matrix: "+eval2.getConfusionMatrix());
        System.out.println("Precision: "+eval2.precision() +" Recall: "+eval2.recall()+" FNR: "+ eval2.falseNegativeRate()+ "FPR: "+ eval2.falsePositiveRate());
        System.out.println(eval2.stats());
        filePrinter(predicted,"predicted"+mainCallCount+".csv");
        long endTime = System.nanoTime();
        filePrinter(lables,"lables"+mainCallCount+".csv");
        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.


    }


    public static void filePrinter(INDArray toBePrinted,String fileName) throws Exception
    {
        BufferedWriter outputWriter = null;
        outputWriter = new BufferedWriter(new FileWriter(fileName));
        int shape[]=toBePrinted.shape();
        for (int i = 0; i < shape[0]; i++) {
            for(int j=0;j<shape[1];j++)
            {
                outputWriter.write((toBePrinted.getFloat(i,j))+"");
                if(j!=(shape[1]-1))outputWriter.write(',');
            }
            outputWriter.newLine();

        }
        outputWriter.flush();
        outputWriter.close();
    }
    public static void arrayPrinter(double errorRate[], String fileName, int length) throws Exception
    {
        BufferedWriter outputWriter = null;
        outputWriter = new BufferedWriter(new FileWriter(fileName));
        for(int j=0;j<length;j++)
        {
            outputWriter.write((errorRate[j])+"");
            if(j!=(length-1))outputWriter.write(',');
        }
        //outputWriter.newLine();

        outputWriter.flush();
        outputWriter.close();
    }
}
