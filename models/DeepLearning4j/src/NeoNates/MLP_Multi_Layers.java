package NeoNates;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Collections;


/* Dataset: apnea.csv 17 columns Label at last column
   Dataset Size: 314

*/
public class MLP_Multi_Layers{
    static String modelName="8H_RELU";
    static String mainCallCount = "D:/documentation/"+modelName+"/"+modelName+"_";

    public static void main(String[] args) throws Exception{


         boolean filecreate = new File("D:/documentation/" + modelName).mkdirs();
         if (!filecreate)
             System.out.println("\nDirectory not Created!");

        //String type="4Hidden";
        int seed = 123; //1234
        double learningRate = 0.01;
        int batchSize=364;  //new Size 314 Old 364
        int nEpochs = 5001;
        int splitTrainNum = (int) (batchSize * .70);
        int iterations= 5;
        int numInputs = 22; // 16 Inputs
        int numOutputs = 2;
        //int numHiddenNodes =35;

        //double regularizationParam=0.001;
        //mainCallCount+=Double.toString(regularizationParam);

        RecordReader rr = new CSVRecordReader();
        //rr.initialize(new FileSplit(new File("apnea.csv")));
        rr.initialize(new FileSplit(new File("C:\\Users\\MAHE\\Desktop\\scientia_sit_potentia\\coursera-notebook\\IITKGP\\apnea_data_2_allfeaturesunedited.csv")));
        DataSetIterator traintestIter = new RecordReaderDataSetIterator(rr,batchSize,numInputs,2);

        DataSet next= traintestIter.next();
        next.normalize();
        /*StandardScaler stdsc = new StandardScaler();
        stdsc.fit(next);
        stdsc.transform(next);*/
        //Check the right effect of Normalization on Discrete multi-Class features
        //System.out.println(next);
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum,new java.util.Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test =  testAndTrain.getTest();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
            .learningRate(learningRate)
            .updater(Updater.NESTEROVS)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numInputs*32)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(numInputs*32).nOut(numInputs*16)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(2, new DenseLayer.Builder().nIn(numInputs*16).nOut(numInputs*8)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
             .layer(3, new DenseLayer.Builder().nIn(numInputs*8).nOut(numInputs*4)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(4, new DenseLayer.Builder().nIn(numInputs*4).nOut(numInputs*2)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(5, new DenseLayer.Builder().nIn(numInputs*2).nOut(numInputs)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(6, new DenseLayer.Builder().nIn(numInputs).nOut(numInputs/2)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(7, new DenseLayer.Builder().nIn(numInputs/2).nOut(numInputs/4)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .momentum(0.9)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numInputs/4).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();

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
        double scoreArray[]= new double[nEpochs*iterations];
        for ( int n = 0; n < nEpochs; n++) {
            model.fit(train);
            scoreArray[len++]=model.score();
            if(n%25==0 && n!=0)
            {
                predicted = model.output(features,false);
                Evaluation eval2 = new Evaluation(numOutputs);
                System.out.println("At Epoch No: "+n);
                eval2.eval(lables, predicted);
                if(eval2.f1()>max_f1_score)
                {
                    max_f1_score=eval2.f1();
                    max_n=n;
                }
                System.out.println("Confusion Matrix: "+eval2.getConfusionMatrix());
                System.out.println("Precision: "+eval2.precision() +" Recall: "+eval2.recall()+" FNR: "+ eval2.falseNegativeRate()+ "FPR: "+ eval2.falsePositiveRate());
                System.out.println(eval2.stats());
                filePrinter(predicted,mainCallCount+"predicted"+Integer.toString(step)+".csv");
                step+=25;
            }
        }
        arrayPrinter(scoreArray,mainCallCount+"scores"+".csv",len);
        long endTime = System.nanoTime();
        filePrinter(lables,mainCallCount+"lables"+".csv");

        //System.out.println("features: "+features+"\nLabels: "+lables+"\nPredicted: "+predicted);
        predicted = model.output(features,false);
        eval.eval(lables, predicted);
        System.out.println("Confusion Matrix: "+eval.getConfusionMatrix());
        System.out.println("Precision: "+eval.precision() +" Recall: "+eval.recall()+" FNR: "+ eval.falseNegativeRate()+ "FPR: "+ eval.falsePositiveRate());
        //Print the evaluation statistics
        System.out.println(eval.stats());
        System.out.println("\nMAX F1 Score: "+max_f1_score+ " At Iteration : "+max_n);
        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
        System.out.println("Evaluate model....");
        //filePrinter(train.getFeatures(),"train_features"+mainCallCount+".csv");
        //filePrinter(train.getLabels(),"train_labels"+mainCallCount+".csv");
        //filePrinter(test.getFeatures(),"test_features"+mainCallCount+".csv");
        //filePrinter(lables,"lables"+mainCallCount+".csv");

        filePrinter(predicted,mainCallCount+"predicted"+".csv");
        System.out.println("Time Taken: "+duration/1000000);

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
