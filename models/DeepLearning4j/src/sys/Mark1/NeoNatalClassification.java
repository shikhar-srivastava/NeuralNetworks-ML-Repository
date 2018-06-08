package sys.Mark1;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.springframework.core.io.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;


public class NeoNatalClassification {

static String mainCallCount="apnea_3_scaled";
    public static void main(String[] args) throws Exception {
        String type="2Hidden";
        int seed = 123;
        double learningRate = 0.005;
       // int trainbatchSize = 300;
       // int testbatchSize=50;
        int batchSize=364;
        int nEpochs = 301;
        int splitTrainNum = (int) (batchSize * .8);
        int iterations= 5;
        int numInputs = 15;
        int numOutputs = 2;
        int numHiddenNodes =30;

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("C:/Users/MAHE/Desktop/scientia_sit_potentia/2.Projects/1.NeonatalApneaDetection/Data/apnea_data_2.csv")));
        DataSetIterator traintestIter = new RecordReaderDataSetIterator(rr,batchSize,15,2);


        //Load the test/evaluation data:
        //RecordReader rrTest = new CSVRecordReader();
       // rrTest.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/test_file_norm.csv")));
       // DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,4);*/
       // RecordReader trainReader= new CSVRecordReader();
        //trainReader.initialize(new FileSplit(new File("C:/Users/MAHE/Desktop/scientia_sit_potentia/2.Projects/1.NeonatalApneaDetection/Data/apnea_data_2_train_rbf_pca_14.csv")));
        //DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader,trainbatchSize ,numInputs,numOutputs);
       // RecordReader testReader= new CSVRecordReader();
       // testReader.initialize(new FileSplit(new File("C:/Users/MAHE/Desktop/scientia_sit_potentia/2.Projects/1.NeonatalApneaDetection/Data/.csv")));
       // DataSetIterator testIter = new RecordReaderDataSetIterator(testReader,testbatchSize,numInputs,numOutputs);

        DataSet next= traintestIter.next();
        //next.normalizeZeroMeanZeroUnitVariance();
        StandardScaler stdsc = new StandardScaler();
        stdsc.fit(next);
        stdsc.transform(next);
        //Check the right effect of Normalization on Discrete multi-Class features
        //System.out.println(next);
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum,new java.util.Random(seed));
        DataSet train = testAndTrain.getTrain();
        //train.normalizeZeroMeanZeroUnitVariance();
        DataSet test = testAndTrain.getTest();
        //test.normalizeZeroMeanZeroUnitVariance();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .updater(Updater.NESTEROVS)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(4*numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(4*numHiddenNodes).nOut(3*numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(2, new DenseLayer.Builder().nIn(3*numHiddenNodes).nOut(2*numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(3, new DenseLayer.Builder().nIn(2*numHiddenNodes).nOut(numHiddenNodes)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(4, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes/2)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax").weightInit(WeightInit.XAVIER)
                .nIn(numHiddenNodes/2).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        long startTime = System.nanoTime();
        Evaluation eval = new Evaluation(numOutputs);
        INDArray features = test.getFeatureMatrix();
        INDArray lables = test.getLabels();
        INDArray predicted;

       model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        for ( int n = 0; n < nEpochs; n++) {
            model.fit(train);
            if(n%25==0 && n!=0)
            {
               predicted = model.output(features,false);

                System.out.println("At Epoch No: "+n);
                eval.eval(lables, predicted);
                System.out.println("Confusion Matrix: "+eval.getConfusionMatrix());
                System.out.println("Precision: "+eval.precision() +" Recall: "+eval.recall()+" FNR: "+ eval.falseNegativeRate()+ "FPR: "+ eval.falsePositiveRate());
                System.out.println(eval.stats());
            }
        }
        long endTime = System.nanoTime();

        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
        System.out.println("Evaluate model....");

        //System.out.println("features: "+features+"\nLabels: "+lables+"\nPredicted: "+predicted);
        predicted = model.output(features,false);
        eval.eval(lables, predicted);
        System.out.println("Confusion Matrix: "+eval.getConfusionMatrix());
        System.out.println("Precision: "+eval.precision() +" Recall: "+eval.recall()+" FNR: "+ eval.falseNegativeRate()+ "FPR: "+ eval.falsePositiveRate());
        //Print the evaluation statistics
        System.out.println(eval.stats());
        filePrinter(train.getFeatures(),"train_features.csv");
        filePrinter(train.getLabels(),"train_labels.csv");
        filePrinter(test.getFeatures(),"test_features.csv");
        filePrinter(lables,"lables.csv");
        filePrinter(predicted,"predicted.csv");
        System.out.println("Time Taken: "+duration/1000000);

        // **********Training is done. Code that follows is to save and print the model parameters***************

        OutputStream fos = Files.newOutputStream(Paths.get("dl4j_0.4_examples/src/main/resources/saved_models/Neo_"+mainCallCount+".bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File("dl4j_0.4_examples/src/main/resources/saved_models/Neo_"+mainCallCount+".json"), model.getLayerWiseConfigurations().toJson());

       /* MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf_MLP_org"+type+"_"+nEpochs+".json")));
        DataInputStream dis = new DataInputStream(new FileInputStream("coefficients_MLP_org"+type+"_"+nEpochs+".bin"));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParams(newParams);*/
        System.out.println("Original network params: " + model.params());
       // System.out.println("Saved network params: " + savedNetwork.params());

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
}

/*Iteration 1: Epochs 20000, Iters 1, 10 layers:
F1 Score: 0.6074
AUC: 0.66
Saved as : iter1.json and iter1.bin*/

//Type 4: linear PCA: Disaster
//Type 5: RBF PCA  :Disaster
