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


public class MLP_Shallow {

    static String mainCallCount="apnea_all_features_6layers_relu_sgd_";
    public static void main(String[] args) throws Exception {
        String type="2Hidden";
        int seed = 123;
        double learningRate = 0.005;
        // int trainbatchSize = 300;
        // int testbatchSize=50;
        int batchSize=364;
        int nEpochs = 5001;
        int splitTrainNum = (int) (batchSize * .8);
        int iterations= 5;
        int numInputs = 22;
        int numOutputs = 2;
        int numHiddenNodes =30;
        //double regularizationParam=0.001;
        //mainCallCount+=Double.toString(regularizationParam);
        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("C:\\Users\\MAHE\\Desktop\\scientia_sit_potentia\\coursera-notebook\\IITKGP\\apnea_data_2_allfeaturesunedited.csv")));
        DataSetIterator traintestIter = new RecordReaderDataSetIterator(rr,batchSize,22,2);

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
        next.normalize();
        /*StandardScaler stdsc = new StandardScaler();
        stdsc.fit(next);
        stdsc.transform(next);*/
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
                filePrinter(predicted,"predicted"+mainCallCount+Integer.toString(step)+".csv");
                step+=25;
            }
        }
        arrayPrinter(scoreArray,"scores"+mainCallCount+".csv",len);
        long endTime = System.nanoTime();
        filePrinter(lables,"lables"+mainCallCount+".csv");
        System.out.println("\nMAX F1 Score: "+max_f1_score+ " At Iteration : "+max_n);
        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
        System.out.println("Evaluate model....");

        //System.out.println("features: "+features+"\nLabels: "+lables+"\nPredicted: "+predicted);
        predicted = model.output(features,false);
        eval.eval(lables, predicted);
        System.out.println("Confusion Matrix: "+eval.getConfusionMatrix());
        System.out.println("Precision: "+eval.precision() +" Recall: "+eval.recall()+" FNR: "+ eval.falseNegativeRate()+ "FPR: "+ eval.falsePositiveRate());
        //Print the evaluation statistics
        System.out.println(eval.stats());
        //filePrinter(train.getFeatures(),"train_features"+mainCallCount+".csv");
        //filePrinter(train.getLabels(),"train_labels"+mainCallCount+".csv");
       // filePrinter(test.getFeatures(),"test_features"+mainCallCount+".csv");
        //filePrinter(lables,"lables"+mainCallCount+".csv");
        filePrinter(predicted,"predicted"+mainCallCount+".csv");
        System.out.println("Time Taken: "+duration/1000000);

        // **********Training is done. Code that follows is to save and print the model parameters***************

        /*OutputStream fos = Files.newOutputStream(Paths.get("dl4j_0.4_examples/src/main/resources/saved_models/Neo_"+mainCallCount+".bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File("dl4j_0.4_examples/src/main/resources/saved_models/Neo_"+mainCallCount+".json"), model.getLayerWiseConfigurations().toJson());
            */
       /* MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf_MLP_org"+type+"_"+nEpochs+".json")));
        DataInputStream dis = new DataInputStream(new FileInputStream("coefficients_MLP_org"+type+"_"+nEpochs+".bin"));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParams(newParams);*/
       // System.out.println("Original network params: " + model.params());
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
