package sys.Mark2;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;

/**
 * @ author- Shikhar Srivastava
 * @ credit due to : Relevant Authors of DeepLearning4J Library
 *
 */
public class MLPClassifier_tester {


    public static void main(String[] args) throws Exception {
       /*
        String type="Mark2_ver0.1";
        int seed = 123;
        double learningRate = 0.01;
        int batchSize_train = 504;

        int nEpochs =9001;
       // int splitTrainNum = (int) (batchSize * .8);
        int iterations= 1;
        int numInputs = 23;
        int numOutputs = 6;
        int numHiddenNodes = 10;

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/train_file_norm.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,4);


        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/test_file_norm.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,4);
        RecordReader train= new CSVRecordReader();
        train.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/smote_k1_ann.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(train,batchSize_train,0,6);
        DataSet train_set= trainIter.next();
        train_set.shuffle();
        train_set.normalizeZeroMeanZeroUnitVariance();
        //System.out.println(train_set);
*/

        //test_set.shuffle();
        // test_set.normalizeZeroMeanZeroUnitVariance();

        /*SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum,new java.util.Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();*/
        //  Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
    /*
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
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
                .nIn(numHiddenNodes).nOut(numHiddenNodes/2)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .momentum(0.9)
                .build())
            .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
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
            model.fit(train_set);
        }


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
*/
   /*     int batchSize_test= 270;
        RecordReader test= new CSVRecordReader();
        test.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_test.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(test,batchSize_test,0,6);
        DataSet test_set= testIter.next();
*/
        double arr[]={25.86,19.52,14.734,0.614,2621.006,5.738,8.231,29.296,0.883,0.73,3.436,-36.16,80872.587,-67260.671,55368.369,3.089,1.756011088,6.431928831,7.52305201,10.48130719,19.88813984,15.57534177,19.62694059};
        double out[]={0,0,0,0,1,0};
        INDArray outs=Nd4j.create(out);
        INDArray tester= Nd4j.create(arr);
        DataSet add= new DataSet();
        add.setFeatures(tester);
        add.setLabels(outs);
        System.out.println(add);
        System.out.println("Orignal Data Above");
        RecordReader testSet= new CSVRecordReader();
        testSet.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/Feature_test_ann.csv")));
        DataSetIterator testIter = (DataSetIterator) new RecordReaderDataSetIterator(testSet,269,0,6);
        DataSet test_set= testIter.next();
        test_set.addRow(add,268);
        System.out.println(test_set);
        test_set.normalizeZeroMeanZeroUnitVariance();
        System.out.println(test_set);
        System.out.println("Normalized Full TestSet above:");
        add=test_set.get(268);
        System.out.println("Orignal Data Is Now: ");
        System.out.println(add);
        /*
        StandardScaler s3= new StandardScaler();
        s3.load(new File("C:/apache-tomcat-8.0.33/webapps/MachineLearningForMedicalDataSets/models/s23mean.bin"),new File("C:/apache-tomcat-8.0.33/webapps/MachineLearningForMedicalDataSets/models/s23std.bin"));
        s3.transform(d);
        System.out.println("Mean was: "+s3.getMean()+" Std. Dev: "+s3.getStd());
        System.out.println(d);*/

        INDArray features=add.getFeatures();
        System.out.println(features);

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("D:/Deeplearning4Java/conf_MLP_orgMark2_ver0.1_8001.json")));
        DataInputStream dis = new DataInputStream(new FileInputStream("D:/Deeplearning4Java/coefficients_MLP_orgMark2_ver0.1_8001.bin"));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork model = new MultiLayerNetwork(confFromJson);
        model.init();
        model.setParams(newParams);

        INDArray predicted = model.output(features,false);
        int j=0;
        while(j<6)
        {
            if(predicted.getInt(j)==1)
                System.out.println("The class predicted: "+j);
            j++;
        }


    }
}
