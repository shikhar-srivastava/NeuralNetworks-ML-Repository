package sys;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.DataOutputStream;
import java.io.File;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;


/* @Author: Shikhar Srivastava
    @Due credit to : Adam gibson (deeplearning4j)
 */


public class AutoEncoderHne {

    //Some globally required variables here

    private static int mainCaller=0;
   // protected static Logger log = LoggerFactory.getLogger(AutoEncoderHne.class);
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static final long seed = 123;
    public static final java.util.Random randNumGen = new java.util.Random(seed);
    protected static int height = 61;
    protected static int width = 61;
    protected static int channels = 3;
    //protected static int numExamples = 80;
    protected static int outputNum = 2;



    public static void main(String[] args) throws Exception {


        mainCaller++;  //To keep track of the calls to this function
        final int numRows = 61;
        final int numColumns = 61;
        int batchSize = 100; //Check for correct Optimization
        int iterations = 40; // DEFINITELY OPTIMIZE THIS incase error is high at End-Of-Training
        int listenerFreq = batchSize / 5;

        final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;




        System.out.println("**********Reading the Datset now****************");
        /*File parentDir = new File(System.getProperty("user.dir"), "src/main/resources/classification/reduced/");
        FileSplit filesInDir = new FileSplit(parentDir,allowedExtensions,randNumGen); // **Check if randomizing the dataset split is Correct
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen,allowedExtensions,labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(trainData);
        ImageRecordReader recordReaderTest = new ImageRecordReader(height,width,channels,labelMaker);
        recordReaderTest.initialize(testData);*/

        //Iterators here:
        //DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        //DataSetIterator testIter = new RecordReaderDataSetIterator(recordReaderTest,batchSize,1,outputNum);

       //The sys.AutoEncoder Structure here:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_test.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,1,0,2);

        RecordReader rrTrain= new CSVRecordReader();
        rrTrain.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_train.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrTrain,339,0,2);



        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
            .gradientNormalizationThreshold(1.0)
            .weightInit(WeightInit.XAVIER)
            .iterations(iterations)
            .momentum(0.5)
             .learningRate(0.1)
            //.momentumAfter(Collections.singletonMap(3, 0.9))
            .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS)
            .list()
            .layer(0, new AutoEncoder.Builder().nIn(3).nOut(500)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .corruptionLevel(0.2)
                .updater(Updater.NESTEROVS)
                .build())
            .layer(1, new AutoEncoder.Builder().nIn(500).nOut(200)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .corruptionLevel(0.2)
                .updater(Updater.NESTEROVS)
                .build())
            .layer(2, new AutoEncoder.Builder().nIn(200).nOut(50)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .corruptionLevel(0.2)
                .updater(Updater.NESTEROVS)
                .build())
            .layer(3, new DenseLayer.Builder().nIn(50).nOut(20)
                .updater(Updater.NESTEROVS)
                .weightInit(WeightInit.XAVIER)
                .momentum(0.9)
                .activation("relu")
                .build())
            .layer(4, new DenseLayer.Builder().nIn(20).nOut(5)
                    .updater(Updater.NESTEROVS)
                    .weightInit(WeightInit.XAVIER)
                    .momentum(0.9)
                    .activation("relu")
                .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation("softmax")
                .nIn(5).nOut(outputNum).build())
            .pretrain(true).backprop(true)
            .build();
        //double lr=new NeuralNetConfiguration.Builder().getLearningRate();
       // System.out.println("The learning rate is: "+ lr);
        //Definitely Change the Network structure. It needs Optimization
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
        System.out.println("Train model....");
        model.fit(trainIter);
        System.out.println("Finishing first training...");// Pre-Training and fine tuning
        trainIter.reset();
        int k=0;
        while(trainIter.hasNext())
         {
             k++;
             System.out.println("This is  Batch: "+k);
            DataSet next = trainIter.next();
             //System.out.println(next);
            model.fit(next);
        }
        System.out.println("Finishing second training...");// Pre-Training and fine tuning
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray predict = model.output(next.getFeatureMatrix());
            System.out.println(next.getLabels());
            eval.eval(next.getLabels(), predict);
        }
        System.out.println(eval.stats());
        System.out.println("****************Example finished********************");

        OutputStream fos = Files.newOutputStream(Paths.get("RGBAutoEncoder"+".bin"));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File("RGBAutoEncoder"+".json"), model.getLayerWiseConfigurations().toJson());

    }
}

/*Iteration 1: Without extra training*,Iters:30
   Iteration 2: With extra training while loop, Iters:30
   Iteration 3: With extra training, Iter: 50
   Iteration 4: Without extra training, Iter: 50

   Iteration 1: Batchsize 1, Iters: 5, Noise: 0.2
   Iteration 2: Batchsize:100,Iters: 2,Noise 0.2

   Trial: Stochastic Gradient
 */
