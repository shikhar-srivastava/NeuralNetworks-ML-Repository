package Weka.RandomForest;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.converters.CSVSaver;

import java.io.*;
import java.util.Random;

/**
 *
 * @ author: Shikhar Srivastava
 * @ Due credit to: Samy
 */
public class RandomForest_Weka {

    /**
     * @param args the command line arguments
     */
    static String modelName="SVM";
    static String dataName="SMOTE";
    public static void main(String[] args) throws Exception {
        BufferedReader br = null;
        int numFolds = 5;
        br = new BufferedReader(new FileReader("D:/Deeplearning4Java/dl4j_0.4_examples/dl4j-examples/src/main/resources/classification/mag_train_smote.arff"));
        BufferedReader testSetReader= new BufferedReader(new FileReader("D:/Deeplearning4Java/dl4j_0.4_examples/dl4j-examples/src/main/resources/classification/mag_test.arff"));
        Instances trainData = new Instances(br);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        Instances testData = new Instances(testSetReader);
        testData.setClassIndex(testData.numAttributes() - 1);
        br.close();
        testSetReader.close();
        double max_f1;
        max_f1 = 0;
        double best_k,best_j;
        best_k=best_j=0;

        /*for(double k=(Math.pow(2,-5));k<=Math.pow(2,15);k=k*4)
        {

            for (double j = Math.pow(2,-15); j <= Math.pow(2,3); j = j *4) {
                LibSVM rf = new LibSVM();
                rf.setGamma(j);
                rf.setCoef0(k);
                rf.buildClassifier(trainData);
                Evaluation evaluation = new Evaluation(trainData);
                evaluation.crossValidateModel(rf,trainData,numFolds,new Random(1));
                if (max_f1 < evaluation.weightedFMeasure()){
                    max_f1 = evaluation.weightedFMeasure();
                    best_j = j;
                    best_k=k;
                }
                System.out.println("F1 score for " + modelName + " with Coef: " + k + " and Gamma: "+j+" is: " + evaluation.weightedFMeasure());
        }
    } */       //System.out.println("\nThe Grid Searched value of No of trees: "+max_j);
        LibSVM rf = new LibSVM();
        //rf.setCoef0(best_k);
        //rf.setGamma(best_j);

        rf.buildClassifier(trainData);
        System.out.println("Kernel: "+rf.getKernelType());
        Evaluation crossEval = new Evaluation(trainData);
        crossEval.crossValidateModel(rf, trainData, numFolds, new Random(1));
        System.out.println("\nUsing GridSearch to optimize SVM parameters in a RBF SVM, using ranges as mentioned of 2^-5 to 2^15 for C and 2^-15 to 2^3 in steps of 2^2\n ");
        System.out.println("\nOptimal C: "+best_k+"\nOptimal Gamma: "+best_j);

        System.out.println("************* Cross Validation Results are: *************");
        System.out.println(crossEval.toSummaryString("\nResults\n======\n", true));
        System.out.println(crossEval.toClassDetailsString());

        Evaluation testEval= new Evaluation(testData);
        testEval.evaluateModel(rf,testData);
        System.out.println("********************TEST SET RESULTS: ******************");
        System.out.println(testEval.toSummaryString("\nResults\n======\n", true));
        System.out.println(testEval.toClassDetailsString());
        double confMat[][] = testEval.confusionMatrix();
        System.out.println("Confusion Matrix : ");
        for(double x[] : confMat)
        {
            for (double y: x)
            {
                System.out.print(y+ " ");
            }
            System.out.println();
        }
        labelPrinter(testData,"labels_"+dataName+modelName+".csv");
        predictedPrinter(rf,testData,"predicted_"+dataName+modelName+".csv");

    }

    public static void filePrinter(Instances toBePrinted, String fileName) throws Exception
    {
        Instances dataSet = toBePrinted;
        CSVSaver saver = new CSVSaver();
        saver.setInstances(dataSet);
        saver.setFile(new File(fileName));
        saver.setDestination(new File(fileName));
        saver.writeBatch();
    }
    public static void labelPrinter(Instances toBePrinted, String fileName) throws Exception
    {
        BufferedWriter outputWriter = null;
        outputWriter = new BufferedWriter(new FileWriter(fileName));
        int length=toBePrinted.numInstances();
        double label;
        for(int j=0;j<length;j++)
        {
            label=toBePrinted.instance(j).classValue();
            outputWriter.write((label)+"");
            if(j!=(length-1))outputWriter.write(',');
        }
        //outputWriter.newLine();

        outputWriter.flush();
        outputWriter.close();
    }
    public static void predictedPrinter(weka.classifiers.Classifier clf,Instances testData, String fileName) throws Exception
    {
        BufferedWriter outputWriter = null;
        outputWriter = new BufferedWriter(new FileWriter(fileName));
        int length=testData.numInstances();
        double label;
        for(int j=0;j<length;j++)
        {
             label=clf.classifyInstance(testData.instance(j));
            outputWriter.write((label)+"");
            if(j!=(length-1))outputWriter.write(',');
        }
        //outputWriter.newLine();

        outputWriter.flush();
        outputWriter.close();
    }
}
