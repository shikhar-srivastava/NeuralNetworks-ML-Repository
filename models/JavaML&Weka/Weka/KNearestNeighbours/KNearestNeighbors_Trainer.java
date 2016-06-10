package Weka.KNearestNeighbours;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.Random;

/**
 *
 * @ author: Shikhar Srivastava
 * @ Due credit to: Samy
 */
public class KNearestNeighbors_Trainer {

    public static void main(String[] args) throws Exception {
        BufferedReader br = null;
        int numFolds = 10;
        br = new BufferedReader(new FileReader("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_weka_3_train.arff"));
        Instances trainData = new Instances(br);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        br.close();
        IBk knn = new IBk();
        knn.buildClassifier(trainData);
        //------Saving Model Here------
        ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/Weka/KNearestNeighbours/knnweka.xml"));
        oos.writeObject(knn);
        oos.flush();
        oos.close();
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.crossValidateModel(knn, trainData, numFolds, new Random(1));

        System.out.println(evaluation.toSummaryString("\nResults\n======\n", true));
        System.out.println(evaluation.toClassDetailsString());
        System.out.println("Results For Class -0- ");
        System.out.println("Precision=  " + evaluation.precision(0));
        System.out.println("Recall=  " + evaluation.recall(0));
        System.out.println("F-measure=  " + evaluation.fMeasure(0));
        System.out.println("Results For Class -1- ");
        System.out.println("Precision=  " + evaluation.precision(1));
        System.out.println("Recall=  " + evaluation.recall(1));
        System.out.println("F-measure=  " + evaluation.fMeasure(1));
        System.out.println("Results For Class -2- ");
        System.out.println("Precision=  " + evaluation.precision(2));
        System.out.println("Recall=  " + evaluation.recall(2));
        System.out.println("F-measure=  " + evaluation.fMeasure(2));


    }
}
