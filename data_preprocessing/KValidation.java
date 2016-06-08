package Java_ML.KNearestNeighbours;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.util.Map;


/*
 * @author: Shikhar Srivastava
 * Implementation of KNearestNeighbour on Sample Dataset
 * @due credit to: Thomas Abeel for the Example for JAVA ML Library
 *
 */

public class KValidation {
    /**
     * Shows the default usage of the KNN algorithm.
     */
    public static void main(String[] args) throws Exception {
        /* Load a data set */


        //---------Train the Required Model---------------
        int k=1;
        double max_prod=0;
        int optimal_k=1;
        double prod=1;
        int iter_limit=150;
        double k_vals[]= new double[iter_limit];
        while(k<iter_limit) {
            Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/Feature_train.csv"), 0, ",");
            Classifier knn = new KNearestNeighbors(k);
            knn.buildClassifier(data);

            //-----------------#Model is now Trained. Serializing Model: SAVE #--------------------

            /*ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/Java_ML/KNearestNeighbours/knn.model"));
            oos.writeObject(knn);
            oos.flush();
            oos.close();*/

            //---------Incase of Large Test Set-----------

            Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/Feature_test.csv"), 0, ",");


            /*int testLength = 0;
            for (Instance inst : dataForClassification) {
                System.out.println(inst.classValue());
                testLength++;
            }
            Object realClassValue[] = new Object[testLength];
            Object predictedClassValue[] = new Object[testLength];
            int i = 0;
            for (Instance inst : dataForClassification) {
                predictedClassValue[i] = knn.classify(inst);
                realClassValue[i] = inst.classValue();
                i++;
            }
            i--;*/
            /*for (int m = 0; m< i; m++) {
                System.out.println("Real Value is : " + realClassValue[m] + " vs Class predicted by Model is: " + predictedClassValue[m]);
            }*/
            prod=1;
            Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, dataForClassification);
            for (Object o : pm.keySet()) {
                //System.out.println(o + ":\nFMeasure: " + pm.get(o).getFMeasure() + "\nTPR: " + pm.get(o).getTPRate() + "\nFPR: " + pm.get(o).getFPRate() + "\nTNR: " + pm.get(o).getTNRate() + "\nFNR: " + pm.get(o).getFNRate());
                System.out.println("FMeasure: "+pm.get(o).getFMeasure());
                prod*=Math.sqrt(pm.get(o).getFMeasure())+1;
            }
            k_vals[k]=prod;
            System.out.println("Product at K: "+k+" is: "+prod);
            if (max_prod<prod)
            {
                max_prod=prod;
                optimal_k=k;
            }
            k++;
        }

        System.out.println("Max prod: "+max_prod);
        System.out.println("Optimal value of K: "+optimal_k);

    }

}
