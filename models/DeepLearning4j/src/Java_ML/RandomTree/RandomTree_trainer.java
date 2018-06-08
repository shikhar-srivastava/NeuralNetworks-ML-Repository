package Java_ML.RandomTree;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.classification.tree.RandomTree;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import java.util.Random;


/*
 * @author: # Shikhar Srivastava
 * Implementation of RandomTree on Sample Dataset
 * @due credit to: Thomas Abeel for the JAVA ML Library
 *
 */

public class RandomTree_trainer {
    /**
     * Shows the default usage of the KNN algorithm.
     */
    public static void main(String[] args) throws Exception {
        /* Load a data set */


        //---------Train the Required Model---------------

        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_2_train.csv"), 0, ",");
        Classifier rt = new RandomTree(3,new Random(12345));
        rt.buildClassifier(data);

        //-----------------#Model is now Trained. Serializing Model: SAVE #--------------------

        ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/RandomTree/rnt.bin"));
        oos.writeObject(rt);
        oos.flush();
        oos.close();

        //---------Incase of Large Test Set-----------

        Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_2_test.csv"), 0, ",");


        int testLength = 0;
        for (Instance inst : dataForClassification) {
            System.out.println(inst.classValue());
            testLength++;
        }
        Object realClassValue[] = new Object[testLength];
        Object predictedClassValue[] = new Object[testLength];
        int i = 0;
        for (Instance inst : dataForClassification) {
            predictedClassValue[i] = rt.classify(inst);
            realClassValue[i] = inst.classValue();
            i++;
        }
        i--;
        for (int m = 0; m< i; m++) {
            System.out.println("Real Value is : " + realClassValue[m] + " vs Class predicted by Model is: " + predictedClassValue[m]);
        }
        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(rt, dataForClassification);
        for (Object o : pm.keySet()) {
            System.out.println(o + ":\nFMeasure: " + pm.get(o).getFMeasure() + "\nTPR: " + pm.get(o).getTPRate() + "\nFPR: " + pm.get(o).getFPRate() + "\nTNR: " + pm.get(o).getTNRate() + "\nFNR: " + pm.get(o).getFNRate());
        }




    }

}
