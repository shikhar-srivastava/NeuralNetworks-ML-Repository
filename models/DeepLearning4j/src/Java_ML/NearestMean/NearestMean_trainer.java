package Java_ML.NearestMean;

import libsvm.SelfOptimizingLinearLibSVM;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Map;


/*
 * @author: Shikhar Srivastava
 * Implementation of NearestMean classifier
 * @due credit to: Thomas Abeel for the JAVA ML Library & helper codes
 *
 */

public class NearestMean_trainer {
    /**
     * Shows the default usage of the KNN algorithm.
     */
    public static void main(String[] args) throws Exception {
        /* Load a data set */


        //---------Train the Required Model---------------

        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_2_train.csv"), 0, ",");
        Classifier nm = new SelfOptimizingLinearLibSVM(1,2);
        nm.buildClassifier(data);

        //-----------------#Model is now Trained. Serializing Model: SAVE #--------------------

        ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/NearestMean/nmc.bin"));
        oos.writeObject(nm);
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
            predictedClassValue[i] = nm.classify(inst);
            realClassValue[i] = inst.classValue();
            i++;
        }
        i--;
        for (int m = 0; m< i; m++) {
            System.out.println("Real Value is : " + realClassValue[m] + " vs Class predicted by Model is: " + predictedClassValue[m]);
        }
        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(nm, dataForClassification);
        for (Object o : pm.keySet()) {
            System.out.println(o + ":\nFMeasure: " + pm.get(o).getFMeasure() + "\nTPR: " + pm.get(o).getTPRate() + "\nFPR: " + pm.get(o).getFPRate() + "\nTNR: " + pm.get(o).getTNRate() + "\nFNR: " + pm.get(o).getFNRate());
        }




    }

}
