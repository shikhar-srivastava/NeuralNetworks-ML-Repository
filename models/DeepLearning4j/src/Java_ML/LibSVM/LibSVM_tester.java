package Java_ML.LibSVM;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Map;


/*
 * @author: Shikhar Srivastava
 * Implementation of LibSVM on Sample Dataset
 * @due credit to: Thomas Abeel for the Example for JAVA ML Library
 *
 */

public class LibSVM_tester {


    public static void main(String[] args) throws Exception {


        //------------De-Serializing the SAVED model---------------------

        ObjectInputStream ois = new ObjectInputStream(
            new FileInputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/LibSVM/svm3.model"));

        Classifier svm = (Classifier) ois.readObject();
        System.out.println(svm.toString());
        ois.close();

        Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_test.csv"), 0, ",");

        //---------Incase of Large Test Set-----------

        int testLength=0;
        for (Instance inst: dataForClassification){
            System.out.println(inst.classValue());
            testLength++;
        }
        Object realClassValue[] = new Object[testLength];
        Object predictedClassValue[] = new Object[testLength];
        int i=0;
        for (Instance inst : dataForClassification) {
            predictedClassValue[i] = svm.classify(inst);
            realClassValue[i] = inst.classValue();
            i++;
        }
        i--;
        for(int k=0;k<i;k++)
        {
            System.out.println("Real Value is : "+realClassValue[k]+" vs Class predicted by Model is: "+predictedClassValue[k]);
        }

       Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(svm, dataForClassification);
        for (Object o : pm.keySet())
            System.out.println(o + ":\nFMeasure: " + pm.get(o).getFMeasure()+"\nPrecision: "+ pm.get(o).getPrecision()+"\nRecall: "+ pm.get(o).getRecall());

        //System.out.println(pm);


    }

}
