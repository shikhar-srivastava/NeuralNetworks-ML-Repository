package Java_ML.KNearestNeighbours;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

import java.io.*;
import java.util.Map;


/*
 * @author: Shikhar Srivastava
 * Implementation of KNearestNeighbour on Sample Dataset
 * @due credit to: Thomas Abeel for the Example for JAVA ML Library
 */

public class KNearestNeighbours_Tester {


    public static void main(String[] args) throws Exception {


        //------------De-Serializing the SAVED model---------------------

        InputStream is = new ObjectInputStream(
            new FileInputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/KNearestNeighbours/knn3.bin"));
        ObjectInputStream ois= (ObjectInputStream) is ;
        Object currentObj=ois.readObject();
        System.out.println(currentObj.getClass());
        Classifier knn = (Classifier)currentObj;
        ois.close();
        System.out.println(knn.toString());
        Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_test.csv"),0,",");
    /*
        int testLength=0;
        for (Instance inst: dataForClassification){
            System.out.println(inst.classValue());
            testLength++;
        }
        Object realClassValue[] = new Object[testLength];
        Object predictedClassValue[] = new Object[testLength];
        int i=0;
        for (Instance inst : dataForClassification) {
            predictedClassValue[i] = knn.classify(inst);
            realClassValue[i] = inst.classValue();
            i++;
        }
        i--;
        for(int k=0;k<i;k++)
        {
            System.out.println("Real Value is : "+realClassValue[k]+" vs Class predicted by Model is: "+predictedClassValue[k]);
        }*/

        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, dataForClassification);
       for (Object o : pm.keySet())
          System.out.println(o + ":\nFMeasure: " + pm.get(o).getFMeasure()+"\nPrecision: "+ pm.get(o).getPrecision()+"\nRecall: "+ pm.get(o).getRecall());

        //System.out.println(pm.getCost());


    }

}
