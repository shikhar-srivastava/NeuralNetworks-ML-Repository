package Java_ML.NaiveBayes;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.filter.discretize.EqualWidthBinning;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;


/*
 * @author: Shikhar Srivastava
 * Implementation of NaiveBayes on Sample Dataset
 * @due credit to: Thomas Abeel for the Example for JAVA ML Library
 *
 */

public class NaiveBayes_tester {

    public static void main(String[] args) throws Exception {


        //------------De-Serializing the SAVED model---------------------

        ObjectInputStream ois = new ObjectInputStream(
            new FileInputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/NaiveBayes/nb.model"));
        Classifier nb = (Classifier) ois.readObject();
        ois.close();
        EqualWidthBinning eb = new EqualWidthBinning(20);

        Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/test_file.csv"), 0, ",");

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
            eb.filter(inst);
            predictedClassValue[i] = nb.classify(inst);
            realClassValue[i] = inst.classValue();
            i++;
        }
        i--;
        for(int k=0;k<i;k++)
        {
            System.out.println("Real Value is : "+realClassValue[k]+" vs Class predicted by Model is: "+predictedClassValue[k]);
        }

       /* Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(nb, dataForClassification);
        for (Object o : pm.keySet())
            System.out.println(o + ":\nAccuracy: " + pm.get(o).getAccuracy()+"\nTPR: "+pm.get(o).getTPRate()+"\nFPR: "+ pm.get(o).getFPRate()+ "\nTNR: "+pm.get(o).getTNRate()+"\nFNR: "+pm.get(o).getFNRate());

        System.out.println(pm);*/


    }

}
