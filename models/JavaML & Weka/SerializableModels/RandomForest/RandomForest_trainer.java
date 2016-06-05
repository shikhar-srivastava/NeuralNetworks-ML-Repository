package Java_ML.RandomForest;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Map;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

/*
 * This emulation is referenced from a tutorial by Thomas Abeel
 * @author Shikhar Srivastava
 * @Due credit to: Thomas Abeel for the JAVA ML Tutorial
 *
 */
/* As per Observation, as no. of trees increases, the maximum accuracy for all classes is
    0:
    Accuracy: 0.9444444444444444
    1:
    Accuracy: 0.9444444444444444
    2:
    Accuracy: 1.0
    3:
    Accuracy: 1.0
*/
public class RandomForest_trainer{


    public static void main(String[] args) throws Exception { //---------Train the Required Model---------------

        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/train_file.csv"), 0, ",");
        Classifier rff = new net.sf.javaml.classification.tree.RandomForest(10);
        rff.buildClassifier(data);

        //-----------------#Model is now Trained. Serializing Model: SAVE #--------------------

        ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/Java_ML/RandomForest/rff.model"));
        oos.writeObject(rff);
        oos.flush();
        oos.close();

        //---------Incase of Large Test Set-----------

        Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/test_file.csv"), 0, ",");


        int testLength=0;
        for (Instance inst: dataForClassification){
            System.out.println(inst.classValue());
            testLength++;
        }
        Object realClassValue[] = new Object[testLength];
        Object predictedClassValue[] = new Object[testLength];
        int i=0;
        for (Instance inst : dataForClassification) {
            predictedClassValue[i] = rff.classify(inst);
            realClassValue[i] = inst.classValue();
            i++;
        }
        i--;
        for(int k=0;k<i;k++)
        {
            System.out.println("Real Value is : "+realClassValue[k]+" vs Class predicted by Model is: "+predictedClassValue[k]);
        }

       /* Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(rff, dataForClassification);
        for (Object o : pm.keySet())
            System.out.println(o + ":\nAccuracy: " + pm.get(o).getAccuracy()+"\nTPR: "+pm.get(o).getTPRate()+"\nFPR: "+ pm.get(o).getFPRate()+ "\nTNR: "+pm.get(o).getTNRate()+"\nFNR: "+pm.get(o).getFNRate());

        System.out.println(pm);*/

    }

}
