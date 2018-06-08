package Java_ML.RandomForest;

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

        Dataset data = FileHandler.loadDataset(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_2_train.csv"), 0, ",");
        Classifier rff = new net.sf.javaml.classification.tree.RandomForest(300);
        rff.buildClassifier(data);
        //EqualWidthBinning eb = new EqualWidthBinning(20);
        //-----------------#Model is now Trained. Serializing Model: SAVE #--------------------

        ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream("dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/RandomForest/rff.bin"));
        oos.writeObject(rff);
        oos.flush();
        oos.close();

        //---------Incase of Large Test Set-----------

        Dataset dataForClassification = FileHandler.loadDataset(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_2_test.csv"), 0, ",");


        int testLength=0;
        for (Instance inst: dataForClassification){
            System.out.println(inst.classValue());
            testLength++;
        }
        Object realClassValue[] = new Object[testLength];
        Object predictedClassValue[] = new Object[testLength];
        int i=0;
        for (Instance inst : dataForClassification) {
            realClassValue[i] = inst.classValue();
            predictedClassValue[i] = rff.classify(inst);
            i++;
        }
        i--;
        for(int k=0;k<i;k++)
        {
            System.out.println("Real Value is : "+realClassValue[k]+" vs Class predicted by Model is: "+predictedClassValue[k]);
        }

        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(rff, dataForClassification);
        for (Object o : pm.keySet())
            System.out.println(o + ":\nFMeasure: " + pm.get(o).getFMeasure()+"\nPrecision: "+ pm.get(o).getPrecision()+"\nRecall: "+ pm.get(o).getRecall());

        //System.out.println(pm.getCost());

    }

}
