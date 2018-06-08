package Java_ML.NaiveBayes;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.filter.discretize.EqualWidthBinning;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.Map;


/*
 * @author: Shikhar Srivastava
 * Implementation of NaiveBayes on Sample Dataset
 * @due credit to: Thomas Abeel for the Example for JAVA ML Library
 *
 */

public class NaiveBayes_trainer {

    public static void main(String[] args) throws Exception {


        //---------Train the Required Model---------------
        		/* Discretize through EqualWidthBinning */
        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/Feature_train.csv"), 0, ",");
        EqualWidthBinning eb = new EqualWidthBinning(20);
        System.out.println("Start discretisation");
        eb.build(data);
        Dataset ddata = data.copy();
        eb.filter(ddata);
        //FileHandler.exportDataset(ddata,new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/train_trial_binned.txt"));

        //boolean useLaplace = true;
       // boolean useLogs = true;
        Classifier nb = new NaiveBayesClassifier(true, true, false);
        nb.buildClassifier(ddata);

        //-----------------#Model is now Trained. Serializing Model: SAVE #--------------------

       ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/NaiveBayes/nb23.model"));


        // /---------Incase of Large Test Set-----------

        Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/Feature_test.csv"), 0, ",");




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

        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(nb, dataForClassification);
        for (Object o : pm.keySet())
            System.out.println(o + ":\nFMeasure: " + pm.get(o).getFMeasure()+"\nTPR: "+pm.get(o).getTPRate()+"\nFPR: "+ pm.get(o).getFPRate()+ "\nTNR: "+pm.get(o).getTNRate()+"\nFNR: "+pm.get(o).getFNRate());

        System.out.println(pm);

    }

}
