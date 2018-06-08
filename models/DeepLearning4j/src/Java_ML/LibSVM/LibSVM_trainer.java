package Java_ML.LibSVM;

import libsvm.LibSVM;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

/*
 * This emulation is referenced from a tutorial by Thomas Abeel
 * @author Shikhar Srivastava
 * @Due credit to: Thomas Abeel for the JAVA ML Tutorial
 *
 */
public class LibSVM_trainer{


    public static void main(String[] args) throws Exception { //---------Train the Required Model---------------

        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_2_train.csv"), 0, ",");
        LibSVM svm = new LibSVM();
        System.out.println("Beginning Training here.......");
        svm.buildClassifier(data);


        System.out.println("Finished Training here.......");
        //-----------------#Model is now Trained. Serializing Model: SAVE #--------------------

        ObjectOutputStream oos = new ObjectOutputStream(
            new FileOutputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/LibSVM/svm.bin"));
        oos.writeObject(svm);
        oos.flush();
        oos.close();

        //---------Incase of Large Test Set-----------
    /*
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
            predictedClassValue[i] = svm.classify(inst);
            realClassValue[i] = inst.classValue();
            i++;
        }
        i--;
        for(int k=0;k<i;k++)
        {
            System.out.println("Real Value is : "+realClassValue[k]+" vs Class predicted by Model is: "+predictedClassValue[k]);
        }

       /* Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(svm, dataForClassification);
        for (Object o : pm.keySet())
            System.out.println(o + ":\nAccuracy: " + pm.get(o).getAccuracy()+"\nTPR: "+pm.get(o).getTPRate()+"\nFPR: "+ pm.get(o).getFPRate()+ "\nTNR: "+pm.get(o).getTNRate()+"\nFNR: "+pm.get(o).getFNRate());

        System.out.println(pm);*/
        System.out.println("-------------Parameters are:------------------ ");
        // svm_parameter s1=svm.getParameters();
        //System.out.println("Raw Decision Value: "+svm.rawDecisionValues(data.get(0)));
        double rawDecs[] = svm.rawDecisionValues(data.get(0));
        for(double x:rawDecs)
        {
            System.out.println("Raw Decision Values : "+x);
        }
        double weights[]= svm.getWeights();
        int[] labels=svm.getLabels();
        for(int x:labels)
        {
            System.out.println("Labels are: "+x);
        }
        int weight_label[]= svm.getParameters().weight_label;
        System.out.println("Weights: ");
        for(double w: weights) {
            System.out.println(w);
        }
        System.out.println("Weight Label: ");
        for(int w: weight_label) {
            System.out.println(w);
        }
        System.out.println("Bias is: "+svm.getParameters().coef0);
    }

}
