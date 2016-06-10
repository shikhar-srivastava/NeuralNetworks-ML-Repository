package Weka.KNearestNeighbours;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileInputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;

/**
 *
 * @ author: Shikhar Srivastava
 * @ Due credit to: Samy
 */
public class KNearestNeighbors_Tester {

    //public static double[] input_array;
    //public static Instance trainerData;
    /*public KNearestNeighbors_Tester(double[] input_array)
    {
        this.input_array=input_array;
        trainerData= new Instance(3);
        trainerData.setValue(0,input_array[0]);
        trainerData.setValue(1,input_array[1]);
        trainerData.setValue(2,input_array[2]);
        trainerData.setClassValue(0);
    }*/

    public static void main(String[] args) throws Exception {
        ArrayList<Attribute> atts = new ArrayList<Attribute>();
        atts.add(new Attribute("0"));
        atts.add(new Attribute("1"));
        atts.add(new Attribute("2"));
        atts.add(new Attribute("@@class@@"));
        Instance trainerData= new Instance(4);
        Instances trainData= new Instances("Inst1",new FastVector(),3);
        trainData.add(trainerData);
        trainData.setClassIndex(trainData.numAttributes() - 1);
        trainerData.setValue(0,173);
        trainerData.setValue(1,192.16);
        trainerData.setValue(2,229.76);
        trainerData.setClassValue(0);
        trainerData.setDataset(trainData);
        InputStream is = new ObjectInputStream(
            new FileInputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/Weka/KNearestNeighbours/knnweka.xml"));
        ObjectInputStream ois= (ObjectInputStream)is;
        Classifier knn = (Classifier) ois.readObject();
            ois.close();
        System.out.println("Class Predicted: "+knn.classifyInstance(trainerData));


    }
}
