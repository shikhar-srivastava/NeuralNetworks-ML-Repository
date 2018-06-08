package Java_ML.KNearestNeighbours;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

import java.io.FileInputStream;
import java.io.InputStream;
import java.io.ObjectInputStream;

/*
 * @author: Shikhar Srivastava
 * Implementation of KNearestNeighbour on Sample Dataset
 * @due credit to: Thomas Abeel for the Example for JAVA ML Library
 */

public class KNearestNeighbor_Final {

    private static double rgb[];
    private static Classifier knn;
    private Exception e;
    public KNearestNeighbor_Final() throws Exception {
        try {
            rgb = null;
            InputStream is = new ObjectInputStream(
                new FileInputStream("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/java/sys.AutoEncoder.Java_ML/RandomForest/rff3.model"));
            ObjectInputStream ois = (ObjectInputStream) is;
            knn = (Classifier) ois.readObject();
            ois.close();

           }catch(Exception e)
                 {
                     e.printStackTrace();
                     throw new ModelException("From KNearestNeighbor Constructor");
                }
    }
    public int classify(int r,int g,int b) throws Exception {


        //------------De-Serializing the SAVED model---------------------
       try {
           rgb = new double[3];
           rgb[0] = r;
           rgb[1] = g;
           rgb[2] = b;
           Instance inst = new DenseInstance(rgb);
           return Integer.parseInt((String)knn.classify(inst));
          }catch(Exception e)
       {
           e.printStackTrace();
           throw new ModelException("From Classify function of KNearestNeighbour");
       }
    }

}

  class ModelException extends Exception {
    String msg;
    public ModelException(String message) {
        super(message);
        msg=message;
    }
    public String toString()
    {
        return "Exception: "+msg;
    }
}
