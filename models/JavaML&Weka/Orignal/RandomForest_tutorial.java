package Java_ML.Orignal;

import java.io.File;
import java.util.Map;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

/**
 * @author: Shikhar Srivastava
 * Implementation of KNearestNeighbour on Sample Dataset
 * @due credit to: Thomas Abeel for JAVA ML Tutorial
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


public class RandomForest_tutorial {
    /**
     * Shows the default usage of the KNN algorithm.
     */
    public static void main(String[] args) throws Exception { /* Load a data set */
        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/train_file.csv"), 0, ",");
        /*
         * Contruct a KNN classifier that uses 3 neighbors to make a decision.
         */
        int no_of_trees=1;
        while(no_of_trees<1000) {

            Classifier knn = new net.sf.javaml.classification.tree.RandomForest(no_of_trees);
            knn.buildClassifier(data);

        /*
         * Load a data set for evaluation, this can be a different one, but for
         * this example we use the same one.
         */
            Dataset dataForClassification = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/test_file.csv"), 0, ",");

            Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, dataForClassification);
            System.out.println("No. of trees:# "+no_of_trees);
            for (Object o : pm.keySet())
                System.out.println(o + ":\nAccuracy: " + pm.get(o).getAccuracy() /*+ "\nTPR: " + pm.get(o).getTPRate() + "\nFPR: " + pm.get(o).getFPRate() + "\nTNR: " + pm.get(o).getTNRate() + "\nFNR: " + pm.get(o).getFNRate()*/);
            no_of_trees++;
        }


    }

}
