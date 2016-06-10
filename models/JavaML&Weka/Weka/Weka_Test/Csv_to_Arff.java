package Weka_Test;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class Csv_to_Arff {
    /**
     * takes 2 arguments:
     * - CSV input file
     * - ARFF output file
     */
    public static void main(String[] args) throws Exception {


        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_weka_3_train.csv"));
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_weka_3_train.arff"));
        saver.setDestination(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_weka_3_train.arff"));
        saver.writeBatch();
    }
}
