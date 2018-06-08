package Java_ML;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;

/**
 * @ Author-Shikhar Srivastava
 */
public class SyntheticMinorityOversampling {

    public static void main(String[] args) throws Exception
    {
        CSVLoader loader = new CSVLoader();
        System.out.println("Error here");
        loader.setSource(new File("dl4j_0.4_examples/src/main/resources/classification/Feature_train.csv"));
        Instances data = loader.getDataSet();
        //Instances data = ConverterUtils.DataSource.read("dl4j_0.4_examples/src/main/resources/classification/Feature_train.csv");
        System.out.println(data);
        //weka.filters.supervised.instance.SMOTE sm = new SMOTE();
        //sm.setInputFormat(data);
        //System.out.println("No. of Nearest neighbours: "+sm.getNearestNeighbors());


    }

}
