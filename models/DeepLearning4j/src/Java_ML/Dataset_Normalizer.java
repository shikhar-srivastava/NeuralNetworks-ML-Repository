package Java_ML;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * @ author: Shikhar Srivastava
 */
public class Dataset_Normalizer {

    public static void main(String[] args) throws Exception
    {
        int batchSize=683;
       // ConverterUtils.DataSource source = new ConverterUtils.DataSource.read("dl4j_0.4_examples/src/main/resources/classification/FeatureData.csv");
        Instances data = DataSource.read("dl4j_0.4_examples/src/main/resources/classification/FeatureData.csv");
        data.setClassIndex(data.numAttributes() - 1);
        Standardize std = new Standardize();
        std.setInputFormat(data);
        std.setIgnoreClass(true);
        Instances processed = Filter.useFilter(data, std);
        System.out.println(processed);

    }



}
