package sys.Mark2;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.StandardScaler;

import java.io.File;

/**
 * @ author- Shikhar Srivastava
 * @ credit due to : Relevant Authors of DeepLearning4J Library
 *
 */
public class Standard_scaler_trainer {


    public static void main(String[] args) throws Exception {

        RecordReader train= new CSVRecordReader();
        train.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/Feature_train_norm_ann.csv")));
        DataSetIterator trainIter = (DataSetIterator) new RecordReaderDataSetIterator(train,413,0,6);
        StandardScaler s23= new StandardScaler();
        s23.fit(trainIter);
        s23.save(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/s23mean.bin"),new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/s23std.bin"));

        RecordReader train_2= new CSVRecordReader();
        train_2.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_train.csv")));
        DataSetIterator trainIterator = (DataSetIterator) new RecordReaderDataSetIterator(train_2,482,0,3);
        StandardScaler s3= new StandardScaler();
        s3.fit(trainIterator);
        s3.save(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/s3mean.bin"),new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/s3std.bin"));



    }
}
