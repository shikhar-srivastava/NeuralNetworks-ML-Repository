package sys.Roc_AUC;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 @Author: Shikhar Srivastava
   */
public class LoadMLP{
    public static void main(String[] args) throws Exception {

        int batchSize=50;
        int numOutputs=2;
        RecordReader testreader= new CSVRecordReader();
        testreader.initialize(new FileSplit(new File("dl4j_0.4_examples/src/main/resources/saved_models/apnea_data_2_test.csv")));
        DataSetIterator testIter =(DataSetIterator) new RecordReaderDataSetIterator(testreader,batchSize,15,numOutputs);
        DataSet test= testIter.next();

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("Neo_15001_2Hidden.json")));
        DataInputStream dis = new DataInputStream(new FileInputStream("Neo_15001_2Hidden.bin"));
        INDArray newParams = Nd4j.read(dis);
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
        savedNetwork.init();
        savedNetwork.setParams(newParams);
        Evaluation eval = new Evaluation(numOutputs);

        INDArray features = test.getFeatureMatrix();
        INDArray lables = test.getLabels();
        INDArray predicted = savedNetwork.output(features,false);
        System.out.println("Lables: "+lables);
        System.out.println("Predicted: "+predicted);
        filePrinter(predicted,"predicted.csv");
        filePrinter(lables,"lables.csv");

        eval.eval(lables, predicted);

        //Print the evaluation statistics

        System.out.println(eval.stats());

    }

    public static void filePrinter(INDArray toBePrinted,String fileName) throws Exception
    {
        BufferedWriter outputWriter = null;
        outputWriter = new BufferedWriter(new FileWriter(fileName));
        int shape[]=toBePrinted.shape();
        for (int i = 0; i < shape[0]; i++) {
           for(int j=0;j<shape[1];j++)
           {
                outputWriter.write((toBePrinted.getFloat(i,j))+"");
                if(j!=(shape[1]-1))outputWriter.write(',');
           }
            outputWriter.newLine();

       }
        outputWriter.flush();
        outputWriter.close();
    }
}
