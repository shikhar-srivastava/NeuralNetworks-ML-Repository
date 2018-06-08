package Java_ML;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;

/**
 * @ author: Shikhar Srivastava
 *  Code to segment Dataset into Test & Train sets based on division of sub-classes seperately.
 */
public class Test_Train_Creator_2 {
    public static void main(String[] args) throws Exception {

        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_3.csv"), 3, ",");
        System.out.println("No. of Attributes: "+data.noAttributes()+"\nClasses: "+ data.classes());
        Instance temp;
        int classCount[] ={0,0,0};
        Dataset train= new DefaultDataset();
        Dataset test= new DefaultDataset();
        Dataset classes[]= new Dataset[3];
        int no_of_classes=3;
        int i;
        for(i=0;i<no_of_classes;i++)
        {
            classes[i]= new DefaultDataset();
        }
        int noOfRows=0;
        while(noOfRows<600)
        {
            temp=data.instance(noOfRows);
            int classValue= Integer.parseInt((String)temp.classValue());
            classCount[classValue]++;
            classes[classValue].add(temp);
            // System.out.println("Class at "+(noOfRows)+" is: "+temp.classValue());
            noOfRows++;
        }

        //----------Dividing into 80:20 training:test for each class------------

        double trainTestSplit=0.8;

        for(i=0;i<3;i++)
        {
            double counter=0;
            while(counter<classCount[i])
            {
                temp=classes[i].instance((int)counter);
                if(counter <=(trainTestSplit*classCount[i]))
                {
                    train.add(temp);
                }
                else test.add(temp);

                counter++;
            }
        }
        //------------------Test and training set created----------


        int noOfRows_test=0;
        int noOfRows_train=0;
        System.out.println("Training Set: ");
        while(noOfRows_train<482)
        {
            temp=train.instance(noOfRows_train);
            noOfRows_train++;
            System.out.println(temp);

        }
        System.out.println("Test Set: ");
        while(noOfRows_test<118)
        {
            temp=test.instance(noOfRows_test);
            noOfRows_test++;
            System.out.println(temp);
        }
        //----------EXPORTING CREATED TEST AND TRAINING DATASET------------------

        FileHandler.exportDataset(train,new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_train.txt"));
        FileHandler.exportDataset(test,new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/rgb_data_3_test.txt"));


        //---------COMPLETED-----------

    }

}
