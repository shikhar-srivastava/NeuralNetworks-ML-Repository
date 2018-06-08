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
public class Test_Train_Creator {
    public static void main(String[] args) throws Exception {

        Dataset data = FileHandler.loadDataset(new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/FeatureData_norm.csv"), 23, ",");
        System.out.println("No. of Attributes: "+data.noAttributes()+"\nClasses: "+ data.classes());
        Instance temp;
        int classCount[] ={0,0,0,0,0,0};
        Dataset train= new DefaultDataset();
        Dataset test= new DefaultDataset();
        Dataset classes[]= new Dataset[6];
        int i;
        for(i=0;i<6;i++)
        {
            classes[i]= new DefaultDataset();
        }
        int noOfRows=0;
        while(noOfRows<683)
        {
            temp=data.instance(noOfRows);
            int classValue= Integer.parseInt((String)temp.classValue());
            classCount[classValue-1]++;
            classes[classValue-1].add(temp);
           // System.out.println("Class at "+(noOfRows)+" is: "+temp.classValue());
            noOfRows++;
        }
        /* DataSet is found to be as follows:
                    683 Rows
            No. of instances for class: 6 is: 14
            No. of instances for class: 5 is: 33
            No. of instances for class: 4 is: 68
            No. of instances for class: 3 is: 209
            No. of instances for class: 2 is: 102
            No. of instances for class: 1 is: 257
         */

        //----------Dividing into 60:40 training:test for each class------------

        double trainTestSplit=0.6;

        for(i=0;i<6;i++)
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
        /*
                        Train Case Size: 413 (60.4%)
                        Test Case Size: 270  (39.6%)
         */

        int noOfRows_test=0;
        int noOfRows_train=0;
       System.out.println("Training Set: ");
        while(noOfRows_train<413)
        {
            temp=train.instance(noOfRows_train);
            noOfRows_train++;
            System.out.println(temp);

        }
        System.out.println("Test Set: ");
        while(noOfRows_test<270)
        {
            temp=test.instance(noOfRows_test);
            noOfRows_test++;
            System.out.println(temp);
        }
        //----------EXPORTING CREATED TEST AND TRAINING DATASET------------------

        FileHandler.exportDataset(train,new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/Feature_train_norm.txt"));
        FileHandler.exportDataset(test,new File("D:/Deeplearning4Java/dl4j_0.4_examples/src/main/resources/classification/Feature_test_norm.txt"));


        //---------COMPLETED-----------

    }

}
