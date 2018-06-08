package sys;

import net.sf.javaml.classification.Classifier;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;


public class ExecuteServlet extends HttpServlet{

    protected void doGet(HttpServletRequest request, HttpServletResponse response)  throws ServletException, IOException  {

        try {
            response.setContentType("text/plain");
            PrintWriter out = response.getWriter();
            System.out.println("In sys.ExecuteServlet");
            String mType = request.getParameter("mType");
            String dType = request.getParameter("dType");
            String data = request.getParameter("data");
            int count;
            if (dType.equals("3")) count = 3;
            else count = 23;
            int i = 0;
            double inst[] = new double[count];
            for (String str : data.split(","))
                inst[i++] = Double.parseDouble(str);
            if (i == (count - 1)) System.out.println("Input Read Successfully: i: " + i);
            for(double d:inst)System.out.println(d);

            //Creating New Instance here
            Instance instance = new DenseInstance(inst);
            String doc="<!DOCTYPE html><html lang=\"en\"><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"/><meta name=\"viewport\" content=\"width=device-width, initial-scale=1, maximum-scale=1.0\"/><title>Machine Learning for Medical Data</title><!-- CSS  --><link href=\"https://fonts.googleapis.com/icon?family=Material+Icons\" rel=\"stylesheet\"><link href=\"css\\materialize.css\" type=\"text/css\" rel=\"stylesheet\" media=\"screen,projection\"/><link href=\"css\\style.css\" type=\"text/css\" rel=\"stylesheet\" media=\"screen,projection\"/></head><body background=\"rsc/lawn.jpg\" onload=\"test();\"><div class=\"navbar-fixed\"><nav class=\"blue darken-1\" role=\"navigation\"><div class=\"nav-wrapper container\"><a id=\"logo-container\" href=\"index.html\" class=\"brand-logo\"><i class=\"material-icons \" style=\"font-size: 30px\">polymer</i></a></div></nav></div><div class=\"container blue-text text-lighten-1\"><div class=\"section\"><div class=\"row hoverable\"><h4 class=\"center light\">";
            //Creating Document
            out.println(doc);

            if (!mType.equals("ann")) {
                ObjectInputStream ois = new ObjectInputStream(
                    new FileInputStream("C:/apache-tomcat-8.0.33/webapps/MachineLearningForMedicalDataSets/models/" + mType + "" + dType + ".model"));
                Classifier cls = (Classifier) ois.readObject();
                ois.close();

                Object predictedClassValue=cls.classify(instance);
                out.println("The class predicted by the model is: "+predictedClassValue);
            }
            else
            {
                MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("ann"+dType+".json")));
                DataInputStream dis = new DataInputStream(new FileInputStream("ann"+dType+".bin"));
                INDArray features= Nd4j.create(inst);
                INDArray newParams = Nd4j.read(dis);
                dis.close();
                MultiLayerNetwork model = new MultiLayerNetwork(confFromJson);
                model.init();
                model.setParams(newParams);
                INDArray predicted = model.output(features,false);
                int j=0;
                int class_count;
                if(count==3)class_count=3;
                else class_count=6;
                while(j<class_count)
                {
                    if(predicted.getInt(j)==1)
                        if(class_count==3) {
                            out.println("The class predicted by the model is: " + j);
                        }
                        else
                            out.println("The class predicted by the model is: " + (j+1));
                    j++;
                }

                out.println("</h4>");
            }


        }catch(Exception e){e.printStackTrace();}

    }

}
