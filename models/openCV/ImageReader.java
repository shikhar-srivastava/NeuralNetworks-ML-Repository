import Java_ML.KNearestNeighbours.KNearestNeighbor_Final;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class ImageReader {

    public static void main(String[] args) {
        for (int m = 1; m <= 6; m++) {

            try {
                BufferedImage currentImage = ImageIO.read(new File("C:/Users/MAHE/Desktop/scientia_sit_potentia/2.Projects/3.ResearchIITKGP/Data/Image/"+m+".jpg"));

                KNearestNeighbor_Final classifier_Object = null;
                try {
                    classifier_Object = new KNearestNeighbor_Final();
                } catch (Exception e) {
                    System.out.println(e);
                }
                int[][] ImageRGB = convertImagetoRGBMatrix(currentImage, classifier_Object);
                System.out.println("Image Matrix obtained. Writing now...");
                int xLength = ImageRGB.length;
                System.out.println("xLength: " + xLength);
                int yLength = ImageRGB[0].length;
                System.out.println("yLength: " + yLength);
                BufferedImage b = new BufferedImage(yLength, xLength, 4);
                for (int x = 0; x < xLength; x++) {
                    for (int y = 0; y < yLength; y++) {
                        System.out.println("x: " + x + " y:" + y);
                        b.setRGB(y, x, ImageRGB[x][y]);
                    }
                }
                boolean done = ImageIO.write(b, "jpg", new File("C:/Users/MAHE/Desktop/scientia_sit_potentia/2.Projects/3.ResearchIITKGP/Data/Image/classified"+m+".jpg"));

                if (done) System.out.println("Written Sucessfully");
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    }
    private static int[][] convertImagetoRGBMatrix(BufferedImage image,KNearestNeighbor_Final classifier_Object) {
        int width = image.getWidth();
        int height = image.getHeight();
        System.out.println("Image size: "+height+" X "+width);
        int[][] resultImage = new int[height][width];
        int predictedClass;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
               try
               {
                   predictedClass=readClassifyRGB(image.getRGB(col,row),classifier_Object);
                   switch(predictedClass)
                   {
                       case 0: resultImage[row][col]= 0x1193648D;break;
                       case 1: resultImage[row][col]= 0x117BC8A4;break;
                       case 2: resultImage[row][col]= 0x11FFC65D;break;
                       //case 3: resultImage[row][col]= 0xFF4CC3D9;break;
                       //case 4: resultImage[row][col]= 0xFFF16745;break;
                       case -1: resultImage[row][col]=0xFFFFFFFF;break;
                       default: throw new Exception();
                   }

               }catch(Exception e){
                   System.out.println(e);}

            }
        }

        return resultImage;
    }
    private static int readClassifyRGB(int pixel,KNearestNeighbor_Final classifier_Object) throws Exception{
        int alpha = (pixel >> 24) & 0xff; //255
        int red = (pixel >> 16) & 0xff;
        int green = (pixel >> 8) & 0xff;
        int blue = (pixel) & 0xff;
        if(red<=10&&green<=10&&blue<=10)return -1;
        else return classifier_Object.classify(red,green,blue);


    }
}
