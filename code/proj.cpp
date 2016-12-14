

/*
Devendra Pratap Yadav 

Preprovessing of Image and storing feature Vector for image.

*/


#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <sstream>

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#define gv(im,x,y)  im.at<uchar>(x,y)
#define im(x,y)  img.at<uchar>(x,y)
#define imo(x,y)  nimg.at<uchar>(x,y)
#define C(x,y)  C.at<uchar>(x,y)
#define scu(x) saturate_cast<uchar> (x)
#define pii pair<double,double>
#define ff first
#define ss second
#define real first
#define imag second
#define scast(x) saturate_cast<uchar>(x)
#define PI 3.14159265359

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}


using namespace patch;
using namespace cv;
using namespace std;

int pri = 0 ;

Mat image, result, sobelx, sobely ; // Mat to store gradients of image

int orows, ocols, cellsizex, cellnox, cellsizey, cellnoy ;
int ang[] = {10, 30, 50, 70, 90, 110, 130, 150, 170} ;

vector <double> hist[32][32] ;
vector <double> feature ;



void print(vector <pair<int,int> > &a )
{
 for (int i=0;i<a.size();i++)
    cout<<a[i].first<<", "<<a[i].second<<" | ";

 cout<<endl;
}


// show image in window
void show(String name, Mat &img, int wait)
{

 namedWindow( name, WINDOW_AUTOSIZE );// Create a window for display.
 imshow( name, img );
 if (wait>0)
 waitKey(wait);
 else
 waitKey(0);

}


// find rmse error for two Mat
double rmse(Mat &img,Mat &nimg)
{


 double sum=0;
 int cc=1;

 for(int i=0;i<img.rows;i++)
    {
      for (int j=0;j<img.cols;j++)
      {
            if (im(i,j)<10)
                continue;

            int diff=im(i,j)-imo(i,j);

            sum+=diff*diff;
            cc++;

      }
    }

    sum/=cc;

    sum=sqrt (sum);

   //cout<<"RMSE : "<<setprecision(3)<<sum<<endl;

    return sum;
} // end bit plane


void print(pii **arr,int r,int c)
{


 for (int i=0;i<r;i++)
  {
    for (int j=0;j<c;j++)
        printf("%6.1f+i%6.1f  ",arr[i][j].ff,arr[i][j].ss);

    cout<<endl;
    cout<<endl;
  }

}


pii ** alloc(int r,int c)
{
  pii **arr;
  arr=(pii **)malloc(2*r*(sizeof(pii* )));

  for (int i=0;i<2*r;i++)
  {
   arr[i]=(pii *)malloc(2*c*(sizeof(pii)));
  }


  for (int i=0;i<2*r;i++)
  {
    for (int j=0;j<2*c;j++)
        arr[i][j]=make_pair(0.0,0.0);

  }
  return arr;

}


pii ** getCopy(pii ** arr,int r,int c)
{
  pii **arr2;
  arr2=(pii **)malloc(2*r*(sizeof(pii* )));

  for (int i=0;i<2*r;i++)
  {
   arr2[i]=(pii *)malloc(2*c*(sizeof(pii)));
  }


  for (int i=0;i<2*r;i++)
  {
    for (int j=0;j<2*c;j++)
        {
        if (i<r && j<c)
        arr2[i][j]=arr[i][j];
        else
        arr2[i][j]=make_pair(0.0,0.0);
        }

  }
  return arr2;

}



Mat makeImage(pii **fmat,int r,int c)
{

    double minn=INT_MAX, maxx=INT_MIN;
    for (int i=0;i<r;i++)
      {
        for (int j=0;j<c;j++)
            {
                minn=min( minn,fmat[i][j].real );
                maxx=max( maxx,fmat[i][j].real );
            }

      }

      for (int i=0;i<r;i++)
      {
        for (int j=0;j<c;j++)
            {
                fmat[i][j].real= (fmat[i][j].real-minn)/(maxx-minn)*255.0;
            }

      }
      Mat img(r,c, CV_8U, Scalar(0,0,0));

      for (int i=0;i<r;i++)
      {
        for (int j=0;j<c;j++)
            {
                im(i,j)=scu(fmat[i][j].real);
            }

      }


      return img;

} // end makeImage





// correct inversion of image. Make foreground white and background black
void correctInversion(Mat &img)
{
   int rr=(double)img.rows*0.05; int cc=(double)img.cols*0.05;

   Rect in(cc,rr, (img.cols-2*cc), (img.rows-2*rr));

    Mat b;
    img.copyTo(b);
    b(in)=Scalar::all(0);

   int val=  mean(b)[0];
    //show("mean",b,0);

   if ( abs(val-0) < abs(val-255) ) // find which value is current backgroung closer to
        img=Scalar::all(255)-img;

} // end correctInversion


// threshold the image and find a bounding box for foreground pixels. Resize image to 32x32
void boundImage(Mat &img)
{
    Mat thr,points;
    threshold(img,thr,0,255,THRESH_BINARY | THRESH_OTSU);   // threshold image using otsu

    findNonZero(thr,points);

    Rect Min_Rect=boundingRect(points); // find bounding rectangle.
    img=img(Min_Rect);

    Size size(28,28);

    resize(img,img,size);
    copyMakeBorder(img,img,2,2,2,2,BORDER_CONSTANT,0);  //  Resize image to 32x32


    //threshold(img,img,20,255,THRESH_BINARY);

}   // end boundImage

// threshold the image using a mask and find a bounding box for foreground pixels. Resize image to 32x32
void boundImageMask(Mat &img,Mat &mask)
{
    Mat thr,points;
    threshold(img,thr,128,255,THRESH_BINARY | THRESH_OTSU); // threshold image using otsu

    img.copyTo(img,mask);       // use maask to select part of image

    findNonZero(thr,points);

    Rect Min_Rect=boundingRect(points); // find bounding rectangle.
    img=img(Min_Rect);

    Size size(32,32);

    resize(img,img,size);
    //copyMakeBorder(img,img,2,2,2,2,BORDER_CONSTANT,0);


    //threshold(img,img,20,255,THRESH_BINARY);

}   // end boundImage


// calculate the histogram for each cell in image
int calc_histogram()
{
	int i, j, k, l, p, xin, yin ;

	double angle, magnitude, te1, te2, tex, tey ;



	for(i=0;i<32;i++)
	{
		for(j=0;j<32;j++)
		{
			hist[i][j].clear() ;

			for(k=0;k<9;k++)
			{
				hist[i][j].push_back(0) ;
			}
		}
	}


	for(i=0;i<cellnox;i++)
	{
		for(j=0;j<cellnoy;j++)
		{

			for(k=0;k<8;k++)
			{
				for(l=0;l<8;l++)
				{
					xin = i*cellsizex + k ;
					yin = j*cellsizey + l ;

					tex = sobelx.at<float>(xin, yin) ;
					tey = sobely.at<float>(xin, yin) ;

					if(tex == 0)
					{
						if(tey != 0)
							angle = 90 ;
						else
							angle = 180 ;
					}
					else
					{
						angle = atan ( (double)tey / (double)tex ) * 180 / PI ;

						if(angle <= 0)
							angle = angle + 180 ;
					}


					magnitude = abs(sobelx.at<float>(xin, yin)) + abs(sobely.at<float>(xin, yin)) ;



					for(p=1;p<9;p++)
					{
						if(angle <= ang[p] && angle >= ang[p-1])
						{
							te1 = (angle - ang[p-1]) / 20 ;
							te2 = (ang[p] - angle) / 20 ;

							hist[i][j][p-1]+= te2 * magnitude ;
							hist[i][j][p]+= te1 * magnitude ;

							break ;
						}
					}

					if(p == 9 && angle <= 10)
					{
						te2 = (10 - angle) / 20 ;
						te1 = 1 - te2 ;

						hist[i][j][8]+= te2 * magnitude ;
						hist[i][j][0]+= te1 * magnitude ;
					}
					else if(p == 9 && angle >= 170)
					{
						te2 = (angle - 170) / 20 ;
						te1 = 1 - te2 ;

						hist[i][j][0]+= te2 * magnitude ;
						hist[i][j][8]+= te1 * magnitude ;
					}
				}
			}

		}
	}

}

// normalize histogram [0,1] for a block based on values from 4 cells forming the block
int normalize_histogram()
{
	int i, j, k, l, p, xin, yin ;

	double sum ;

	vector <double> temp ;

	feature.clear() ;


	for(i=0;i<(cellnox-1);i++)
	{
		for(j=0;j<(cellnoy-1);j++)
		{
			temp.clear() ;
			sum = 0 ;

			for(k=0;k<9;k++)
				temp.push_back(hist[i][j][k]) ;

			for(k=0;k<9;k++)
				temp.push_back(hist[i+1][j][k]) ;

			for(k=0;k<9;k++)
				temp.push_back(hist[i][j+1][k]) ;

			for(k=0;k<9;k++)
				temp.push_back(hist[i+1][j+1][k]) ;

			for(k=0;k<36;k++)
				sum = sum + temp[k]*temp[k] ;

			sum = sqrt(sum) ;

			if(sum != 0)
			{
				for(k=0;k<36;k++)
					temp[k] = temp[k] / sum ;
			}

			for(k=0;k<36;k++)
				feature.push_back(temp[k]) ;
		}
	}


	for(i=0;i<cellnox;i++)
	{
		for(j=0;j<cellnoy;j++)
		{
			sum = 0 ;

			for(k=0;k<9;k++)
			{
				sum = sum + hist[i][j][k]*hist[i][j][k] ;
			}

			sum = sqrt(sum) ;

			if(sum != 0)
			{
				for(k=0;k<9;k++)
					hist[i][j][k] = hist[i][j][k] / sum ;
			}

		}
	}

}

// display gradients as image
int visualize_hog()
{
	int i, j, k, l, p, xin, yin, index ;

	double angle, grad, xam ;

	//feature.clear() ;

	for(i=0;i<cellnox;i++)
	{
		for(j=0;j<cellnoy;j++)
		{
			angle = 0 ;
			grad = 0 ;
			xam = 0 ;
			index = -1 ;



			for(k=0;k<9;k++)
			{
				if(xam < hist[i][j][k])
				{
					xam = hist[i][j][k] ;
					index = k ;
				}

				angle+= ang[k] * hist[i][j][k] ;

				grad+= hist[i][j][k] ;
			}


			if(grad < 0.2)
				continue ;


			angle = (ang[index]*PI) / 180 ;


			angle = tan(angle) ;


			xin = 4 ;
			yin = 4 ;



			result.at<uchar>(xin + i*cellsizex , yin + j*cellsizey) = 255 ;

			while(yin < 8 && xin < 8 && yin >= 0 && xin >= 0)
			{
				result.at<uchar>(round(xin) + i*cellsizex, round(yin) + j*cellsizey) = 255 ;

				xin = xin + 1 ;
				yin = yin + angle ;
			}

			xin = 4 ;
			yin = 4 ;

			while(yin < 8 && xin < 8 && yin >= 0 && xin >= 0)
			{
				result.at<uchar>(round(xin) + i*cellsizex, round(yin) + j*cellsizey) = 255 ;

				xin = xin - 1 ;
				yin = yin - angle ;
			}

			if(ang[index] == 90)
			{
				for(k=0;k<8;k++)
				{
					result.at<uchar>(4 + i*cellsizex, k + j*cellsizey) = 255 ;
				}
			}
		}
	}
}

// extract HOG features from image
int extractHOG(Mat image)
{
	// Extract HOG Features

    orows = image.rows ;
	ocols = image.cols ;

	cellsizex = 8 ;
	cellsizey = 8 ;

	cellnox = orows / cellsizex ;
	cellnoy = ocols / cellsizey ;

	result = Mat::zeros(orows, ocols, CV_8U) ;

	Mat gray =image;
	//cvtColor(image, gray, CV_BGR2GRAY);

	Sobel(gray, sobely, CV_32F, 1, 0);
	Sobel(gray, sobelx, CV_32F, 0, 1);

	calc_histogram() ;

	normalize_histogram() ;

	visualize_hog() ;

	if(pri == 1)
	{
		namedWindow( "Compressed Image", WINDOW_NORMAL );
		imshow( "Compressed Image", result );

		waitKey(10) ;
	}
}

// get path of input images
string getPath(int sn, int in)  // for font characters
{
    string sNo="0"+to_string(sn);
    string path="../Sample"+sNo+"/img"+sNo+"-";


    if (in<10)
    path=path+"0000";
    else if (in<100)
    path=path+"000";
    else if (in<1000)
    path=path+"00";
    else
    path=path+"0";

    path+=to_string(in)+".png";


    return path;
} // end genPath


// get path of input images
string getPath2(int sn, int in) // for handwritten characters
{
    string sNo="";
    if (sn<10)
    sNo="00";
    else
    sNo="0";

    sNo=sNo+to_string(sn);
    string path="../data/Sample"+sNo+"/img"+sNo+"-";


    if (in<10)
    path=path+"00";
    else
    path=path+"0";

    path+=to_string(in)+".png";


    return path;
} // end genPath2


// get path of input images
string getPathImg(int sn, int in,int mask)  // for img characters
{

    string sNo="";
    if (sn<10)
    sNo="00";
    else
    sNo="0";

    sNo=sNo+to_string(sn);

    string fld="bmp";
    if (mask==1)
        fld="msk";


    string path="../data-img/"+fld+"/Sample"+sNo+"/img"+sNo+"-";


    if (in<10)
    path=path+"0000";
    else if (in<100)
    path=path+"000";
    else if (in<1000)
    path=path+"00";
    else
    path=path+"0";

    path+=to_string(in)+".png";


    return path;
} // end genPath


// write feature vectors to file - HOG
void writeData(FILE *out,Mat arr[],int ssize,int label)
{
    //FILE *out=fopen("features.txt","w");

    // print size of input vectors


    for (int i=1;i<=ssize;i++)
    {
        fprintf(out,"%d,",label);

         //cout<<arr[i].rows<<", "<<arr[i].cols<<"\n";

        for (int r=0;r<arr[i].rows;r++)
        {
            for (int c=0;c<arr[i].cols;c++)
                {

                    if(c==arr[i].cols-1 && r==arr[i].rows-1)
                        fprintf(out,"%f", arr[i].at<float>(r,c) );
                    else
                        fprintf(out,"%f,", arr[i].at<float>(r,c) );

                }

        }

        fprintf(out,"\n");

    }


} // end writeData


// write feature vectors to file - pixel values
void writeDataPixel(FILE *out,Mat arr[],int ssize,int label)
{
    //FILE *out=fopen("features.txt","w");

    // print size of input vectors


    for (int i=1;i<=ssize;i++)
    {
        fprintf(out,"%d,",label);

         //cout<<arr[i].rows<<", "<<arr[i].cols<<"\n";

        for (int r=0;r<arr[i].rows;r++)
        {
            for (int c=0;c<arr[i].cols;c++)
                {

                    if(c==arr[i].cols-1 && r==arr[i].rows-1)
                        fprintf(out,"%d", (int)arr[i].at<uchar>(r,c) );
                    else
                        fprintf(out,"%d,", (int)arr[i].at<uchar>(r,c) );

                }

        }

        fprintf(out,"\n");

    }


} // end writeData





int main( int argc, char** argv )
{

    int sampleNo=1; // value of class being read

    int sampleSize=55;  // number of files in a class

    FILE *out=fopen("features.txt","w");

    int usingImg=1; // flag for reading image character files rather than handwritten characters

    // number of samples for each class for image characters
    int imgCount[]={0,106,80,71,51,48,65,65,49,33,35,559,116,216,192,447,80,144,194,303,
    78,93,216,150,364,383,160,36,390,343,313,93,85,68,81,68,56,159,39,64,47,228,38,37,53,128,34,
    35,59,44,136,149,38,55,127,125,112,42,34,37,36,44,36};


    // loop for each class
    for (int fno=1;fno<63;fno++)
    {
        sampleNo=fno;   // set number of samples

        if (usingImg)
        sampleSize=imgCount[fno]-1;

        Mat images[sampleSize+1];   // array for images

        Mat mask[sampleSize+1]; // array for image masks


        Mat feat[sampleSize+1];

        //read images
        for (int i=1;i<=sampleSize;i++)
        {
            string path=getPathImg(sampleNo,i,0);
            cout<<"Reading IMG: "<<path<<endl;
            images[i]=imread(path.c_str(),CV_LOAD_IMAGE_GRAYSCALE);

            path=getPathImg(sampleNo,i,1);
            cout<<"Reading MASK: "<<path<<endl;
            mask[i]=imread(path.c_str(),CV_LOAD_IMAGE_GRAYSCALE);

            //show("Original",images[i],0);
            //show("Mask",mask[i],0);
            //swaitKey(10);

        }


        // read images into array
        for (int i=1;i<=sampleSize;i++)
        {
            //images[i]=Scalar::all(255)-images[i];
            //show("Original",images[i],0);
            if(usingImg)
            threshold(images[i],images[i],128,255,THRESH_BINARY | THRESH_OTSU);

            correctInversion(images[i]);    // make foreground white
            Mat img2;

            // perform adaptive median smoothing
            medianBlur(images[i],img2,5);
            int err=rmse(images[i],img2);

            if (err>50)
                medianBlur(images[i],img2,3);

            cout<<i<<" RMSE = "<<rmse(images[i],img2)<<endl;

            if (err<50)
                images[i]=img2;


            // find bounding box for image
            if (usingImg)
             boundImageMask(images[i],mask[i]);
             else
            boundImage(images[i]);
            //show("Processed",images[i],0);

            // Extract HOG features for image
            extractHOG(images[i]);

            //cout<<feature.size()<<endl;
            int len = feature.size() - 1 ;
            feat[i] = Mat::zeros(1, len+1, CV_32F) ;

            while(len >= 0)
            {
            	feat[i].at<float>(0, len) = feature[len] ;
            	len-- ;
        	}



        }

        printf("\n****************************************************************\n");


        /********************************************************

        UNCOMMENT THE LINE ACCORDING TO FEATURE VECTOR REQUIRED

        ********************************************************/


        //writeDataPixel(out,images,sampleSize,sampleNo);       // Write pixel values to features.txt


        writeData(out,feat,sampleSize,sampleNo);                // Write HOG feature values to features.txt

    }// end for fno

















    //show("median",images[100],0);
    //Mat img=images[100];
    //show("orig",img,-1);

    /*
    cout<<img.rows<<" , "<<img.cols<<endl;

    Mat thr,points;
    threshold(img,thr,254,255,THRESH_BINARY);


    findNonZero(thr,points);


    Rect Min_Rect=boundingRect(points);
    img=img(Min_Rect);

    Size size(28,28);

    resize(img,img,size);
    copyMakeBorder(img,img,2,2,2,2,BORDER_CONSTANT,0);


    threshold(img,img,20,255,THRESH_BINARY);

     show("bound",img,0);


    for (int i=0;i<img.rows;i++)
    {
        for (int j=0;j<img.cols;j++)
            cout<< (int)gv(img,i,j) <<" ";

            cout<<endl;
    }


    show("original",img,1);
    medianBlur(img,img,3);
    show("median",img,0);
    */




    return 0;
} // end main

