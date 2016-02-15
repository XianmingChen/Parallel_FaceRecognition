#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include "EBGM_FeatureVectors.h"
#include "EBGM_FaceComparison.h"

double Feature_Vectors[Total_train_face][500][41][2];
double Temp_Feature_Vectors[Total_train_face*500*41*2];
double Local_Vectors[Local_train_Max][500][41][2];


void read_image(char *filepath,double image[][Width])
{
	int i,j;
	CvScalar s;
	IplImage *img=cvLoadImage(filepath,0);
    for(i=0;i<img->height;i++)
	{
        for(j=0;j<img->width;j++)
		{
			s=cvGet2D(img,i,j); 
			image[i][j]=s.val[0]/255;
        }
    }
    cvReleaseImage(&img);  
}


double ****Malloc4D(double ****Array,int n1,int n2,int n3,int n4)
{
	int i,j,k,m;
	Array=(double ****)malloc(n1*sizeof(double ***));
	for (i=0;i<n1;i++)
	{
		Array[i]=(double ***)malloc(n2*sizeof(double **));
		for (j=0;j<n2;j++)
		{
			Array[i][j]=(double **)malloc(n3*sizeof(double *));
			for (k=0;k<n3;k++)
			{
				Array[i][j][k]=(double *)malloc(n4*sizeof(double));
				for (m=0;m<n4;m++)
				{
					Array[i][j][k][m]=0.0;
				}
			}
		}
	}
	return Array;
}

double ****Free4DArray(double ****Array,int n1,int n2,int n3)
{
	int i,j,k;
	for (i=0;i<n1;i++)
	{
		for (j=0;j<n2;j++)
		{
			for (k=0;k<n3;k++)
			{
				free(Array[i][j][k]);
				Array[i][j][k]=0;
			}
		}
	}		
	for (i=0;i<n1;i++)
	{
		for (j=0;j<n2;j++)
		{
			free(Array[i][k]);
			Array[i][k]=0;
		}
	}
	for (i=0;i<n1;i++)
	{
		free(Array[i]);
		Array[i]=0;
	}
	free(Array);
	Array=NULL;
	return NULL;
}


void main(int argc, char* argv[])
{
	int my_rank; 
    int p;   
	int Local_trianface=0;
	int Local_probeface=0;

	char image_path[255];
	int i;
	double trainface[Height][Width]={0.0};
	double ****Gabor_Respone=NULL;

	double Mean_Value[Filter_Num][2]={0.0};
	double Each_Feature_Vectors[500][41][2]={0.0};

	int train_feature_count[Total_train_face]={0};
	int local_feature_count[Local_train_Max]={0};
	int each_feature_count=0;
	int probe_feature_count=0;

	int Accuracy=0;
	int Local_count=0;
	int Probe_count=0;

	double start_time,end_time;
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	
	start_time=MPI_Wtime();
	Gabor_Respone=Malloc4D(Gabor_Respone,Filter_Num,Height,Width,2);

	if (my_rank==0)
	{	
		Local_trianface=Total_train_face/p;
	}
	MPI_Bcast(&Local_trianface,1,MPI_INT,0,MPI_COMM_WORLD);

	for (i=0;i<Local_trianface;i++)
	{
		printf("Training image %d by processor %d...\n",my_rank*Local_trianface+i+1,my_rank);
		sprintf(image_path,"Aligned_FERET/input/trainfaces/%d.jpg",my_rank*Local_trianface+i+1);
		read_image(image_path,trainface);	
		GaborFilterResponse(trainface,Gabor_Respone,Mean_Value);
		EBGM_FeatureVectors(Gabor_Respone,Mean_Value,&each_feature_count,Each_Feature_Vectors);
		local_feature_count[i]=each_feature_count;
		each_feature_count=0;
		memcpy(Local_Vectors[i],Each_Feature_Vectors,500*41*2*8);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allgather(local_feature_count,Local_trianface,MPI_INT,train_feature_count,Local_trianface,MPI_INT,MPI_COMM_WORLD);
	MPI_Allgather(Local_Vectors,Local_trianface*500*41*2,MPI_DOUBLE,Temp_Feature_Vectors,Local_trianface*500*41*2,MPI_DOUBLE,MPI_COMM_WORLD);
	memcpy(Feature_Vectors,Temp_Feature_Vectors,Total_train_face*500*41*2*8);

	if (my_rank==0)
	{
		printf("Begin to probe images...\n");
		Local_probeface=Total_probe_face/p;
	}
	MPI_Bcast(&Local_probeface,1,MPI_INT,0,MPI_COMM_WORLD);

	for (i=0;i<Local_probeface;i++)
	{
		printf("Probing image %d by processor %d...\n",my_rank*Local_trianface+i+1,my_rank);
		sprintf(image_path,"Aligned_FERET/input/probefaces/%d.jpg",my_rank*Local_trianface+i+1);
		read_image(image_path,trainface);
		GaborFilterResponse(trainface,Gabor_Respone,Mean_Value);
		EBGM_FeatureVectors(Gabor_Respone,Mean_Value,&each_feature_count,Each_Feature_Vectors);
		probe_feature_count=each_feature_count;
		each_feature_count=0;
		Accuracy=EBGM_FaceComparison(Total_train_face,train_feature_count,probe_feature_count,Each_Feature_Vectors,my_rank*Local_trianface+i);

		if (Accuracy==1)
		{
			Local_count++;
			printf("Image %d Probe successfully by %d!\n",my_rank*Local_trianface+i+1,my_rank);
		}
		else
		{
			printf("Image %d Probe failed by %d!\n",my_rank*Local_trianface+i+1,my_rank);
		}  	
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&Local_count,&Probe_count,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

	Free4DArray(Gabor_Respone,Filter_Num,Height,Width);
	if (my_rank==0)
	{
		printf("%d Images probe completed!\n",Total_probe_face);
		printf("Probe accuary= %f\n",(double)Probe_count/Total_probe_face);
		end_time=MPI_Wtime();
		printf("Computational time is: %fs.\n",end_time-start_time);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}