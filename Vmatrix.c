#include<stdio.h>
#include<stdlib.h>


/* After getting the individual x-, y- and z- component integration, use this function to calculate the final 64-by-64 matrix  */
int main(void)
{
	double X[4][4]={
					{0.05,0.058333,-0.1,-0.008333},
					{0.058333,0.283333,-0.241667,-0.1},
					{-0.1,-0.241667,0.283333,0.058333},
					{-0.008333,-0.1,0.058333,0.05}
				   };
    double Y[4][4]={
					{0.05,0.058333,-0.1,-0.008333},
					{0.058333,0.283333,-0.241667,-0.1},
					{-0.1,-0.241667,0.283333,0.058333},
					{-0.008333,-0.1,0.058333,0.05}
				   };
    double Z[4][4]={
					{0.003968,0.025595,0.011905,0.000198},
					{0.025595,0.235714,0.185119,0.011905},
					{0.011905,0.185119,0.235714,0.025595},
					{0.000198,0.011905,0.025595,0.003968}
				   };
    
	
	double temp[16][16];
	double Vmatrix[64][64];

	FILE* fp;

	int i,j;

	/* Calculate the temporary 16*16 matrix */
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i][j]=Y[0][0]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i][j+4]=Y[0][1]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i][j+8]=Y[0][2]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i][j+12]=Y[0][3]*Z[i][j];
				}
			}
//////////////////////////////////////////////////////////////////////////////
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+4][j]=Y[1][0]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+4][j+4]=Y[1][1]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+4][j+8]=Y[1][2]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+4][j+12]=Y[1][3]*Z[i][j];
				}
			}
//////////////////////////////////////////////////////////////////////
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+8][j]=Y[2][0]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+8][j+4]=Y[2][1]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+8][j+8]=Y[2][2]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+8][j+12]=Y[2][3]*Z[i][j];
				}
			}
//////////////////////////////////////////////////////////////////////
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+12][j]=Y[3][0]*Z[i][j];
				}
			}

	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+12][j+4]=Y[3][1]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+12][j+8]=Y[3][2]*Z[i][j];
				}
			}
	for(i=0;i<4;i++)
			{
				for(j=0;j<4;j++)
				{
					temp[i+12][j+12]=Y[3][3]*Z[i][j];
				}
			}

	/*    calculate the 64*64 V matrix     */

	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i][j]=X[0][0]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i][j+16]=X[0][1]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i][j+32]=X[0][2]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i][j+48]=X[0][3]*temp[i][j];
				}
			}
/////////////////////
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+16][j]=X[1][0]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+16][j+16]=X[1][1]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+16][j+32]=X[1][2]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+16][j+48]=X[1][3]*temp[i][j];
				}
			}
//////////////////////
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+32][j]=X[2][0]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+32][j+16]=X[2][1]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+32][j+32]=X[2][2]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+32][j+48]=X[2][3]*temp[i][j];
				}
			}
//////////////////////////////////////
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+48][j]=X[3][0]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+48][j+16]=X[3][1]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+48][j+32]=X[3][2]*temp[i][j];
				}
			}
	for(i=0;i<16;i++)
			{
				for(j=0;j<16;j++)
				{
					Vmatrix[i+48][j+48]=X[3][3]*temp[i][j];
				}
			}
       /*  just for testing   */
	//printf("\n%15e", Vmatrix[0][0]);
	//printf("\n%15e", Vmatrix[63][63]);
	//printf("\n%15e", Vmatrix[0][63]);
	//printf("\n%15e", Vmatrix[15][15]);
	//printf("\n%15e", Vmatrix[30][12]);
	//printf("\n%15e", Vmatrix[10][50]);
	//printf("\n%15e", Vmatrix[4][32]);

   /*   save the final result to a txt file  */
	fp = fopen( "output.txt", "w" );
	for(i=0;i<64;i++)
	{
		for(j=0;j<64;j++)
			{
				fprintf(fp, "%15e", Vmatrix[i][j]);
			}			
		fprintf(fp, "\n");
	}
	return 0;

}