#include<stdio.h>
#include<stdlib.h>

int MMultip(int r1, int c1, float **M1, int r2, int c2, float **M2, float **P);
float getv(float **S);
int printM(int r, int c, float **A);

// function to calculate the individual integration of the 4-by-4 matrices of vx, vy, vz


int main(void)
{
	float	B[4][4]={
					{1.0/6.0,-1.0/2.0,1.0/2.0,-1.0/6.0},
					{2.0/3.0,0.0,-1.0,1.0/2.0},
					{1.0/6.0,1.0/2.0,1.0/2.0,-1.0/2.0},
					{0.0,0.0,0.0,1.0/6.0}
				  };    
	float   rx=1.0, ry=1.0, rz=1.0;           /*      set the grid spacing of x- y- and z- direction    */
	float	RX[4][4]={
					{1.0,0.0,0.0,0.0},
					{0.0,rx,0.0,0.0},
					{0.0,0.0,rx*rx,0.0},
					{0.0,0.0,0.0,rx*rx*rx}
				  };    
	float	RY[4][4]={
					{1.0,0.0,0.0,0.0},
					{0.0,ry,0.0,0.0},
					{0.0,0.0,ry*ry,0.0},
					{0.0,0.0,0.0,ry*ry*ry}
				  };    
	float	RZ[4][4]={
					{1.0,0.0,0.0,0.0},
					{0.0,rz,0.0,0.0},
					{0.0,0.0,rz*rz,0.0},
					{0.0,0.0,0.0,rz*rz*rz}
				  };    

    float    delta1[4][4]={{0.0,0.0,0.0,0.0},
                           {1.0,0.0,0.0,0.0},
                           {0.0,2.0,0.0,0.0},
                           {0.0,0.0,3.0,0.0}};
    float    delta2[4][4]={{0.0,0.0,0.0,0.0},
                           {0.0,0.0,0.0,0.0},
                           {2.0,0.0,0.0,0.0},
                           {0.0,6.0,0.0,0.0}};
	int Br=4,Bc=4,Dr=4,Dc=4;
	float   QX[4][4], QY[4][4], QZ[4][4];
	float   QX1[4][4], QY1[4][4], QZ1[4][4], QX2[4][4], QY2[4][4], QZ2[4][4];

	float vx[4][4], vy[4][4], vz[4][4];

	/*      Get the product of B and the grid spacing of x- y- z- direction, and save in matrix Q     */
	MMultip(Br,Bc,(float**)B,Dr,Dc,(float**)RX,(float**)QX);
	MMultip(Br,Bc,(float**)B,Dr,Dc,(float**)RY,(float**)QY);
	MMultip(Br,Bc,(float**)B,Dr,Dc,(float**)RZ,(float**)QZ); 

	
// Get the product of QX,QY,QZ and delta, QX1 means the first-order derivative of x, and QX2 means the second-order derivative of x
// The same with QY1, QY2, QZ1, QZ2
	MMultip(Br,Bc,(float**)QX,Dr,Dc,(float**)delta1,(float**)QX1);
	MMultip(Br,Bc,(float**)QY,Dr,Dc,(float**)delta1,(float**)QY1);

/*  Get the individual x-, y-, z- component of the integration result  */
	IInt((float**)QX1, (float**)vx);
	IInt((float**)QY1, (float**)vy);
	IInt((float**)QZ, (float**)vz);
	printM(4, 4, (float**)vx);
	printM(4, 4, (float**)vy);
	printM(4, 4, (float**)vz);
	return 0;
}


int IInt(float **Q1, float **VXX)
{	
	float   Q11[1][4],Q12[1][4],Q13[1][4],Q14[1][4];	
	float   QT11[4][1],QT12[1][4],QT13[1][4],QT14[1][4];	
	float   S1[4][4],S2[4][4],S3[4][4],S4[4][4],S5[4][4],S6[4][4],S7[4][4],S8[4][4],S9[4][4];
	float   S10[4][4],S11[4][4],S12[4][4],S13[4][4],S14[4][4],S15[4][4],S16[4][4];
	
	float y;
	/* get the 1st row of the Q1 Matrix, and saved in Q11 as a row vector*/
	*((float*)Q11)=*((float*)Q1);
	*((float*)Q11+1)=*((float*)Q1+1);
	*((float*)Q11+2)=*((float*)Q1+2);
	*((float*)Q11+3)=*((float*)Q1+3);
	//printM(1, 4, (float**)Q11);
	/* transpose the Q11, and saved in QT11 as a colume vector*/
	trans(1, 4, (float**)Q11, (float**)QT11);
	//printM(4, 1, (float**)QT11);
	
	/* get the 2nd row of the Q1 Matrix, and saved in Q12 as a row vector*/
	*((float*)Q12)=*((float*)Q1+4);
	*((float*)Q12+1)=*((float*)Q1+5);
	*((float*)Q12+2)=*((float*)Q1+6);
	*((float*)Q12+3)=*((float*)Q1+7);
	//printM(1, 4, (float**)Q12);
	/* transpose the Q12, and saved in QT12 as a colume vector*/
	trans(1, 4, (float**)Q12, (float**)QT12);
	//printM(4, 1, (float**)QT12);

	/* get the 3rd row of the Q1 Matrix, and saved in Q13 as a row vector*/
	*((float*)Q13)=*((float*)Q1+8);
	*((float*)Q13+1)=*((float*)Q1+9);
	*((float*)Q13+2)=*((float*)Q1+10);
	*((float*)Q13+3)=*((float*)Q1+11);
	//printM(1, 4, (float**)Q13);
	/* transpose the Q13, and saved in QT13 as a colume vector*/
	trans(1, 4, (float**)Q13, (float**)QT13);
	//printM(4, 1, (float**)QT13);

	/* get the 4th row of the Q1 Matrix, and saved in Q14 as a row vector*/
	*((float*)Q14)=*((float*)Q1+12);
	*((float*)Q14+1)=*((float*)Q1+13);
	*((float*)Q14+2)=*((float*)Q1+14);
	*((float*)Q14+3)=*((float*)Q1+15);
	//printM(1, 4, (float**)Q14);
	/* transpose the Q14, and saved in QT14 as a colume vector*/
	trans(1, 4, (float**)Q14, (float**)QT14);
	//printM(4, 1, (float**)QT14);

	/*  multiply the colume vector with the row vector to produce a 4*4 matrix  */
	MMultip(4,1,(float**)QT11,1,4,(float**)Q11,(float**)S1);
	//printM(4, 4, (float**)S1);

	//printf("%10f", y);
	/*  get the value of the integration  */
	*((float*)VXX)=getv((float**)S1);


	MMultip(4,1,(float**)QT11,1,4,(float**)Q12,(float**)S2);
	//printM(4, 4, (float**)S2);
	*((float*)VXX+1)=getv((float**)S2);

	MMultip(4,1,(float**)QT11,1,4,(float**)Q13,(float**)S3);
	//printM(4, 4, (float**)S3);
	*((float*)VXX+2)=getv((float**)S3);

	MMultip(4,1,(float**)QT11,1,4,(float**)Q14,(float**)S4);
	//printM(4, 4, (float**)S1);
	*((float*)VXX+3)=getv((float**)S4);

	MMultip(4,1,(float**)QT12,1,4,(float**)Q11,(float**)S5);
	//printM(4, 4, (float**)S5);
	*((float*)VXX+4)=getv((float**)S5);

	MMultip(4,1,(float**)QT12,1,4,(float**)Q12,(float**)S6);
	//printM(4, 4, (float**)S6);
	*((float*)VXX+5)=getv((float**)S6);

	MMultip(4,1,(float**)QT12,1,4,(float**)Q13,(float**)S7);
	//printM(4, 4, (float**)S7);
	*((float*)VXX+6)=getv((float**)S7);

	MMultip(4,1,(float**)QT12,1,4,(float**)Q14,(float**)S8);
	//printM(4, 4, (float**)S8);
	*((float*)VXX+7)=getv((float**)S8);

	MMultip(4,1,(float**)QT13,1,4,(float**)Q11,(float**)S9);
	//printM(4, 4, (float**)S9);
	*((float*)VXX+8)=getv((float**)S9);

	MMultip(4,1,(float**)QT13,1,4,(float**)Q12,(float**)S10);
	//printM(4, 4, (float**)S10);
	*((float*)VXX+9)=getv((float**)S10);

	MMultip(4,1,(float**)QT13,1,4,(float**)Q13,(float**)S11);
	//printM(4, 4, (float**)S11);
	*((float*)VXX+10)=getv((float**)S11);

	MMultip(4,1,(float**)QT13,1,4,(float**)Q14,(float**)S12);
	//printM(4, 4, (float**)S12);
	*((float*)VXX+11)=getv((float**)S12);

	MMultip(4,1,(float**)QT14,1,4,(float**)Q11,(float**)S13);
	//printM(4, 4, (float**)S13);
	*((float*)VXX+12)=getv((float**)S13);

	MMultip(4,1,(float**)QT14,1,4,(float**)Q12,(float**)S14);
	//printM(4, 4, (float**)S14);
	*((float*)VXX+13)=getv((float**)S14);

	MMultip(4,1,(float**)QT14,1,4,(float**)Q13,(float**)S15);
	//printM(4, 4, (float**)S15);
	*((float*)VXX+14)=getv((float**)S15);

	MMultip(4,1,(float**)QT14,1,4,(float**)Q14,(float**)S16);
	//printM(4, 4, (float**)S16);
	*((float*)VXX+15)=getv((float**)S16);

	return	0;

}



/*  function to perform the multiplication of matrices */
int MMultip(int r1, int c1, float **M1, int r2, int c2, float **M2, float **P)
	{
			int i,j,k;
			for(i=0;i<r1;i++)
			{
				for(j=0;j<c2;j++)
				{
					*((float*)P+c2*i+j)=0;
				}
			}
			
			for(i=0;i<r1;i++)
			{
				for(j=0;j<c2;j++)
				{
					for(k=0;k<c1;k++)
					{
						*((float*)P+c2*i+j)+=(*((float*)M1+c1*i+k))*(*((float*)M2+c2*k+j));
					}
				}
			}
	
		return 0;
	}



/*   function to print out the matrix  */
int printM(int r, int c, float **A)
	{
		int i, j;

		for(i=0;i<r;i++)
		{
			for(j=0;j<c;j++)
			{
				printf("%10f", *((float*)A+c*i+j));
			}
			printf("\n");
		}
		return 0;
	}



/*  function to transpose a matrix  */
int trans(int r, int c, float **A, float **B)
	{
		int i, j;

		for(i=0;i<c;i++)
		{
			for(j=0;j<r;j++)
			{
				*((float*)B+r*i+j)=*((float*)A+c*j+i);
			}
			
		}
		return 0;
	}

/*  function to calculate the result of the integration  */
float getv(float **S)

	{
		float   i[7][1]={{1.0},{1.0/2.0},{1.0/3.0},{1.0/4.0},{1.0/5.0},{1.0/6.0},{1.0/7.0}};
		float	T[1][7];
		float	x[1][1];
		float	z;
		*((float*)T)=*((float*)S);
		*((float*)T+1)=*((float*)S+1)+*((float*)S+4);
		*((float*)T+2)=*((float*)S+2)+*((float*)S+5)+*((float*)S+8);
		*((float*)T+3)=*((float*)S+3)+*((float*)S+6)+*((float*)S+9)+*((float*)S+12);
		*((float*)T+4)=*((float*)S+7)+*((float*)S+10)+*((float*)S+13);
		*((float*)T+5)=*((float*)S+11)+*((float*)S+14);
		*((float*)T+6)=*((float*)S+15);

		//printM(1, 7, (float**)T);

		MMultip(1,7,(float**)T,7,1,(float**)i,(float**)x);
		//printf("%10f", *((float*)x));
		z=x[0][0];
		return z;
	}
