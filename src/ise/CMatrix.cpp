#include "CMatrix.h"
#include <math.h>
#include <fstream>
using namespace std;

void CMatrix::Create(int s,int t,double init)
{
 int i,j;

 m_row=s;
 m_col=t;

 data = new double* [m_row];
 for(i=0;i < m_row ; i++)
	data[i]=new double[m_col];
  
 for(i=0;i<m_row;i++)
  for(j=0;j<m_col;j++)
   data[i][j]=init;
  
}	

CMatrix::CMatrix()
{
}

CMatrix::CMatrix(int s,int t,double init)
{
 Create(s,t,init);
}

CMatrix::CMatrix(CMatrix& other)
{
 int i,j;
 
 Create(other.m_row,other.m_col);

 for(i=0;i<m_row;i++)
  for(j=0;j<m_col;j++)
   data[i][j]=other.data[i][j];
 
}

CMatrix::~CMatrix()
{
 int i;
 for(i=0;i<m_row;i++)
  delete [] data[i];
 
 delete [] data;  
}

CMatrix& CMatrix::operator=(const CMatrix& other)
{
 int i,j;

 if(this != &other)
 {
  for(i=0;i<m_row;i++)
   for(j=0;j<m_col;j++)
    data[i][j]=other.data[i][j];

 }
 return *this;  
}

double* CMatrix::operator[](int index)
{
 if(index <0 || index >= m_row)
 {
//  cerr << "Index is invalid. Index range must be from 0 to " << m_row-1 << endl;
 // cerr << "Current index is " << index << endl;
 // exit(0); 
 }

 return data[index];
}

// CMatrix addition
CMatrix CMatrix::operator+(CMatrix& other)
{
 int i,j;
 CMatrix sum(m_row,m_col);

 for(i=0;i<m_row;i++)
  for(j=0;j<m_col;j++)
   sum.data[i][j]=data[i][j]+other.data[i][j];
  
 return sum;
}

// CMatrix subtraction
CMatrix CMatrix::operator-(CMatrix& other)
{
 int i,j;
 CMatrix sub(m_row,m_col);

 for(i=0;i<m_row;i++)
  for(j=0;j<m_col;j++)
   sub.data[i][j]=data[i][j]-other.data[i][j];
  
 return sub;
}

// CMatrix multiplication
CMatrix CMatrix::operator*(const CMatrix& other)
{
 int i,j,k;
 CMatrix multi(m_row,other.m_col);
 
 if(m_col != other.m_row)
 {
  //cerr << "Matrix multiplication can not be defined!!" << endl;
  //exit(0);
 }

 for(i=0;i<m_row;i++)
  for(k=0;k<other.m_col;k++)
   for(j=0;j<m_col;j++)
    multi.data[i][k] += data[i][j] * other.data[j][k];

 return multi;
}

// Transpose operator
CMatrix CMatrix::operator!()
{
 int i,j;
 CMatrix trans(m_col,m_row);

 for(i=0;i<m_col;i++)
  for(j=0;j<m_row;j++)
   trans.data[i][j]=data[j][i];

 return trans;

}

CMatrix CMatrix::operator/(int r)
{
 int i,j;
 CMatrix div(m_row,m_col);

 for(i=0;i<m_row;i++)
  for(j=0;j<m_col;j++)
   div.data[i][j] = data[i][j] / r;

 return div;
}

double CMatrix::Inverse(CMatrix& inv)
{
 double *a,t,pivot[1400]={0,},temp;
 double *det=&temp;
 long i,j,ipvot[1400]={0,},index[1400][2]={0,};
 long k,l,li,im_row=0,icol=0;
 
 a=new double[m_row*m_col];
 for(i=0;i<m_row*m_col;i++)
  a[i]=0;

 *det=1.0;

 // Converting 2D to 1D
 for(j=0;j<m_col;j++)
 {
  for(i=0;i<m_row;i++)
   a[j*m_col+i]=data[i][j];

  ipvot[j]=0;
 }

 for(i=0;i<m_col;i++)
 {
  t=0;
  for(j=0;j<m_col;j++)
  {
   if((ipvot[j]-1) != 0)
   {
    for(k=0;k<m_col;k++)
    {   
     if((ipvot[k]-1) > 0)
     {
  //    cerr << "Error in inverse operation!!" << endl;
    //  exit(0);
     }
    
     else if((ipvot[k]-1) < 0 && (fabs(t) < fabs(*(a+j*m_row+k))))
     {
      im_row=j;
      icol=k;
      t= *(a+j*m_row+k);
     }
    }
   }
  }

  ipvot[icol]++;

  // To put pivot element on diagonal
  if (im_row != icol)
  {
   *det *= -1;
   for (l=0;l<m_col;l++)
   {
    t= *(a+im_row*m_row+l);
    *(a+im_row*m_row+l) = *(a+icol*m_row+l);
    *(a+icol*m_col+l) = t;
   }
  }

  index[i][0]=im_row;
  index[i][1]=icol;
  pivot[i]= *(a+icol*m_row+icol);
  *det *= pivot[i];
 
  /*
  if (*det > 1e37)
   cout << "Overflow error in inverse " << i << " det= " << *det << endl;
  */

  // To divide pivot row by pivot element
  *(a+icol*m_row+icol) = 1.0;

  for(l=0;l<m_col;l++)
   *(a+icol*m_row+l) /= pivot[i];

  for(li=0;li<m_col; li++)
   if(li != icol)
   {
    t= *(a+li*m_row+icol);
    *(a+li*m_row+icol) =0;
    for (l=0;l<m_col;l++)
     *(a+li*m_row+l) -= (*(a+icol*m_row+l))*t;
   }
 } 

 // To interchange columns
 for(l=m_col-1; l> -1; l--)
  if(index[l][0] != index[l][1])
  {
   im_row=index[l][0];
   icol=index[l][1];

   for (k=0; k<m_col; k++)
   {
    t= *(a+k*m_row+im_row);
    *(a+k*m_row+im_row) = *(a+k*m_row+icol);
    *(a+k*m_row+icol) =t;
   }
  }

 // Converting 1D to 2D
 for(j=0;j<m_row;j++)
  for(i=0;i<m_col;i++)
   inv.data[j][i]=a[m_col*i+j];

 //for(i=0;i<m_row;i++)
  //for(j=0;j<m_col;j++)
   //if(fabs(inv.data[i][j]) < 1e-8)
    //inv.data[i][j]=0;

 if(*det == 1 && inv.data[0][0] == data[0][0])
 {
  *det=0;
  //cout << "Inverse of this matrix is not defined!!" << endl;
  //exit(0);
 }

 return *det;
}

void CMatrix::Load_File(char *filename)
{
	/*
 int i,j;
 ifstream in(filename);

 if(in.fail())
 {
  cerr << "Error!! No <"<< filename << "> file in this directory.." << endl;
  exit(0);
 }
 
 for(i=0;i<m_row;i++)
 {
  for(j=0;j<m_col;j++)
   in >> data[i][j];

 }

 in.close();*/
}

void CMatrix::Show(int x1,int x2,int y1,int y2)
{
	/*
 int i,j;
 for(i=x1;i<=x2;i++)
 {
  for(j=y1;j<=y2;j++)
   cout << data[i][j] << "\t";

  cout << endl;

 }
 cout << endl;*/
}

void CMatrix::Save_File(char *filename)
{

	int i,j;
	std::ofstream out(filename);

	for(i=0;i<m_row;i++)
	{
		for(j=0;j<m_col;j++)
			out << data[i][j] << "\t";

		out << "\n";
	}

	out.close();
}



 

////////////////////////////////// matrix.cpp //////////////////////////////////////
/*
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include "CMatrix.h"

int main(void)
{
 int i, j;
 int row=2, col=2;   //2x2 MATRIX일 경우
 double deter=0.0; //INVESR MATRIX의 determinant 값
 CMatrix data(row, col, 0);
 CMatrix inver(row, col, 0);
 CMatrix in(row, col, 0);
 CMatrix out(row, col, 0);

 data[0][0]=1;     //(example) MATRIX의 입력값 설정
 data[0][1]=3;
 data[1][0]=1;
 data[1][1]=2;

 in[0][0]=1;
 in[0][1]=3;
 in[1][0]=1;
 in[1][1]=2;

 deter=data.Inverse(inver);     //MATRIX의 INVERSE (data를 INVERSE하여 inver에 저장) 
 //printf("%f\n", deter);

 out=in*data; //MATRIX의 multiplication

 for(i=0; i<row; i++){
  for(j=0; j<col; j++){
   printf("%f\n",inver[i][j]);
 }} //INVERSE MATRIX의 출력

 for(i=0; i<row; i++){
  for(j=0; j<col; j++){
   printf("%f\n",out[i][j]);
 }} //multiplication의 결과 MATRIX의 출력
 return 0;
}*/
