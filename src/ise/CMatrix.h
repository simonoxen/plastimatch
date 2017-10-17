/****************************************************************/
/*                */
/*        CMatrix.h        */
/*                */ 
/*      Programmed by       */       
/*     Shin     Gun     Shik      */
/*                */
/*  at Super-Resolution Image Processing laboratory   */
/*     in Yonsei University      */
/*                */
/****************************************************************/

#ifndef _CMATRIX_H_
#define _CMATRIX_H_

//#define pi acos(-1)

class CMatrix  
{
 private:
    
 public:
  int m_row,m_col;
  double **data;
        
  CMatrix();
  CMatrix(int,int,double init=0);
  CMatrix(CMatrix&); 
  ~CMatrix();
  void Create(int s,int t,double init=0);
  CMatrix& operator=(const CMatrix&);
  double* operator[](int);
  CMatrix operator+(CMatrix&);
  CMatrix operator-(CMatrix&);
  CMatrix operator*(const CMatrix&);
  CMatrix operator!();
  CMatrix operator/(int r);
  double Inverse(CMatrix& inv);
  void Load_File(char*);
  void Show(int x1,int x2,int y1,int y2);
  void Save_File(char*);
};

#endif
