#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//////////////////////////////////////////////////////////
// JAS 2011.06.30
//
// reg_testbench.c
//
//   This is a slightly optimized version of
//   xyzInt.c and Vmatrix.c merged together in a
//   way that will make debugging the code in reg.cxx
//   a little easier.
//
//   This file is to help me keep in sync more easily
//   with Qi's work.
//
// THIS FILE IS INDEPENDENT OF PLASTIMATCH
//
// To Compile:
//   gcc -o reg_testbench reg_testbench.c
//
////////



// from math_util.h
////////////////////////////////////////////////////////////////////////
#define m_idx(m1,c,i,j) m1[i*c+j]

static inline void
mat_mult_mat (
    double* m1, 
    const double* m2, int m2_rows, int m2_cols, 
    const double* m3, int m3_rows, int m3_cols
)
{
    int i,j,k;
    for (i = 0; i < m2_rows; i++) {
	for (j = 0; j < m3_cols; j++) {
	    double acc = 0.0;
	    for (k = 0; k < m2_cols; k++) {
		acc += m_idx(m2,m2_cols,i,k) * m_idx(m3,m3_cols,k,j);
	    }
	    m_idx(m1,m3_cols,i,j) = acc;
	}
    }
}

static inline void
vec_outer (
    double* v1,
    const double* v2,
    const double* v3,
    const int n
)
{
    int i,j;
    for (j=0; j<n; j++) {
        for (i=0; i<n; i++) {
            v1[n*j + i] = v2[j] * v3[i];
            v1[n*j + i] = v2[j] * v3[i];
            v1[n*j + i] = v2[j] * v3[i];
            v1[n*j + i] = v2[j] * v3[i];
        }
    }
}
////////////////////////////////////////////////////////////////////////


// Matrix Display Utilities
////////////////////////////////////////////////////////////////////////
void
fprintM (FILE* fp, double* M, int r, int c)
{
    int i, j;

    for (j=0; j<r; j++) {
        for (i=0; i<c; i++) {
            fprintf (fp, "%15e", M[c*j+i]);
        }
        fprintf (fp, "\n");
    }
}

void
printM (double* M, int r, int c)
{
    int i, j;

    for (j=0; j<r; j++) {
        for (i=0; i<c; i++) {
            printf ("%10f", M[c*j+i]);
        }
        printf ("\n");
    }
}
////////////////////////////////////////////////////////////////////////


// Algorithm Routines
////////////////////////////////////////////////////////////////////////

/* main() from Vmatrix.c */
void
get_Vmatrix (double* V, double* X, double* Y, double* Z)
{
    int i,j;
    double tmp[256];       /* 16 x 16 matrix */

    /* Calculate the temporary 16*16 matrix */
    for (j=0; j<4; j++) {
        for (i=0; i<4; i++) {
            tmp[16*(j+ 0) + (i+ 0)] = Y[4*0 + 0] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+ 4)] = Y[4*0 + 1] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+ 8)] = Y[4*0 + 2] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+12)] = Y[4*0 + 3] * Z[4*j + i];

            tmp[16*(j+ 4) + (i+ 0)] = Y[4*1 + 0] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+ 4)] = Y[4*1 + 1] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+ 8)] = Y[4*1 + 2] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+12)] = Y[4*1 + 3] * Z[4*j + i];

            tmp[16*(j+ 8) + (i+ 0)] = Y[4*2 + 0] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+ 4)] = Y[4*2 + 1] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+ 8)] = Y[4*2 + 2] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+12)] = Y[4*2 + 3] * Z[4*j + i];

            tmp[16*(j+12) + (i+ 0)] = Y[4*3 + 0] * Z[4*j + i];
            tmp[16*(j+12) + (i+ 4)] = Y[4*3 + 1] * Z[4*j + i];
            tmp[16*(j+12) + (i+ 8)] = Y[4*3 + 2] * Z[4*j + i];
            tmp[16*(j+12) + (i+12)] = Y[4*3 + 3] * Z[4*j + i];
        }
    }

    /* Calculate the 64*64 V matrix */
    for (j=0; j<16; j++) {
        for (i=0; i<16; i++) {
            V[64*(j+ 0) + (i+ 0)] = X[4*0 + 0] * tmp[16*j + i];
            V[64*(j+ 0) + (i+16)] = X[4*0 + 1] * tmp[16*j + i];
            V[64*(j+ 0) + (i+32)] = X[4*0 + 2] * tmp[16*j + i];
            V[64*(j+ 0) + (i+48)] = X[4*0 + 3] * tmp[16*j + i];

            V[64*(j+16) + (i+ 0)] = X[4*1 + 0] * tmp[16*j + i];
            V[64*(j+16) + (i+16)] = X[4*1 + 1] * tmp[16*j + i];
            V[64*(j+16) + (i+32)] = X[4*1 + 2] * tmp[16*j + i];
            V[64*(j+16) + (i+48)] = X[4*1 + 3] * tmp[16*j + i];

            V[64*(j+32) + (i+ 0)] = X[4*2 + 0] * tmp[16*j + i];
            V[64*(j+32) + (i+16)] = X[4*2 + 1] * tmp[16*j + i];
            V[64*(j+32) + (i+32)] = X[4*2 + 2] * tmp[16*j + i];
            V[64*(j+32) + (i+48)] = X[4*2 + 3] * tmp[16*j + i];

            V[64*(j+48) + (i+ 0)] = X[4*3 + 0] * tmp[16*j + i];
            V[64*(j+48) + (i+16)] = X[4*3 + 1] * tmp[16*j + i];
            V[64*(j+48) + (i+32)] = X[4*3 + 2] * tmp[16*j + i];
            V[64*(j+48) + (i+48)] = X[4*3 + 3] * tmp[16*j + i];
        }
    }
}

/* main() from xyzInt.c */
void
init_analytic (double **QX, double **QY, double **QZ, double* gs)
{
    /* grid spacing */
    double rx = 1.0/gs[0];
    double ry = 1.0/gs[1];
    double rz = 1.0/gs[2];

    double B[16] = {
        1.0/6.0, -1.0/2.0,  1.0/2.0, -1.0/6.0,
        2.0/3.0,  0.0    , -1.0    ,  1.0/2.0,
        1.0/6.0,  1.0/2.0,  1.0/2.0, -1.0/2.0,
        0.0    ,  0.0    ,  0.0    ,  1.0/6.0
    };

    double RX[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  rx,   0.0,      0.0,
        0.0, 0.0, rx*rx,      0.0,
        0.0, 0.0,   0.0, rx*rx*rx
    };

    double RY[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  ry,   0.0,      0.0,
        0.0, 0.0, ry*ry,      0.0,
        0.0, 0.0,   0.0, ry*ry*ry
    };

    double RZ[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  rz,   0.0,      0.0,
        0.0, 0.0, rz*rz,      0.0,
        0.0, 0.0,   0.0, rz*rz*rz
    };

    double delta1[16] = {
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0
    };

    double delta2[16] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0, 0.0,
        0.0, 6.0, 0.0, 0.0
    };

    // Let's call Q the product of the recripocal grid spacing
    // matrix (R) and the B-spline coefficient matrix (B).
    mat_mult_mat (QX[0], B, 4, 4, RX, 4, 4);
    mat_mult_mat (QY[0], B, 4, 4, RY, 4, 4);
    mat_mult_mat (QZ[0], B, 4, 4, RZ, 4, 4);

    // Get the product of QX, QY, QZ and delta.
    //   QX1 is the  first-order derivative of X
    //   QX2 is the second-order derivative of X
    //   QY1 is the  first-order derivative of Y
    //   ... etc
    mat_mult_mat (QX[1], QX[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QX[2], QX[0], 4, 4, delta2, 4, 4);    
    mat_mult_mat (QY[1], QY[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QY[2], QY[0], 4, 4, delta2, 4, 4);    
    mat_mult_mat (QZ[1], QZ[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QZ[2], QZ[0], 4, 4, delta2, 4, 4);    
}

/* IInt() and getv() from xyzInt.c */
void
eval_integral (double* V, double* Qn, double is)
{
    int i,j;
    double S[16];

    double I[7] = {
        is,
        (1.0/2.0) * (is * is),
        (1.0/3.0) * (is * is * is),
        (1.0/4.0) * (is * is * is * is),
        (1.0/5.0) * (is * is * is * is * is),
        (1.0/6.0) * (is * is * is * is * is * is),
        (1.0/7.0) * (is * is * is * is * is * is * is)
    };

    // Generate 4 4x4 matrix by taking the outer
    // product of the each row in the Q matrix with
    // every other row in the Q matrix. We use these
    // to generate each element in V.
    for (j=0; j<4; j++) {
        for (i=0; i<4; i++) {
            vec_outer (S, Qn+(4*j), Qn+(4*i), 4);
            V[4*j + i] = (I[0] *  S[ 0])
                       + (I[1] * (S[ 1] + S[ 4]))
                       + (I[2] * (S[ 2] + S[ 5] + S[ 8]))
                       + (I[3] * (S[ 3] + S[ 6] + S[ 9] + S[12]))
                       + (I[4] * (S[ 7] + S[10] + S[13]))
                       + (I[5] * (S[11] + S[14]))
                       + (I[6] * (S[15]));
        }
    }
}
////////////////////////////////////////////////////////////////////////



int
main (void)
{
    FILE* fp;

    int i,n;

    double *QX[3], *QY[3], *QZ[3];      /* Arrays of array addresses */
    double QX0[16], QY0[16], QZ0[16];   /*  4 x  4 matrix */
    double QX1[16], QY1[16], QZ1[16];   /*  4 x  4 matrix */
    double QX2[16], QY2[16], QZ2[16];   /*  4 x  4 matrix */
    double X[256];                      /* 16 x 16 matrix */
    double Y[256];                      /* 16 x 16 matrix */
    double Z[256];                      /* 16 x 16 matrix */
    double V[4096];                     /* 64 x 64 matrix */
    int knots[64];                      /* local set for current region */

    double grid_spacing[3];

    QX[0] = QX0;    QY[0] = QY0;    QZ[0] = QZ0;
    QX[1] = QX1;    QY[1] = QY1;    QZ[1] = QZ1;
    QX[2] = QX2;    QY[2] = QY2;    QZ[2] = QZ2;

    for (i=0; i<3; i++) {
        memset (QX[i], 0, 16*sizeof(double));
        memset (QY[i], 0, 16*sizeof(double));
        memset (QZ[i], 0, 16*sizeof(double));
    }

    /* (in mm) */
    grid_spacing[0] = 1.0;
    grid_spacing[1] = 1.0;
    grid_spacing[2] = 1.0;

    init_analytic (QX, QY, QZ, grid_spacing);

    /* Produce output from xyzInt.c */
    eval_integral (X, QX[1], grid_spacing[0]);
    eval_integral (Y, QY[1], grid_spacing[1]);
    eval_integral (Z, QZ[0], grid_spacing[2]);

    printM (X, 4, 4);
    printM (Y, 4, 4);
    printM (Z, 4, 4);

    /* Produce output from Vmatrix.c */
    get_Vmatrix (V, X, Y, Z);

    fp = fopen ("output_jas.txt", "w");
    fprintM (fp, V, 64, 64);
    fclose (fp);

    return 0;
}
