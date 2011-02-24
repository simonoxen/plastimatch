/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bspline.h"
#include "bspline_landmarks.h"
#include "bspline_opts.h"
#include "landmark_warp.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "rbf_cluster.h"
#include "vf.h"
#include "volume.h"

//k-means++ clustering algorithm to separate landmarks into user-specified number of clusters
void
rbf_cluster_kmeans_plusplus(Landmark_warp *lw)
{
	int num_landmarks = lw->m_fixed_landmarks->num_points;
	int num_clusters = lw->num_clusters;
	float *mx, *my, *mz;
	float *D, *DD;
	int i,j;
	float xmin, ymin, zmin, xmax, ymax, zmax;
	float r, d, dmin;
	int clust_id;
	int kcurrent, count, reassigned, iter_count =0;
	
	mx = (float *)malloc(num_clusters*sizeof(float));
	my = (float *)malloc(num_clusters*sizeof(float));
	mz = (float *)malloc(num_clusters*sizeof(float));
	D  = (float *)malloc(num_landmarks*sizeof(float));
	DD = (float *)malloc(num_landmarks*sizeof(float));
		
	for(i=0;i<num_landmarks;i++) lw->cluster_id[i]=-1;

	xmin = xmax = lw->m_fixed_landmarks->points[0*3+0];
	ymin = ymax = lw->m_fixed_landmarks->points[0*3+1];
	zmin = zmax = lw->m_fixed_landmarks->points[0*3+2];

//kmeans++ initialization

	i = (int)((double)rand()/RAND_MAX*(num_landmarks-1.));
	mx[0]=lw->m_fixed_landmarks->points[i*3+0];
	my[0]=lw->m_fixed_landmarks->points[i*3+1]; 
	mz[0]=lw->m_fixed_landmarks->points[i*3+2];
	kcurrent=1;

do 
{
	for(i=0;i<num_landmarks;i++) {
		for(j=0;j<kcurrent;j++) {
		d =   (lw->m_fixed_landmarks->points[i*3+0]-mx[j])
		     *(lw->m_fixed_landmarks->points[i*3+0]-mx[j]) 
		    + (lw->m_fixed_landmarks->points[i*3+1]-my[j])
		     *(lw->m_fixed_landmarks->points[i*3+1]-my[j]) 
		    + (lw->m_fixed_landmarks->points[i*3+2]-mz[j])
		     *(lw->m_fixed_landmarks->points[i*3+2]-mz[j]);
		if (j==0) { dmin=d; }
		if (d<=dmin) { D[i]=dmin; }
		}
	}

//DD is a normalized cumulative sum of D
d=0;
for(i=0;i<num_landmarks;i++) d+=D[i];
for(i=0;i<num_landmarks;i++) D[i]/=d;
d=0;
for(i=0;i<num_landmarks;i++) { d+=D[i]; DD[i]=d; }

// randomly select j with probability proportional to D
r = ((double)rand())/RAND_MAX;
for(i=0;i<num_landmarks;i++) {
if ( i==0 && r<=DD[i] ) j = 0;
if ( i>0  && DD[i-1]<r && r<=DD[i] ) j = i;
}

mx[kcurrent] = lw->m_fixed_landmarks->points[j*3+0]; 
my[kcurrent] = lw->m_fixed_landmarks->points[j*3+1]; 
mz[kcurrent] = lw->m_fixed_landmarks->points[j*3+2];
kcurrent++;

} while(kcurrent < num_clusters);


//standard k-means algorithm
do {
reassigned = 0;

// assign
for(i=0;i<num_landmarks;i++) {
	for(j=0;j<num_clusters;j++) {
	d =  (lw->m_fixed_landmarks->points[i*3+0]-mx[j])
	    *(lw->m_fixed_landmarks->points[i*3+0]-mx[j]) + 
	     (lw->m_fixed_landmarks->points[i*3+1]-my[j])
	    *(lw->m_fixed_landmarks->points[i*3+1]-my[j]) + 
	     (lw->m_fixed_landmarks->points[i*3+2]-mz[j])
	    *(lw->m_fixed_landmarks->points[i*3+2]-mz[j]);
    if (j==0) { dmin=d; clust_id = 0; }
    if (d<=dmin) { dmin =d; clust_id = j; }
    }
    
    if ( lw->cluster_id[i] != clust_id) reassigned = 1;
    lw->cluster_id[i] = clust_id;
}

// calculate new means
for(j=0;j<num_clusters;j++) {
mx[j]=0; my[j]=0; mz[j]=0; count=0;
	for(i=0;i<num_landmarks;i++) {
    if (lw->cluster_id[i]==j) { 
	mx[j]+=lw->m_fixed_landmarks->points[i*3+0]; 
	my[j]+=lw->m_fixed_landmarks->points[i*3+1]; 
	mz[j]+=lw->m_fixed_landmarks->points[i*3+2]; 
	count++; 
	}
    }
    mx[j]/=count; my[j]/=count; mz[j]/=count;
}

iter_count++;

} while(reassigned && (iter_count<10000));

fprintf(stderr,"iter count %d\n", iter_count);



free(D);
free(DD);
free(mx);
free(my);
free(mz);
}

//calculate adaptive radius of each RBF
void
rbf_cluster_find_adapt_radius(Landmark_warp *lw)
{
int i,j,k, count;
int num_clusters = lw->num_clusters;
int num_landmarks = lw->m_fixed_landmarks->num_points; 
float d, D, dmax=-1;
float *d_nearest_neighb;

// NB what to do if there is just one landmark in a cluster??

for(k=0; k<num_clusters; k++) {
    D = 0; count = 0;
    for(i=0; i<num_landmarks; i++) {
	for(j=i; j<num_landmarks; j++) {
	    if ( lw->cluster_id[i] == k && lw->cluster_id[j] == k  && j != i ) {
		d = (lw->m_fixed_landmarks->points[i*3+0]-lw->m_fixed_landmarks->points[j*3+0])
		   *(lw->m_fixed_landmarks->points[i*3+0]-lw->m_fixed_landmarks->points[j*3+0]) + 
		    (lw->m_fixed_landmarks->points[i*3+1]-lw->m_fixed_landmarks->points[j*3+1])
		   *(lw->m_fixed_landmarks->points[i*3+1]-lw->m_fixed_landmarks->points[j*3+1]) + 
		    (lw->m_fixed_landmarks->points[i*3+2]-lw->m_fixed_landmarks->points[j*3+2])
		   *(lw->m_fixed_landmarks->points[i*3+2]-lw->m_fixed_landmarks->points[j*3+2]);
		D  += sqrt(d);
		if (sqrt(d)>dmax) dmax = sqrt(d);
		count++;
		}
	    }
	}
    D /= count;	
    D = D * 2 ; //a magic number

    printf("nclust %d   nland %d   dmax = %f  D = %f\n", num_clusters, num_landmarks, dmax, D);
    // single long cluster needs other treatment
    if ( (num_clusters == 1) && (dmax/(0.5*D) > 1.5) ) { 
	printf("long cluster, dmax %f D %f\n", dmax, D); D = dmax/2.1; 
        
	// radius is the max distance between nearest neighbors
	
	d_nearest_neighb = (float *)malloc(num_landmarks*sizeof(float));
	for(i=0;i<num_landmarks;i++) d_nearest_neighb[i]=1e20;
    
	for(i=0;i<num_landmarks;i++) {
	    for(j=0;j<num_landmarks;j++) {
		if (i==j) continue;
		d = (lw->m_fixed_landmarks->points[i*3+0]-lw->m_fixed_landmarks->points[j*3+0])
		   *(lw->m_fixed_landmarks->points[i*3+0]-lw->m_fixed_landmarks->points[j*3+0]) + 
		    (lw->m_fixed_landmarks->points[i*3+1]-lw->m_fixed_landmarks->points[j*3+1])
		   *(lw->m_fixed_landmarks->points[i*3+1]-lw->m_fixed_landmarks->points[j*3+1]) + 
		    (lw->m_fixed_landmarks->points[i*3+2]-lw->m_fixed_landmarks->points[j*3+2])
		   *(lw->m_fixed_landmarks->points[i*3+2]-lw->m_fixed_landmarks->points[j*3+2]);
		d = sqrt(d);	    
		if (d<d_nearest_neighb[i]) d_nearest_neighb[i]=d;
	    }
	}
	
	D = d_nearest_neighb[0];
	for(i=0;i<num_landmarks;i++) {
	    if (d_nearest_neighb[i]>D) D = d_nearest_neighb[i];
	    }
    
	free(d_nearest_neighb);
    }

    for(i=0; i<num_landmarks; i++)
	if (lw->cluster_id[i] == k) lw->adapt_radius[i] = 2*D;
}
	
return;
}
