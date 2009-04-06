/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "contour_statistics.h"
#define BUFLEN 2048

void calculate_mass(SURFACE* surface){

	float x_sum=0;
	float y_sum=0;
	float z_sum=0;
	
	VERTICES_LIST* vertices=(VERTICES_LIST*)malloc(sizeof(VERTICES_LIST));
	TRIANGLE_LIST* triangles=(TRIANGLE_LIST*)malloc(sizeof(TRIANGLE_LIST));
	MASS* center_mass=(MASS*)malloc(sizeof(MASS));
	
	memset(vertices,0,sizeof(VERTICES_LIST));
	memset(triangles,0,sizeof(TRIANGLE_LIST));
	memset(center_mass,0,sizeof(MASS));
	vertices->num_vertices=0;
	triangles->num_triangles=0;
	center_mass->num_triangles=0;

	vertices->x=(float*)malloc(sizeof(float));
	memset(vertices->x,0,sizeof(float));
	vertices->y=(float*)malloc(sizeof(float));
	memset(vertices->y,0,sizeof(float));
	vertices->z=(float*)malloc(sizeof(float));
	memset(vertices->z,0,sizeof(float));

	triangles->first=(int*)malloc(sizeof(int));
	memset(triangles->first,0,sizeof(int));
	triangles->second=(int*)malloc(sizeof(int));
	memset(triangles->second,0,sizeof(int));
	triangles->third=(int*)malloc(sizeof(int));
	memset(triangles->third,0,sizeof(int));

	center_mass->x=(float*)malloc(sizeof(float));
	memset(center_mass->x,0,sizeof(float));
	center_mass->y=(float*)malloc(sizeof(float));
	memset(center_mass->y,0,sizeof(float));
	center_mass->z=(float*)malloc(sizeof(float));
	memset(center_mass->z,0,sizeof(float));
	//center_mass->triangle_index=(int*)malloc(sizeof(int));
	//memset(center_mass->triangle_index,0,sizeof(int));

	vertices=&surface->vertices;
	triangles=&surface->triangles;
	center_mass=&surface->centres;

	center_mass->num_triangles=triangles->num_triangles;

	for(int i=0; i<triangles->num_triangles; i++){
			center_mass->x=(float*)realloc(center_mass->x,(i+1)*sizeof(float));
			center_mass->y=(float*)realloc(center_mass->y,(i+1)*sizeof(float));
			center_mass->z=(float*)realloc(center_mass->z,(i+1)*sizeof(float));

			x_sum=vertices->x[triangles->first[i]-1]+vertices->x[triangles->second[i]-1]+vertices->x[triangles->third[i]-1];
			y_sum=vertices->y[triangles->first[i]-1]+vertices->y[triangles->second[i]-1]+vertices->y[triangles->third[i]-1];
			z_sum=vertices->z[triangles->first[i]-1]+vertices->z[triangles->second[i]-1]+vertices->z[triangles->third[i]-1];

			center_mass->x[i]=x_sum/3;
			center_mass->y[i]=y_sum/3;
			center_mass->z[i]=z_sum/3;

			//printf("INDEX: %d %d %d\n",triangles->first[i],triangles->second[i],triangles->third[i]);
			//printf("CENTER OF MASS: %f %f %f\n",center_mass->x[i],center_mass->y[i],center_mass->z[i]);
	}

}




void cp(SURFACE* surface){

	float x_min=0;
	float y_min=0;
	float z_min=0;
	float d=0;
	float d_prev=0;
	//FILE* test;
	
	VERTICES_LIST* vertices=(VERTICES_LIST*)malloc(sizeof(VERTICES_LIST));
	VERTICES_LIST* MDpoints=(VERTICES_LIST*)malloc(sizeof(VERTICES_LIST));
	MASS* center_mass=(MASS*)malloc(sizeof(MASS));
	CORR* corresp=(CORR*)malloc(sizeof(CORR));
	

	memset(vertices,0,sizeof(VERTICES_LIST));
	memset(MDpoints,0,sizeof(VERTICES_LIST));
	memset(center_mass,0,sizeof(MASS));
	memset(corresp,0,sizeof(CORR));
	vertices->num_vertices=0;
	MDpoints->num_vertices=0;
	center_mass->num_triangles=0;
	corresp->num_points=0;

	vertices->x=(float*)malloc(sizeof(float));
	memset(vertices->x,0,sizeof(float));
	vertices->y=(float*)malloc(sizeof(float));
	memset(vertices->y,0,sizeof(float));
	vertices->z=(float*)malloc(sizeof(float));
	memset(vertices->z,0,sizeof(float));

	MDpoints->x=(float*)malloc(sizeof(float));
	memset(MDpoints->x,0,sizeof(float));
	MDpoints->y=(float*)malloc(sizeof(float));
	memset(MDpoints->y,0,sizeof(float));
	MDpoints->z=(float*)malloc(sizeof(float));
	memset(MDpoints->z,0,sizeof(float));

	center_mass->x=(float*)malloc(sizeof(float));
	memset(center_mass->x,0,sizeof(float));
	center_mass->y=(float*)malloc(sizeof(float));
	memset(center_mass->y,0,sizeof(float));
	center_mass->z=(float*)malloc(sizeof(float));
	memset(center_mass->z,0,sizeof(float));

	corresp->corrpoint_index=(int*)malloc(sizeof(int));
	memset(corresp->corrpoint_index,0,sizeof(int));
	
	corresp=&surface->correspondance;
	MDpoints=&surface->MDpoints;
	center_mass=&surface->centres;
	corresp->num_points=MDpoints->num_vertices;
	//printf("NUM CORR: %d \t NUM MDpoints:%d",corresp->num_points,MDpoints->num_vertices);
	corresp->corrpoint_index=(int*)realloc(corresp->corrpoint_index,
		corresp->num_points*sizeof(int));
	//test=fopen("testing_bari.txt","w");
	for(int k=0; k<MDpoints->num_vertices; k++){
		d=999999999;
		d_prev=999999999;
		for(int i=0; i<center_mass->num_triangles; i++){
			x_min=center_mass->x[i]-MDpoints->x[k];
			y_min=center_mass->y[i]-MDpoints->y[k];
			z_min=center_mass->z[i]-MDpoints->z[k];
			d=(x_min*x_min)+(y_min*y_min)+(z_min*z_min);
			
			if(d<d_prev){
					corresp->corrpoint_index[k]=i;
					d_prev=d;
			}
			x_min=0;
			y_min=0;
			z_min=0;
		}
		//fprintf(test,"%f %f %f\n",center_mass->x[ center_mass->corrpoint_index[k]], center_mass->y[ center_mass->corrpoint_index[k]],center_mass->z[ center_mass->corrpoint_index[k]]);
	}
	//printf("NUM CORR: %d \t NUM MDpoints:%d\n",corresp->num_points,MDpoints->num_vertices);

	//for(int r=0; r<center_mass->num_triangles/100; r++)
	//		printf("CORR: %d\n",center_mass->corrpoint_index[r]);
	//fclose(test);
}
void compute_plane(SURFACE* surface){
	float matrix[3][3];
	int i=0;

	VERTICES_LIST* vertices=(VERTICES_LIST*)malloc(sizeof(VERTICES_LIST));
	PLANE* plane=(PLANE*)malloc(sizeof(PLANE));
	TRIANGLE_LIST* triangles=(TRIANGLE_LIST*)malloc(sizeof(TRIANGLE_LIST));
	CORR* corresp=(CORR*)malloc(sizeof(CORR));
	
	memset(vertices,0,sizeof(VERTICES_LIST));
	memset(triangles,0,sizeof(TRIANGLE_LIST));
	memset(corresp,0,sizeof(CORR));
	memset(plane,0,sizeof(PLANE));

	vertices->num_vertices=0;
	triangles->num_triangles=0;
	corresp->num_points=0;
	plane->num_planes=0;

	vertices->x=(float*)malloc(sizeof(float));
	memset(vertices->x,0,sizeof(float));
	vertices->y=(float*)malloc(sizeof(float));
	memset(vertices->y,0,sizeof(float));
	vertices->z=(float*)malloc(sizeof(float));
	memset(vertices->z,0,sizeof(float));

	corresp->corrpoint_index=(int*)malloc(sizeof(int));
	memset(corresp->corrpoint_index,0,sizeof(int));	

	triangles->first=(int*)malloc(sizeof(int));
	memset(triangles->first,0,sizeof(int));
	triangles->second=(int*)malloc(sizeof(int));
	memset(triangles->second,0,sizeof(int));
	triangles->third=(int*)malloc(sizeof(int));
	memset(triangles->third,0,sizeof(int));

	plane->a0=(float*)malloc(sizeof(float));
	memset(plane->a0,0,sizeof(float));
	plane->a1=(float*)malloc(sizeof(float));
	memset(plane->a1,0,sizeof(float));
	plane->a2=(float*)malloc(sizeof(float));
	memset(plane->a2,0,sizeof(float));
	plane->a3=(float*)malloc(sizeof(float));
	memset(plane->a3,0,sizeof(float));
	//plane->a0=0;
	//plane->a1=0;
	//plane->a2=0;
	//plane->a3=0;
	
	vertices=&surface->vertices;
	//center_mass=&surface->centres;
	triangles=&surface->triangles;
	corresp=&surface->correspondance;
	plane=&surface->planes;
	plane->num_planes=corresp->num_points;
	//printf("corresp->num_points: %d NUMPLANES: %d\n", corresp->num_points, plane->num_planes);

	plane->a0=(float*)realloc(plane->a0,corresp->num_points*sizeof(float));
	plane->a1=(float*)realloc(plane->a1,corresp->num_points*sizeof(float));
	plane->a2=(float*)realloc(plane->a2,corresp->num_points*sizeof(float));
	plane->a3=(float*)realloc(plane->a3,corresp->num_points*sizeof(float));

	for(i=0; i<corresp->num_points; i++){
		//printf("corresp->num_points: %d", corresp->num_points);
		//exit(-1);
		//a0
		matrix[0][0]=vertices->x[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][0]=vertices->x[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][0]=vertices->x[triangles->third[corresp->corrpoint_index[i]]-1];
		matrix[0][1]=vertices->y[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][1]=vertices->y[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][1]=vertices->y[triangles->third[corresp->corrpoint_index[i]]-1];
		matrix[0][2]=vertices->z[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][2]=vertices->z[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][2]=vertices->z[triangles->third[corresp->corrpoint_index[i]]-1];
		//printf("TRIANGOLO: %d\n", corresp->corrpoint_index[i]);
		//printf("INDEX POINT: %d %d %d\n", triangles->first[corresp->corrpoint_index[i]]-1,triangles->second[corresp->corrpoint_index[i]]-1, triangles->third[corresp->corrpoint_index[i]]-1);
		//printf("MATRIX:\n");
		//printf("%f %f %f\n",matrix[0][0],matrix[0][1],matrix[0][2]);
		//printf("%f %f %f\n",matrix[1][0],matrix[1][1],matrix[1][2]);
		//printf("%f %f %f\n",matrix[2][0],matrix[2][1],matrix[2][2]);

		

		plane->a0[i]=matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-matrix[1][0]*(matrix[0][1]*matrix[2][2]-matrix[2][1]*matrix[0][2])+matrix[2][0]*(matrix[0][1]*matrix[1][2]-matrix[1][1]*matrix[0][2]);
		//printf("a0= %f\n", plane->a0[i]);
		//exit(-1);

		//a1
		matrix[0][0]=1;
		matrix[1][0]=1;
		matrix[2][0]=1;
		matrix[0][1]=vertices->y[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][1]=vertices->y[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][1]=vertices->y[triangles->third[corresp->corrpoint_index[i]]-1];
		matrix[0][2]=vertices->z[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][2]=vertices->z[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][2]=vertices->z[triangles->third[corresp->corrpoint_index[i]]-1];

		plane->a1[i]=matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-matrix[1][0]*(matrix[0][1]*matrix[2][2]-matrix[2][1]*matrix[0][2])+matrix[2][0]*(matrix[0][1]*matrix[1][2]-matrix[1][1]*matrix[0][2]);
		//printf("a1= %f\n", plane->a1[i]);
		//a2
		matrix[0][0]=vertices->x[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][0]=vertices->x[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][0]=vertices->x[triangles->third[corresp->corrpoint_index[i]]-1];
		matrix[0][1]=1;
		matrix[1][1]=1;
		matrix[2][1]=1;
		matrix[0][2]=vertices->z[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][2]=vertices->z[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][2]=vertices->z[triangles->third[corresp->corrpoint_index[i]]-1];

		plane->a2[i]=matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-matrix[1][0]*(matrix[0][1]*matrix[2][2]-matrix[2][1]*matrix[0][2])+matrix[2][0]*(matrix[0][1]*matrix[1][2]-matrix[1][1]*matrix[0][2]);
		//printf("a2= %f\n", plane->a2[i]);
		//a3
		matrix[0][0]=vertices->x[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][0]=vertices->x[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][0]=vertices->x[triangles->third[corresp->corrpoint_index[i]]-1];
		matrix[0][1]=vertices->y[triangles->first[corresp->corrpoint_index[i]]-1];
		matrix[1][1]=vertices->y[triangles->second[corresp->corrpoint_index[i]]-1];
		matrix[2][1]=vertices->y[triangles->third[corresp->corrpoint_index[i]]-1];
		matrix[0][2]=1;
		matrix[1][2]=1;
		matrix[2][2]=1;
		
		plane->a3[i]=matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-matrix[1][0]*(matrix[0][1]*matrix[2][2]-matrix[2][1]*matrix[0][2])+matrix[2][0]*(matrix[0][1]*matrix[1][2]-matrix[1][1]*matrix[0][2]);
		//printf("a3= %f\n", plane->a3[i]);
		//exit(-1);

	}
	//printf("FIRST PLANE: %f %f %f %f",plane->a0[0],plane->a1[0], plane->a2[0], plane->a3[0]);
	//exit(-1);
}
void do_cp(FILE* mesh,FILE* MDpoints, SURFACE* surface, FILE* output){

	float d=0;
	float pt[3];
	float coeff[4];
	
	VERTICES_LIST* points=(VERTICES_LIST*)malloc(sizeof(VERTICES_LIST));
	PLANE* plane=(PLANE*)malloc(sizeof(PLANE));
				
	memset(points,0,sizeof(VERTICES_LIST));
	memset(plane,0,sizeof(PLANE));

	points->num_vertices=0;
	plane->num_planes=0;

	points->x=(float*)malloc(sizeof(float));
	memset(points->x,0,sizeof(float));
	points->y=(float*)malloc(sizeof(float));
	memset(points->y,0,sizeof(float));
	points->z=(float*)malloc(sizeof(float));
	memset(points->z,0,sizeof(float));

	plane->a0=(float*)malloc(sizeof(float));
	memset(plane->a0,0,sizeof(float));
	plane->a1=(float*)malloc(sizeof(float));
	memset(plane->a1,0,sizeof(float));
	plane->a2=(float*)malloc(sizeof(float));
	memset(plane->a2,0,sizeof(float));
	plane->a3=(float*)malloc(sizeof(float));
	memset(plane->a3,0,sizeof(float));

	read_obj(mesh,surface);
	calculate_mass(surface);
	read_MDcontours(MDpoints,surface);
	cp(surface);
	compute_plane(surface);
	//printf("FIRST PLANE: %f %f %f %f",plane->a0[0],plane->a1[0], plane->a2[0], plane->a3[0]);
	//exit(-1);

	points=&surface->MDpoints;
	plane=&surface->planes;	
	plane->num_planes=points->num_vertices;

	//printf("NUMPLANES: %d",plane->num_planes);
	//printf("FIRST PLANE: %f %f %f %f",plane->a0[0],plane->a1[0], plane->a2[0], plane->a3[0]);
	//exit(-1);

	for(int i=0; i<plane->num_planes; i++){
		pt[0]=points->x[i];
		pt[1]=points->y[i];
		pt[2]=points->z[i];
		coeff[0]=plane->a0[i];
		coeff[1]=plane->a1[i];
		coeff[2]=plane->a2[i];
		coeff[3]=plane->a3[i];

		d=abs((coeff[1]*pt[0]+coeff[2]*pt[1]+coeff[3]*pt[2]-coeff[0])/sqrt(coeff[1]*coeff[1]+coeff[2]*coeff[2]+coeff[3]*coeff[3]));
		//fprintf(output,"%f %f %f %f %f\n",plane->a0[i],plane->a1[i],plane->a2[i],plane->a3[i],d);
		fprintf(output,"%f\n",d);
		//exit(-1);

	}
	fclose(output);

}

