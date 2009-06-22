/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "contour_statistics.h"

void print_usage (void)
{
	printf ("Usage: contour_statistics \n");
	printf ("  mode (options: global, experts,  cp, cm)\t");
	printf ("  file1\t");
	printf ("  file2\t");
	printf ("  [file3] or [filename]\t");
	printf ("  [filename]\n");
	printf ("  OPTIONS EXPLANATION: \n");
	printf ("  global= Dice coeff computation between 2 volumes (in this case file1=reference_volume, file2=warped or other volume, filename=output *.txt file\n\n\n");
	printf ("  experts= Dice coeff extension to the case of three experts - inter-rater variability computation (file1=first expert, file2=second expert, file3=third expert, filename=output *.txt file\n\n\n");
	printf ("  cp= Closest Point computation between mesh file and reference points: file1 *.obj file, file2 *.txt file with the reference points, filename output *.txt file\n\n\n");
	printf ("  cm= Closest Mesh computation: file1 *.obj file with reference mesh, file2 *.obj file with the other mesh, filename output *.txt file\n");
	exit (-1);
}

int main(int argc, char* argv[])
{
	ImgType::Pointer reference=ImgType::New();
	ImgType::Pointer warped=ImgType::New();
	ImgType::Pointer ex_1=ImgType::New();
	ImgType::Pointer ex_2=ImgType::New();
	ImgType::Pointer ex_3=ImgType::New();
	FILE* mesh;
	FILE* refMesh;
	FILE* MDpoints;
	FILE* output;

	if (argc<4)
		print_usage();

	if(strcmp("global",argv[1])==0){
		reference=load_uchar(argv[2], 0);
		warped=load_uchar(argv[3], 0);
	}else if(strcmp("experts",argv[1])==0){
		ex_1=load_uchar(argv[2], 0);
		ex_2=load_uchar(argv[3], 0);
		ex_3=load_uchar(argv[4], 0);
	}else if(strcmp("cp",argv[1])==0){
		mesh=fopen(argv[2],"r");
		MDpoints=fopen(argv[3],"r");
		if(!mesh || !MDpoints){
			fprintf(stderr,"Error: could not open the files for the cp calculation for reading!\n");
			if(!mesh)
				fprintf(stderr,"This file could not be opened: %s\n",argv[2]);
			else
				fprintf(stderr,"This file could not be opened: %s\n",argv[3]);
			exit(-1);
		}
	}else if (strcmp("cm",argv[1])==0){
		refMesh=fopen(argv[2],"r");
		mesh=fopen(argv[3],"r");
		if(!refMesh || !mesh){
			fprintf(stderr,"Error: could not open the files for the cp calculation for reading!\n");
			if(!refMesh)
				fprintf(stderr,"This file could not be opened: %s\n",argv[2]);
			else
				fprintf(stderr,"This file could not be opened: %s\n",argv[3]);
			exit(-1);
		}
	}else{
		fprintf(stderr,"Sorry! you typed in the wrong mode");
		exit(-1);
	}

	if (argc<5){
		if(strcmp("global",argv[1])==0){
			output=fopen("dice_global.txt","w");
		}else if(strcmp("experts",argv[1])==0){
			output=fopen("interrater.txt","w");
		}else if(strcmp("cp",argv[1])==0){
			output=fopen("cp_dist.txt","w");
		}else{
			output=fopen("mesh_dist.txt","w");
		}
	}else if (argc==5){
		output=fopen(argv[4],"w");
	}else if(argc==6){
		output=fopen(argv[5],"w");
	}

	if(!output){
		fprintf(stderr, "An error occurred while opening the file for writing the outputs!");
		exit(-1);
	}

	if(strcmp("global",argv[1])==0){
		do_dice_global(reference, warped, output);
	}else if(strcmp("experts",argv[1])==0){
		do_dice_expert(ex_1, ex_2, ex_3, output);
	}else if(strcmp("cp",argv[1])==0){

		SURFACE* surface=(SURFACE*)malloc(sizeof(SURFACE));
		memset(surface,0,sizeof(SURFACE));
		printf("Allocated Surface\n");

		do_cp(mesh,MDpoints,surface,output);

	}else if(strcmp("cm",argv[1])==0){
		printf("function in development\n");
		exit(-1);
	}
	return 0;
}




		//TRIANGLE_LIST* triangles=(TRIANGLE_LIST*)malloc(sizeof(TRIANGLE_LIST));
		//memset(triangles,0,sizeof(TRIANGLE_LIST));
		//triangles->num_triangles=0;
		//triangles=&surface->triangles;
		//printf("LAST: %d %d %d\n",triangles->first[triangles->num_triangles-1],
		//	triangles->second[triangles->num_triangles-1], triangles->third[triangles->num_triangles-1]);

		//MASS* center_mass=(MASS*)malloc(sizeof(MASS));
		//memset(center_mass,0,sizeof(MASS));
		//center_mass->num_triangles=0;
		//center_mass->x=(float*)malloc(sizeof(float));
		//memset(center_mass->x,0,sizeof(float));
		//center_mass->y=(float*)malloc(sizeof(float));
		//memset(center_mass->y,0,sizeof(float));
		//center_mass->z=(float*)malloc(sizeof(float));
		//memset(center_mass->z,0,sizeof(float));
		//center_mass->corrpoint_index=(int*)malloc(sizeof(int));
		//memset(center_mass->corrpoint_index,0,sizeof(int));

		//for(int r=0; r<center_mass->num_triangles; r++)
		//	printf("CORR: %d\n",center_mass->corrpoint_index[r]);



		//fclose(mesh);
