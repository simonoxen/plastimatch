/*===========================================================
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
===========================================================*/
#include "contour_statistics.h"

void print_usage (void)
{
	/*printf ("This executable computes the DICE coefficient for 2 give binary volumes \n");*/
	printf ("Usage: contour_statistics \n");
	printf ("  mode (options: global, experts,  cp)\t");
	printf ("  file1\t");
	printf ("  file2\t");
	printf ("  [file3] or [filename]\t");
	printf ("  [filename]\n");
	printf ("  OPTIONS EXPLANATION: \n");
	printf ("  case Dice's coeff computation: the user can choose between calculating it for the whole volume (mode=global).\n NB: in this case file1=reference volume, file2=warped volume\n\n");
	printf ("  case Experts: this is designed to compute the inter-raters variability. It will compute the overlapping region between the three volumes and divide it by the union.\n");
	printf ("  case Closest Point computation: mode should be set to 'cp' , file1 should be your *.obj file and file2 should be your *.txt file with the points from the physician\n\n");
	//printf ("  filename: the user can specify (optional) a filename for the output file in which the program should write the outputs (either the Dice's coeff values and volume ovelap % or the distances from the mesh for the cp calculation.\n");
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
	FILE* MDpoints;
	FILE* output;

	if (argc<4)
		print_usage();

	//if(strcmp("global",argv[1])==0 || strcmp("slice",argv[1])==0){
	if(strcmp("global",argv[1])==0){
		reference=load_uchar(argv[2]);
		warped=load_uchar(argv[3]);
	}else if(strcmp("experts",argv[1])==0){
		ex_1=load_uchar(argv[2]);
		ex_2=load_uchar(argv[3]);
		ex_3=load_uchar(argv[4]);
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
	}else{
		fprintf(stderr,"Sorry! you typed in the wrong mode");
		exit(-1);
	}

	if (argc<5){
		if(strcmp("cp",argv[1])==0){
			output=fopen("cp_dist.txt","w");
		//}else if(strcmp("slice",argv[1])==0){
		//	output=fopen("dice_slice.txt","w");
		}else if(strcmp("global",argv[1])==0){
			output=fopen("dice_global.txt","w");
		}else if(strcmp("experts",argv[1])==0){
			output=fopen("interrater.txt","w");
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
	//}else if(strcmp("slice",argv[1])==0){
	//	do_dice_slice(reference, warped, output);
	}else if(strcmp("experts",argv[1])==0){
		do_dice_expert(ex_1, ex_2, ex_3, output);
	}else if(strcmp("cp",argv[1])==0){

		SURFACE* surface=(SURFACE*)malloc(sizeof(SURFACE));
		memset(surface,0,sizeof(SURFACE));
		printf("Allocated Surface\n");

		do_cp(mesh,MDpoints,surface,output);




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
	}
	return 0;
}