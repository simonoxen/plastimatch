/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "contour_statistics.h"

void print_usage (void)
{
	printf ("Usage: dice_stats \n");
	printf ("  mode (options: global, experts)  ");
	printf ("  file1  ");
	printf ("  file2  ");
	printf ("  [file3] or [filename]  ");
	printf ("  [filename]\n");
	printf ("  OPTIONS EXPLANATION: \n");
	printf ("  global= Dice coeff computation between 2 volumes (in this case file1=reference_volume, file2=warped or other volume, filename=output *.txt file\n\n\n");
	printf ("  experts= Dice coeff extension to the case of three experts - inter-rater variability computation (file1=first expert, file2=second expert, file3=third expert, filename=output *.txt file\n\n\n");
	exit (-1);
}

int main(int argc, char* argv[])
{
	ImgType::Pointer reference=ImgType::New();
	ImgType::Pointer warped=ImgType::New();
	ImgType::Pointer ex_1=ImgType::New();
	ImgType::Pointer ex_2=ImgType::New();
	ImgType::Pointer ex_3=ImgType::New();
	FILE* output = 0;

	if (argc<4)
		print_usage();

	if(strcmp("global",argv[1])==0){
		reference = itk_image_load_uchar(argv[2], 0);
		warped = itk_image_load_uchar(argv[3], 0);
	}else if(strcmp("experts",argv[1])==0){
		//ex_1 = itk_image_load_uchar(argv[2], 0);
		//ex_2 = itk_image_load_uchar(argv[3], 0);
		//ex_3 = itk_image_load_uchar(argv[4], 0);
	}else{
		fprintf(stderr,"Sorry! you typed in the wrong mode");
		print_usage();
		exit(-1);
	}

	if (argc<5){
		if(strcmp("global",argv[1])==0){
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
		do_dice_global(reference, warped, output, static_cast<unsigned char>(0));
	}else if(strcmp("experts",argv[1])==0){
		//do_dice_expert(ex_1, ex_2, ex_3, output);
	}
	return 0;
}
