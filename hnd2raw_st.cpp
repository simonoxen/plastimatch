/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#if defined (_DEBUG) && defined (DETECT_MEM_LEAKS)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include <memory>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cstdio>
#include <string>
#include <vector>
#include "file_util.h"
#include "plm_int.h"
#include "print_and_exit.h"
#include "timer.h"

using namespace std;

//#define DSR 1
#define DSR 2

int main(int argc, char* argv[])
{

    string hndfp, rawfp;
    char* hndDir, * hndPrefix, * hndPostfix, * rawDir, * rawPrefix, * rawPostfix;
    int startIndex, endIndex;
    fpos_t pstart;
    FILE * pFile;
    char rbuffer[100];
    char sbuffer[100];
    char ubuffer[100];
    char indexString[32];
    int lastprj=1000;
    int iniprj=0;
    Timer timer;

    FILE * angleFile;
    if (argc != 9 && argc != 10 && argc != 11)
    {
	printf("About: This program will convert a batch of hnd files to 32 bit grayscale raw files.\n\n"
	    "Usage: hnd_to_raw hnd_dir hnd_prefix hnd_postfix raw_dir raw_prefix raw_postfix start_index end_index\n"
	    "hnd_dir         - The directory of the hnd files\n"
	    "hnd_prefix      - The prefix of the hnd files\n"
	    "hnd_postfix     - The postfix of the hnd files\n"
	    "raw_dir         - The directory where the raw files will be created.\n"
	    "raw_prefix      - The prefix of the newly created raw files\n"
	    "raw_postfix     - The postfix of the raw files\n"
	    "start/end index - If the hnd_prefix is \"hnd_\", and the hnd_postfix is \"\", and the start_index "
	    "is \"0\", and the end_index is \"100\", all the files between hnd_0.hnd to hnd_100.hnd"
	    " will be converted.\n"
	    "[OPTIONAL] /f(num) This specifies the input numbered filesnames have fixed lengths (with"
	    "trailing zeros. Example: /f3 means that the files will have names like file_001, "
	    "file_002, file_003, etc.\n"
	    "[OPTIONAL] /f(num) This specifies the output numbered filesnames have fixed lengths (with"
	    "trailing zeros. Example: /f3 means that the files will have names like file_001, "
	    "file_002, file_003, etc.\n"
	    "As an example:\n"
	    "hnd2rawst.exe \"\\hnd\" \"Proj_\" \".hnd\" \"\\raw\" \"Proj_\" \".raw\" \"0\" \"688\" \"\\f5\" \"\\f3\" \n"
	);
	fflush(stdout);

	system("pause");
	return false;
    }

    plm_timer_start (&timer);

#ifndef START_INDEX
    startIndex = atoi(argv[7]);
#else
    startIndex = START_INDEX;
#endif

#ifndef END_INDEX
    endIndex = atoi(argv[8]);
#else
    endIndex = END_INDEX;
#endif

    hndDir     = argv[1];
    hndPrefix  = argv[2];
    hndPostfix = argv[3];
    rawDir     = argv[4];
    rawPrefix  = argv[5];
    rawPostfix = argv[6];
    int inFixedWidth, outFixedWidth;
    if (argc >= 10)
	inFixedWidth = atoi(argv[9] + 2);
    else
	inFixedWidth = -1;

    if (argc == 11)
	outFixedWidth = atoi(argv[10] + 2);
    else
	outFixedWidth = -1;

    angleFile=fopen((string(rawDir) + "/ProjAngles.txt").c_str(), "w");;
    if(angleFile==NULL)
    {
	printf("Cannot open angle file");
	exit(1);
    }
    for (int proj = startIndex; proj <= endIndex; ++proj)
    {
	int fileId = proj - startIndex;
	if (inFixedWidth != -1)
	    sprintf(indexString, "%0*d", inFixedWidth, proj);
	else
	    //_itoa(proj, indexString, 10);
	    snprintf (indexString, 32, "%d", proj);

	hndfp = hndDir;
	hndfp += '/';
	hndfp += hndPrefix;
	hndfp += indexString;
	hndfp += hndPostfix;

	printf ("Looking for %s\n", hndfp.c_str());

	//if (GetFileAttributes(hndfp.c_str()) == INVALID_FILE_ATTRIBUTES)
	if (!file_exists (hndfp.c_str()))
	{
	    printf("Error: hnd_file (%s) does not exist.\n", hndfp.c_str());
	    return 0;
	}

	if (outFixedWidth != inFixedWidth)
	{
	    if (outFixedWidth != -1)
		sprintf(indexString, "%0*d", outFixedWidth, proj);
	    else
		//_itoa(proj, indexString, 10);
		snprintf (indexString, 32, "%d", proj);
	}

	rawfp = rawDir;
	rawfp += '/';
	rawfp += rawPrefix;
	rawfp += indexString;
	rawfp += rawPostfix;

	printf ("Opening %s\n", rawfp.c_str());

	pFile = fopen(hndfp.c_str(), "rb");


	if(pFile==NULL){
	    printf("Cannot open %s open error\n",rbuffer);
	    exit(1);
	}
	fgetpos(pFile, &pstart);
	unsigned char *buffer=(unsigned char *)malloc(10000000);
	if(buffer==NULL)
	{
	    printf("malloc error");
	    exit(1);
	}
	int TFileLength=fread(buffer,1,10000000,pFile);
	free(buffer);
	fsetpos(pFile, &pstart);

	char   sFileType[32];
	uint32_t    FileLength;
	char   sChecksumSpec[4];
	uint32_t    nCheckSum;
	char   sCreationDate[8];
	char   sCreationTime[8];
	char   sPatientID[16];
	uint32_t    nPatientSer;
	char   sSeriesID[16];
	uint32_t    nSeriesSer;
	char   sSliceID[16];
	uint32_t    nSliceSer;
	uint32_t    SizeX;
	uint32_t    SizeY;
	double dSliceZPos;
	char   sModality[16];
	uint32_t    nWindow;
	uint32_t    nLevel;
	uint32_t    nPixelOffset;
	char   sImageType[4];
	double dGantryRtn;
	double dSAD;
	double dSFD;
	double dCollX1;
	double dCollX2;
	double dCollY1;
	double dCollY2;
	double dCollRtn;
	double dFieldX;
	double dFieldY;
	double dBladeX1;
	double dBladeX2;
	double dBladeY1;
	double dBladeY2;
	double dIDUPosLng;
	double dIDUPosLat;
	double dIDUPosVrt;
	double dIDUPosRtn;
	double dPatientSupportAngle;
	double dTableTopEccentricAngle;
	double dCouchVrt;
	double dCouchLng;
	double dCouchLat;
	double dIDUResolutionX;
	double dIDUResolutionY;
	double dImageResolutionX;
	double dImageResolutionY;
	double dEnergy;
	double dDoseRate;
	double dXRayKV;
	double dXRayMA;
	double dMetersetExposure;
	double dAcqAdjustment;
	double dCTProjectionAngle;
	double dCTNormChamber;
	double dGatingTimeTag;
	double dGating4DInfoX;
	double dGating4DInfoY;
	double dGating4DInfoZ;
	double dGating4DInfoTime;

	printf ("Reading...\n");

	fread(( void *)sFileType, sizeof(char), 32, pFile);
	fread(( void *)&FileLength, sizeof(uint32_t), 1, pFile);
	fread(( void *)sChecksumSpec, sizeof(char), 4, pFile);
	fread(( void *)&nCheckSum, sizeof(uint32_t), 1, pFile);
	fread(( void *)sCreationDate, sizeof(char), 8, pFile);
	fread(( void *)sCreationTime, sizeof(char), 8, pFile);
	fread(( void *)sPatientID, sizeof(char), 16, pFile);
	fread(( void *)&nPatientSer, sizeof(uint32_t), 1, pFile);
	fread(( void *)sSeriesID, sizeof(char), 16, pFile);
	fread(( void *)&nSeriesSer, sizeof(uint32_t), 1, pFile);
	fread(( void *)sSliceID, sizeof(char), 16, pFile);
	fread(( void *)&nSliceSer, sizeof(uint32_t), 1, pFile);
	fread(( void *)&SizeX, sizeof(uint32_t), 1, pFile);
	fread(( void *)&SizeY, sizeof(uint32_t), 1, pFile);
	fread(( void *)&dSliceZPos, sizeof(double), 1, pFile);
	fread(( void *)sModality, sizeof(char), 16, pFile);
	fread(( void *)&nWindow, sizeof(uint32_t), 1, pFile);
	fread(( void *)&nLevel, sizeof(uint32_t), 1, pFile);
	fread(( void *)&nPixelOffset, sizeof(uint32_t), 1, pFile);
	fread(( void *)sImageType, sizeof(char), 4, pFile);
	fread(( void *)&dGantryRtn, sizeof(double), 1, pFile);
	fread(( void *)&dSAD, sizeof(double), 1, pFile);
	fread(( void *)&dSFD, sizeof(double), 1, pFile);
	fread(( void *)&dCollX1, sizeof(double), 1, pFile);
	fread(( void *)&dCollX2, sizeof(double), 1, pFile);
	fread(( void *)&dCollY1, sizeof(double), 1, pFile);
	fread(( void *)&dCollY2, sizeof(double), 1, pFile);
	fread(( void *)&dCollRtn, sizeof(double), 1, pFile);
	fread(( void *)&dFieldX, sizeof(double), 1, pFile);
	fread(( void *)&dFieldY, sizeof(double), 1, pFile);
	fread(( void *)&dBladeX1, sizeof(double), 1, pFile);
	fread(( void *)&dBladeX2, sizeof(double), 1, pFile);
	fread(( void *)&dBladeY1, sizeof(double), 1, pFile);
	fread(( void *)&dBladeY2, sizeof(double), 1, pFile);
	fread(( void *)&dIDUPosLng, sizeof(double), 1, pFile);
	fread(( void *)&dIDUPosLat, sizeof(double), 1, pFile);
	fread(( void *)&dIDUPosVrt, sizeof(double), 1, pFile);
	fread(( void *)&dIDUPosRtn, sizeof(double), 1, pFile);
	fread(( void *)&dPatientSupportAngle, sizeof(double), 1, pFile);
	fread(( void *)&dTableTopEccentricAngle, sizeof(double), 1, pFile);
	fread(( void *)&dCouchVrt, sizeof(double), 1, pFile);
	fread(( void *)&dCouchLng, sizeof(double), 1, pFile);
	fread(( void *)&dCouchLat, sizeof(double), 1, pFile);
	fread(( void *)&dIDUResolutionX, sizeof(double), 1, pFile);
	fread(( void *)&dIDUResolutionY, sizeof(double), 1, pFile);
	fread(( void *)&dImageResolutionX, sizeof(double), 1, pFile);
	fread(( void *)&dImageResolutionY, sizeof(double), 1, pFile);
	fread(( void *)&dEnergy, sizeof(double), 1, pFile);
	fread(( void *)&dDoseRate, sizeof(double), 1, pFile);
	fread(( void *)&dXRayKV, sizeof(double), 1, pFile);
	fread(( void *)&dXRayMA, sizeof(double), 1, pFile);
	fread(( void *)&dMetersetExposure, sizeof(double), 1, pFile);
	fread(( void *)&dAcqAdjustment, sizeof(double), 1, pFile);
	fread(( void *)&dCTProjectionAngle, sizeof(double), 1, pFile);
	fread(( void *)&dCTNormChamber, sizeof(double), 1, pFile);
	fread(( void *)&dGatingTimeTag, sizeof(double), 1, pFile);
	fread(( void *)&dGating4DInfoX, sizeof(double), 1, pFile);
	fread(( void *)&dGating4DInfoY, sizeof(double), 1, pFile);
	fread(( void *)&dGating4DInfoZ, sizeof(double), 1, pFile);
	fread(( void *)&dGating4DInfoTime, sizeof(double), 1, pFile);


	unsigned char *SkipHead=(unsigned char *)malloc(sizeof(unsigned char)*1024);

	int LookupTableSize=(int)((float)SizeX*((float)SizeY-1.0)/4.0+0.5);

	unsigned char *LookupTable=(unsigned char *)malloc(sizeof(unsigned char)*LookupTableSize);

	int CDataLength=(TFileLength-1024-LookupTableSize);
	unsigned char *CD=(unsigned char *)malloc(sizeof(unsigned char)*CDataLength);

	int *LookupType=(int *)malloc(sizeof(int)*(SizeX*SizeY+4));
	long *Img=(long *)malloc(sizeof(long)*SizeX*SizeY);
	//unsigned short *uImg=(unsigned short*)malloc(sizeof(unsigned short)*SizeX*SizeY);
	unsigned short *uImg=(unsigned short*)malloc(sizeof(unsigned short)*SizeX*SizeY/DSR/DSR);

	if (!SkipHead || !LookupTable || !CD || !LookupType || !Img) {
	    print_and_exit ("Malloc error\n");
	}

	int hFileLength = FileLength>>16;
	int lFileLength = FileLength - hFileLength<<16;
	int fFileLength = lFileLength<<16 + hFileLength;

	printf ("Setting position...\n");

	fsetpos(pFile, &pstart);
	fread(SkipHead, 1, 1024, pFile); 

	int ActLook=fread(LookupTable,1,LookupTableSize,pFile);
	int k=SizeX+1;

	for(int i=0;i<LookupTableSize; i++)
	{
	    int sLook0=3&(LookupTable[i]>>(0*2));
	    int sLook1=3&(LookupTable[i]>>(1*2));
	    int sLook2=3&(LookupTable[i]>>(2*2));
	    int sLook3=3&(LookupTable[i]>>(3*2));
	    LookupType[i*4+SizeX+1]=sLook0;
	    LookupType[i*4+SizeX+1+1]=sLook1;
	    LookupType[i*4+SizeX+1+2]=sLook2;
	    LookupType[i*4+SizeX+1+3]=sLook3;	
	}



	int ActCDLength=fread(CD,1,CDataLength,pFile);
	fclose(pFile);

	unsigned char *pCD;
	char * pchar;
	short * pshort;
	long * plong;
	int x,y;

	int CDi=0;

	y=0;
	plong=(long *)CD;
	for(x=0; x<SizeX; x++)
	{
	    Img[y*SizeX+x]=*plong;
	    plong++;
	    CDi+=4;
	}
	y=1;
	x=0;
	Img[y*SizeX+x]=*plong;
	CDi+=4;

	plong++;

	pCD=(unsigned char*)plong;

	for(y=1; y<SizeY; y++)
	{
	    for(x=0; x<SizeX; x++)
	    {
		if(y==1&&x==0)
		{
		    continue;
		}
		switch (LookupType[y*SizeX+x]){
		case 0:
		    {
			pchar=(char *)pCD;
			Img[y*SizeX+x]=(long)*pchar;
			pchar++;
			pCD=(unsigned char *)pchar;
			CDi++;
			break;
		    }
		case 1:
		    {
			pshort=(short *)pCD;
			Img[y*SizeX+x]=(long)*pshort;
			pshort++;
			pCD=(unsigned char *)pshort;
			CDi+=2;
			break;
		    }
		case 2:
		    {
			plong=(long *)pCD;
			Img[y*SizeX+x]=(long)*plong;
			plong++;
			pCD=(unsigned char *)plong;
			CDi+=4;
			break;
		    }
		}//switch
	    }//for x
	}//for y

	/**/
	bool overfloat=false;
	for(y=1; y<SizeY; y++)
	{
	    for(x=0; x<SizeX; x++)
	    {
		if(y==1&&x==0)
		{
		    continue;
		}
		Img[y*SizeX+x]=Img[y*SizeX+x] // longDiff
		    - Img[(y-1)*SizeX+(x-1)]  // - A
		    + Img[y*SizeX+(x-1)]           // + C
		    + Img[(y-1)*SizeX+x];//+B   
		//		if(Img[y*SizeX+x]>(1<<16)||Img[y*SizeX+x]<0){
		//			printf("uImg[] int overfloat.\n");
		//			overfloat=true;
		//			}
	    }
	}
	//	
	//int S=SizeX*SizeY;
	//for(i=0; i<S; i++){
	//		uImg[i]=Img[i];
	//}

	//		printf("rows=%d, columns=%d\n",SizeY, SizeX);
	/**/
	fprintf(angleFile,"%f\n",dCTProjectionAngle);
	printf("%f;\n",dCTProjectionAngle);


	int SizeYD=SizeY/DSR;
	int SizeXD=SizeX/DSR;
	for(int i=0; i<SizeYD; i++)
	{
	    for(int j=0; j<SizeXD; j++)
	    {
		float temp=0.0;
		for(int di=0; di<DSR; di++)
		{
		    for(int dj=0; dj<DSR; dj++)
		    {
			temp+=Img[(DSR*i+di)*SizeX+DSR*j+dj];
		    }
		}
		uImg[i*SizeXD+j]=0.4*temp/DSR/DSR;
	    }
	}

	//if(overfloat==false){
	FILE * wp=fopen(rawfp.c_str(), "wb");;
	if(pFile==NULL){
	    printf("file open error.\n");
	}
	int wi=fwrite(uImg,sizeof(unsigned short),SizeXD*SizeYD,wp);
		
	printf((string(hndPrefix) + indexString + hndPostfix + " -> "
		+ rawPrefix + indexString + rawPostfix + "\n").c_str());
	fflush(stdout);
	fclose(wp);
	//}
	/**else{
	   FILE * wp=fopen(sbuffer,"wb");
	   if(pFile==NULL){
	   printf("file open error.\n");
	   }
	   fwrite(Img,sizeof(long),SizeX*SizeY,wp);
	   printf("%s = >%s (int32)\n",rbuffer,sbuffer);		
	   }
	**/





	free(SkipHead);
	free(LookupTable);
	free(LookupType);
	free(CD);
	free(Img);
	free(uImg);

    }//prj
    printf("finished\n");
    fclose(angleFile);
    printf("Took %f s\n", plm_timer_report (&timer));
    fflush(stdout);
    return 0;
}

