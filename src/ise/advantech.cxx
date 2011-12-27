/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <string>
#include <stdio.h>
#include <windows.h>
#include "advantech.h"

/* Card is Advantech PCI-1760U */
/* USB box is Advantech USB-4761 */
#include "driver.h"  /* Advantech */
#include "os.h"      /* Advantech */

#if (ADVANTECH_FOUND)
static PT_DioWritePortByte ptDioWritePortByte;
static PT_DioReadPortByte ptDioReadPortByte;
#endif /* ADVANTECH_FOUND */

Advantech::Advantech ()
{
    LRESULT rc;
    SHORT num_devices;

    this->have_device = false;
    this->device_number = 0;

    /* Find number of devices */
    rc = DRV_DeviceGetNumOfList(&num_devices);
    if (rc != SUCCESS) {
	char error_msg[80];
	DRV_GetErrorMessage (rc, error_msg);
	printf ("DRV_DeviceGetNumOfList returned %d (%s)\n",
	    rc, error_msg);
	exit (-1);
    }
    printf ("Advantech: %d devices found.\n", (int) num_devices);

    /* Retrieve device list */
    DEVLIST* device_list = (DEVLIST*) malloc (num_devices * sizeof(DEVLIST));
    SHORT num_out_entries;
    rc = DRV_DeviceGetList (device_list, num_devices, &num_out_entries);
    if (rc != SUCCESS) {
	char error_msg[80];
	DRV_GetErrorMessage (rc, (LPSTR)error_msg);
	printf ("DRV_DeviceGetNumOfList returned %d (%s)\n",
	    rc, error_msg);
	exit (-1);
    }

    /* Find the USB device */
    for (int i = 0; i < num_devices; i++) {
	std::string device_name = device_list[i].szDeviceName;
	std::string usb_device_name ("USB-4761");
	printf ("Advantech device %2d: %d, %s\n", 
	    i, device_list[i].dwDeviceNum, device_list[i].szDeviceName);
	if (!device_name.compare (0, usb_device_name.size(), usb_device_name))
	{
	    if (!this->have_device) {
		printf ("  >> found USB-4761 at device %d\n", 
		    device_list[i].dwDeviceNum);
		this->have_device = true;
		this->device_number = device_list[i].dwDeviceNum;
	    }
	}
    }

    if (!this->have_device) {
	printf ("Sorry, no USB device found\n");
	exit (-1);
    }

    /* Open the device */
    rc = DRV_DeviceOpen (this->device_number, &this->driver_handle);
}

Advantech::~Advantech ()
{
    if (this->have_device) {
	DRV_DeviceClose (&this->driver_handle);
    }
}

void
do_something (void)
{
#if defined (commentout)
    int result = HCP_NO_ERR;
    UQueryProgInfo crntStatus;
    SModeInfo  modeInfo;

    time_t clock_valid; 
    time_t clock_relay;
    time_t clock_complete;
    time_t clock_valid2;
    time_t clock_complete2;


    result = GetModeInfo(modeInfo);
    if (result != HCP_NO_ERR)
    {
	printf("*** unable to get mode info\n");
	return result;
    }
////////////////////////////////////log files
    FILE * LogFile;
    LogFile = fopen("log.txt", "a");
    time_t clock_call = clock();
    fprintf(LogFile, "Function Called at time = %g\n" , clock_call/(double)CLOCKS_PER_SEC );
//////////////////////////////////////////	
	
    printf("Enabling software handshaking...\n");
    printf ("BEFORE ENABLE SW HANDSHAKE\n");
    QueryProgress__();
    result = vip_enable_sw_handshaking(TRUE);
    if (result != HCP_NO_ERR)
    {
	printf("*** vip_enable_sw_handshaking returned error %d\n", result);
	return result;
    }

    result = vip_sw_handshaking(VIP_SW_PREPARE, 1);

    switch (AqState)
    {
//////////////////////////////////Free Run Single Frame Acquisition//////////////////////////////////
    case 1:				
	fprintf(LogFile, "Free Run Single Frame Acquisition\n");
	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 1);		//X-Rays Valid TRUE
	clock_valid = clock();

	ptDioWritePortByte.state = 1;							//Relay Close
	DRV_DioWritePortByte(DriverHandle,(LPT_DioWritePortByte)&ptDioWritePortB
yte);
	clock_relay = clock();

	result = QueryWaitOnComplete(crntStatus, 0);			//Wait For Image
	clock_complete = clock();

	result = GetImageToFile(modeInfo.ColsPerFrame, modeInfo.LinesPerFrame, "newimage.raw");		//Get Image to File
	break;

///////////////////////////Vip Reset Preceded Single Frame Acquisition////////////////////////////
    case 2:			
	fprintf(LogFile, "Vip Reset Preceded Single Frame Acquisition\n");
	vip_reset_state();										//VIP RESET

	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 1);		//X-Rays Valid TRUE
	clock_valid = clock();

	ptDioWritePortByte.state = 1;							//Relay Close
	DRV_DioWritePortByte(DriverHandle,(LPT_DioWritePortByte)&ptDioWritePortByte);
	clock_relay = clock();

	result = QueryWaitOnComplete(crntStatus, 0);			//Wait For Image
	clock_complete = clock();

	result = GetImageToFile(modeInfo.ColsPerFrame, modeInfo.LinesPerFrame, "newimage.raw");		//Get Image to File
	break;

///////////////////////////Query Complete Proceded Single Frame Acquisition////////////////////////////
    case 3:		
	fprintf(LogFile, "Query Complete Proceded Single Frame Acquisition\n");
	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 1);		//Dummy Acquisition
	QueryWaitOnComplete(crntStatus, 0);
	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 0);		
	result = vip_sw_handshaking(VIP_SW_PREPARE, 0);
	result = vip_sw_handshaking(VIP_SW_PREPARE, 1);

	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 1);		//X-Rays Valid TRUE
	clock_valid = clock();

	ptDioWritePortByte.state = 1;							//Relay Close
	DRV_DioWritePortByte(DriverHandle,(LPT_DioWritePortByte)&ptDioWritePortByte);
	clock_relay = clock();

	result = QueryWaitOnComplete(crntStatus, 0);			//Wait For Image
	clock_complete = clock();

	result = GetImageToFile(modeInfo.ColsPerFrame, modeInfo.LinesPerFrame, "newimage.raw");		//Get Image to File
	break;

///////////////////////////500ms Sleep Proceded Double Frame Acquisition////////////////////////////
    case 4:				
	fprintf(LogFile, "500ms Sleep Proceded Double Frame Acquisition\n");
	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 1);		//First Acquisition
	clock_valid = clock();
	Sleep(500);

	ptDioWritePortByte.state = 1;							//Relay Close
	DRV_DioWritePortByte(DriverHandle,(LPT_DioWritePortByte)&ptDioWritePortByte);
	clock_relay = clock();

	QueryWaitOnComplete(crntStatus, 0);									//Wait For Image
	clock_complete = clock();

	result = GetImageToFile(modeInfo.ColsPerFrame, modeInfo.LinesPerFrame, "newimage.raw");		//Get 1st Image to File

	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 0);		
	result = vip_sw_handshaking(VIP_SW_PREPARE, 0);
	result = vip_sw_handshaking(VIP_SW_PREPARE, 1);

	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 1);		//Second Acquisition
	clock_valid2 = clock();

	result = QueryWaitOnComplete(crntStatus, 0);			//Wait For Image
	clock_complete2 = clock();

	result = GetImageToFile(modeInfo.ColsPerFrame, modeInfo.LinesPerFrame, "newimage2.raw");	//Get 2nd Image to File

    }

/////////////////////////////////////////////END MODE SPECIFIC CODE//////////////////////////////////

    ptDioWritePortByte.state = 0;
    DRV_DioWritePortByte(DriverHandle,(LPT_DioWritePortByte)&ptDioWritePortByte);
    result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 0);
    result = vip_sw_handshaking(VIP_SW_PREPARE, 0);

    fprintf(LogFile, "Xrays valid TRUE at time = %g\n" , clock_valid/(double)CLOCKS_PER_SEC );
    fprintf(LogFile, "Relays Closed at time = %g\n" , clock_relay/(double)CLOCKS_PER_SEC );
    fprintf(LogFile, "Image Complete at time = %g\n" , clock_complete/(double)CLOCKS_PER_SEC );
    fprintf(LogFile, "Time from Called to Relay Closed = %g\n" , (clock_relay/(double)CLOCKS_PER_SEC) - (clock_call/(double)CLOCKS_PER_SEC) );
    fprintf(LogFile, "Time from Relay Closed to Image Complete = %g\n" , (clock_complete/(double)CLOCKS_PER_SEC) - (clock_relay/(double)CLOCKS_PER_SEC) );

    if (AqState == 4)
    {
	fprintf(LogFile, "Xrays valid TRUE 2nd time at time = %g\n" , clock_valid2/(double)CLOCKS_PER_SEC );
	fprintf(LogFile, "2nd Image Complete at time = %g\n" , clock_complete2/(double)CLOCKS_PER_SEC );
	fprintf(LogFile, "Time from Relay Closed to 2nd Image Complete = %g\n" , (clock_complete2/(double)CLOCKS_PER_SEC) - (clock_relay/(double)CLOCKS_PER_SEC) );
	if (DataState == 2)
	{
	    FILE * StatsFile;
	    StatsFile = fopen("stats.csv", "a");
	    fprintf(StatsFile, "%g,%g,%g,%g,%g,%g\n" , clock_call/(double)CLOCKS_PER_SEC, clock_valid/(double)CLOCKS_PER_SEC, clock_relay/(double)CLOCKS_PER_SEC, clock_complete/(double)CLOCKS_PER_SEC,clock_valid2/(double)CLOCKS_PER_SEC,clock_complete2/(double)CLOCKS_PER_SEC);
	    fclose(StatsFile);
	}
    }
    else{
	if (DataState == 2)
	{
	    FILE * StatsFile;
	    StatsFile = fopen("stats.csv", "a");
	    fprintf(StatsFile, "%g,%g,%g,%g\n" , clock_call/(double)CLOCKS_PER_SEC, clock_valid/(double)CLOCKS_PER_SEC, clock_relay/(double)CLOCKS_PER_SEC, clock_complete/(double)CLOCKS_PER_SEC);
	    fclose(StatsFile);
	}
    }



    vip_enable_sw_handshaking(FALSE);
    fprintf(LogFile, "\n &&&&&&&&&&&&&&&&&&&&&&&&&&&& \n IMAGE TAKEN \n");
    fclose(LogFile);
    return result;
#endif
}

#if defined (commentout)

int 
main(int argc, char* argv[])
{
    char *path = default_path;
    int choice = 0;
    int result;
    int DataState = 0;
    int AqState = 0;

    SOpenReceptorLink  orl;
    SSysInfo  sysInfo;
    time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    printf ( "Current local time and date: %s", asctime (timeinfo) );

    memset(&orl, 0, sizeof(SOpenReceptorLink));
    printf("RadTest - Sample Code for Radiographic Image Acquisition\n\n");

    if (argc > 1)			// Check for receptor path on the command line
	path = argv[1];

    orl.StructSize = sizeof(SOpenReceptorLink);
    strcpy(orl.RecDirPath, path);

    // if we want to turn debug on so that it flushes to a file ..
    // or other settings see Virtual CP Communications Manual uncomment
    // and modify the following line if required
    //	orl.DebugMode = HCP_DBG_ON_FLSH;
    printf("Opening link to %s\n", orl.RecDirPath);
    result = vip_open_receptor_link(&orl);

    // The following call is for test purposes only
    result = DisableMissingCorrections(result);

    printf("Calling vip_check_link\n");
    result = CheckRecLink();
    printf("vip_check_link returns %d\n", result);
	
    //////////////////////LOG FILE STUFF
    FILE * LogFile;
    LogFile = fopen("log.txt", "a");
    fprintf(LogFile, "\n ______________________________________________ \n Rad Pannel Test Modified by Matthew Bieniosek 1/20/2010 \n BOOT on %s\n", asctime (timeinfo));

    FILE * StatsFile;
    StatsFile = fopen("stats.csv", "w");
    /////////////////////////////////

    ///////////////////SET OUTPUTS TO 0
    output = 0;
    ptDioWritePortByte.port  = 0;
    ptDioWritePortByte.mask  = 0xff;
    ptDioWritePortByte.state = output;
    DRV_DioWritePortByte(DriverHandle,(LPT_DioWritePortByte)&ptDioWritePortByte);
    /////////////////////////////////
	
    /////////////////Open IO CARD///////
    ErrCde = DRV_DeviceOpen(0,(LONG far *)&DriverHandle);
    ptDioReadPortByte.port = 0;
    ptDioReadPortByte.value = (USHORT far *)&input;
    ////////////////////////////////////

    if (result == HCP_NO_ERR)
    {
	GetSysInfo(sysInfo);

	result = vip_select_mode(crntModeSelect);

	if (result == HCP_NO_ERR)
	{
	    DataState = DisplayPromptData();		//Get Data State from user
	    if (DataState ==2)
	    {
		AqState = DisplayPromptMode(DataState);
		if(AqState !=4)
		{
		    switch (AqState)
		    {
		    case 1:
			fprintf(StatsFile, "Free Run Single Frame Acquisition, BOOT on %s\n", asctime (timeinfo));
			break;
		    case 2:
			fprintf(StatsFile, "Vip Reset Preceded Single Frame Acquisition, BOOT on %s\n", asctime (timeinfo));
			break;
		    case 3:
			fprintf(StatsFile, "Query Complete Proceded Single Frame Acquisition, BOOT on %s\n", asctime (timeinfo));
			break;
		    }
		    fprintf(StatsFile, "Time Called, Time X-Ray Valid, Time Relay Closed, Time Image Complete\n");
		}
		else
		{
		    fprintf(StatsFile, "500ms Sleep Proceded Double Frame Acquisition, BOOT on %s\n", asctime (timeinfo));
		    fprintf(StatsFile, "Time Called, Time First X-Ray Valid, Time Relay Closed, Time First Image Complete, Time Second X-Ray Valid, Time Second Image Complete\n");
		}

			
	    }
	    for (bool running = true; running;)
	    {
		if (AqState ==0 || DataState == 1)		//Get Acquisition State from user
		{
		    AqState = DisplayPromptMode(DataState);
		}
			
		if (DataState == 2)		//if taking stats do random sleep
		{
		    Sleep(rand()/10);
		}
		fclose(StatsFile);
		PerformSwRadAcquisition(DataState, AqState); //Call Image Aqcuisition

		Sleep(100);
		////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Code to take input from IO card!!!!!!!!!!!!!!!
		////char msg[256];
		////		
		///////////////////////////////////////////////////////////////////////////INPUT/////////


		////if ((ErrCde = DRV_DioReadPortByte(DriverHandle,
		////	    (LPT_DioReadPortByte)&ptDioReadPortByte)) != 0)
		////{
		////    //MessageBox(NULL, "error", "test",MB_OK);
		////}
		////else
		////{	
		////    time_t clock_input = clock();
		////    if(input != oldinput)
		////    {
		////	oldinput = input;
		////	fprintf(LogFile, "Input Changed to %d", input);
		////	fprintf(LogFile, " at time = %g\n", (double)(clock_input)/(double)CLOCKS_PER_SEC );
		////	printf("Input Changed to %d", input);
		////	printf( " at time = %g\n", (double)(clock_input)/(double)CLOCKS_PER_SEC );
		////	if(input == 11)
		////	{
		////	    //while(input !=8)
		////	    //{
		////	    //	time_t clock_input = clock();
		////	    //	ErrCde = DRV_DioReadPortByte(DriverHandle,(LPT_DioReadPortByte)&ptDioReadPortByte);
		////	    //	if(input != oldinput)
		////	    //	{
		////	    //		fprintf(LogFile, "Input Changed to %d", input);
		////	    //		fprintf(LogFile, " at time = %g\n", (double)(clock_input)/(double)CLOCKS_PER_SEC );
		////	    //		printf("Input Changed to %d", input);
		////	    //		printf(" at time = %g\n", (double)(clock_input)/(double)CLOCKS_PER_SEC );
		////	    //		oldinput = input;
		////	    //	}
		////	    ////	printf("\nlooping....\n");
		////	    //	
		////	    //}
		////	    time_t clock_call = clock();
		////	    fprintf(LogFile, "Calling Acquisition Function at time = %g\n" , (double)(clock_call)/(double)CLOCKS_PER_SEC );
		////	    printf("Calling Acquisition Function at time = %g\n" , (double)(clock_call)/(double)CLOCKS_PER_SEC );
		////	    fclose(LogFile);

		////	    ///////////////////Set output to 2?
		////	    //output = 2;
		////	    //ptDioWritePortByte.state = output;

		////	    //time_t clock_write2 = clock();
		////	    //DRV_DioWritePortByte(DriverHandle,(LPT_DioWritePortByte)&ptDioWritePortByte);
		////	    //fprintf(LogFile, "Output to x-ray generator set to %d", output);
		////	    //fprintf(LogFile, " at time = %g\n" , (double)(clock_write2) /(double)CLOCKS_PER_SEC);
		////	    /////////////////////
		////	    PerformSwRadAcquisition(1);
		////	    LogFile = fopen("log.txt", "a");
		////	}
		////	}
		////	}
		////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!End Code to take input from IO card!!!!!!!!!!!!!!!
	    }
	}
	else
	    printf("vip_select_mode(%d) returns error %d\n", crntModeSelect, result);

	vip_close_link();
    }
    else
	printf("vip_open_receptor_link returns error %d\n", result);

    printf("\n**Hit any key to exit");
    _getch();
    while(!_kbhit()) Sleep (100);

    return 0;
}
#endif

