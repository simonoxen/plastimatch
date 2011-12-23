/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#define _CRT_SECURE_NO_DEPRECATE

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <dos.h>
#include <math.h>

#include "HcpErrors.h"
#include "HcpFuncDefs.h"
#include "iostatus.h"

// The string such as "A422-07" is the imager serial number
char *default_path = "C:\\IMAGERs\\A422-07"; // Path to IMAGER tables
//char *default_path = "C:\\IMAGERs\\A663-11"; // Path to IMAGER tables

// In the following the mode number for the rad mode required should be set
int  crntModeSelect = 0;


#define ESC_KEY   (0x1B)
#define ENTER_KEY (0x0D)

#define HCP_SIGNAL_TIMEOUT        (-2)
#define HCP_SIGNAL_KEY_PRESSED    (-1)


int  CheckRecLink()
{
    SCheckLink clk;
    memset(&clk, 0, sizeof(SCheckLink));
    clk.StructSize = sizeof(SCheckLink);
    int result = vip_check_link(&clk);
//#if 0
    while(result != HCP_NO_ERR)
    {
	printf ("Retry (%d)\n", result);
	Sleep(1000);
	result = vip_check_link(&clk);
    }
//#endif
    return result;
}

//----------------------------------------------------------------------
//
//  DisableMissingCorrections
//  -------------------------
//  This call allows the code to be used in test and development
//  environments in which not all correction files are available
//  for the receptor. It disables all image corrections on any of
//  the non-fatal codes HCP_OFST_ERR, HCP_GAIN_ERR< HCP_DFCT_ERR.
//
//----------------------------------------------------------------------

int DisableMissingCorrections(int result)
{
    SCorrections corr;
    memset(&corr, 0, sizeof(SCorrections));
    corr.StructSize = sizeof(SCorrections);

    // If caller has no error, try to fetch the correction error
    if (result == HCP_NO_ERR)
	result = vip_get_correction_settings(&corr);

    switch (result)
    {
    case HCP_OFST_ERR:
	printf("Requested corrections not available: offset file missing\n");
	break;
    case HCP_GAIN_ERR:
	printf("Requested corrections not available: gain file missing\n");
	break;
    case HCP_DFCT_ERR:
	printf("Requested corrections not available: defect file missing\n");
	break;
    default:
	return result;
    }

    memset(&corr, 0, sizeof(SCorrections));
    corr.StructSize = sizeof(SCorrections);

    // this means not all correction files are available
    // here we will just turn corrections off but IN REAL APPLICATION
    // WE MUST BE SURE CORRECTIONS ARE ON AND THE RECEPTOR IS CALIBRATED
    result = vip_set_correction_settings(&corr);
    if(result == VIP_NO_ERR) printf("\n\nCORRECTIONS ARE OFF!!");

    return result;
}

//----------------------------------------------------------------------
//
//  GetModeInfo
//
//----------------------------------------------------------------------

int GetModeInfo(SModeInfo &modeInfo)
{
    int result = HCP_NO_ERR;

    memset(&modeInfo, 0, sizeof(modeInfo));
    modeInfo.StructSize = sizeof(SModeInfo);
    printf("Calling vip_get_mode_details\n");

    result = vip_get_mode_info(crntModeSelect, &modeInfo);

    if (result == HCP_NO_ERR)
    {
	printf("  >> ModeDescription=\"%s\"\n", modeInfo.ModeDescription);
	printf("  >> AcqType=             %5d\n", modeInfo.AcqType);
	printf("  >> FrameRate=          %6.3f, AnalogGain=         %6.3f\n",
	    modeInfo.FrameRate, modeInfo.AnalogGain);
	printf("  >> LinesPerFrame=       %5d, ColsPerFrame=        %5d\n",
	    modeInfo.LinesPerFrame, modeInfo.ColsPerFrame);
	printf("  >> LinesPerPixel=       %5d, ColsPerPixel=        %5d\n",
	    modeInfo.LinesPerPixel, modeInfo.ColsPerPixel);
    }
    else
	printf("**** vip_get_mode_info returns error %d\n", result);
    return result;
}


//----------------------------------------------------------------------
//
//  GetSysInfo
//
//----------------------------------------------------------------------

int GetSysInfo(SSysInfo &sysInfo)
{
    int result = HCP_NO_ERR;

    printf("Calling vip_get_sys_info\n");

    memset(&sysInfo, 0, sizeof(sysInfo));
    sysInfo.StructSize = sizeof(SSysInfo);

    result = vip_get_sys_info(&sysInfo);

    if (result == HCP_NO_ERR)
    {
	printf("  >> SysDescription=\"%s\"\n", sysInfo.SysDescription);
	printf("  >> NumModes=         %5d,   DfltModeNum=   %5d\n",
	    sysInfo.NumModes, sysInfo.DfltModeNum);
	printf("  >> MxLinesPerFrame=  %5d,   MxColsPerFrame=%5d\n",
	    sysInfo.MxLinesPerFrame, sysInfo.MxColsPerFrame);
	printf("  >> MxPixelValue=     %5d,   HasVideo=      %5d\n",
	    sysInfo.MxPixelValue, sysInfo.HasVideo);
	printf("  >> StartUpConfig=    %5d,   NumAsics=      %5d\n",
	    sysInfo.StartUpConfig, sysInfo.NumAsics);
	printf("  >> ReceptorType=     %5d\n", sysInfo.ReceptorType);
    }
    else
	printf("**** vip_get_sys_info returns error %d\n", result);

    return result;
}


//----------------------------------------------------------------------
//
//  QueryProgress
//
//----------------------------------------------------------------------

int QueryProgress(UQueryProgInfo &crntStatus, bool showAll = false)
{
    UQueryProgInfo prevStatus = crntStatus;
    memset(&crntStatus, 0, sizeof(SQueryProgInfo));
    crntStatus.qpi.StructSize = sizeof(SQueryProgInfo);
    int result = vip_query_prog_info(HCP_U_QPI, &crntStatus);

    if (result == HCP_NO_ERR)
    {
	if (showAll
	    || (prevStatus.qpi.NumFrames != crntStatus.qpi.NumFrames)
	    || (prevStatus.qpi.Complete != crntStatus.qpi.Complete)
	    || (prevStatus.qpi.NumPulses != crntStatus.qpi.NumPulses)
	    || (prevStatus.qpi.ReadyForPulse != crntStatus.qpi.ReadyForPulse))
	{
	    printf("vip_query_prog_info: frames=%d complete=%d pulses=%d ready=%d\n",
		crntStatus.qpi.NumFrames,
		crntStatus.qpi.Complete,
		crntStatus.qpi.NumPulses,
		crntStatus.qpi.ReadyForPulse);
	}
	if (_kbhit())
	{
	    int ch = _getch();
	    if (ch == ESC_KEY)
	    {
		printf("<esc> key - cancelling operation\n");
		return HCP_SIGNAL_KEY_PRESSED;
	    }
	}
    }
    else
	printf("**** vip_query_prog_info returns error %d\n", result);

    return result;
}

//----------------------------------------------------------------------
//
//  QueryWaitOnComplete
//
//----------------------------------------------------------------------

int QueryWaitOnComplete(UQueryProgInfo &crntStatus, int timeoutMsec=0)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    crntStatus.qpi.Complete = FALSE;
    printf("Waiting for Complete == TRUE...\n");
    while (result == HCP_NO_ERR)
    {
	result = QueryProgress(crntStatus);
	if(crntStatus.qpi.Complete == TRUE) break;
	if (timeoutMsec > 0)
	{
	    totalMsec += 100;
	    if (totalMsec >= timeoutMsec)
	    {
		printf("*** TIMEOUT ***\n");
		return HCP_SIGNAL_TIMEOUT;
	    }
	}
	Sleep(100);
    }
    return result;
}

//----------------------------------------------------------------------
//
//  QueryWaitOnNumFrames
//
//----------------------------------------------------------------------

int QueryWaitOnNumFrames(UQueryProgInfo &crntStatus, int numRequested, int timeoutMsec=0)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    crntStatus.qpi.Complete = FALSE;
    printf("Waiting for Complete == TRUE...\n");
    while (result == HCP_NO_ERR)
    {
	result = QueryProgress(crntStatus);
	if(crntStatus.qpi.NumFrames >= numRequested)
	    break;
	if (timeoutMsec > 0)
	{
	    totalMsec += 100;
	    if (totalMsec >= timeoutMsec)
	    {
		printf("*** TIMEOUT ***\n");
		return HCP_SIGNAL_TIMEOUT;
	    }
	}
	Sleep(100);
    }
    return result;
}

//----------------------------------------------------------------------
//
//  QueryWaitOnNumPulsesChange
//
//----------------------------------------------------------------------

int QueryWaitOnNumPulsesChange(UQueryProgInfo &crntStatus, int timeoutMsec=0)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    int numPulses = crntStatus.qpi.NumPulses;
    printf("Waiting for Complete == TRUE...\n");
    while (result == HCP_NO_ERR)
    {
	result = QueryProgress(crntStatus);
	if(crntStatus.qpi.NumPulses != numPulses)
	    break;
	if (timeoutMsec > 0)
	{
	    totalMsec += 100;
	    if (totalMsec >= timeoutMsec)
	    {
		printf("*** TIMEOUT ***\n");
		return HCP_SIGNAL_TIMEOUT;
	    }
	}
	Sleep(100);
    }
    return result;
}

//----------------------------------------------------------------------
//
//  QueryWaitOnReadyForPulse
//
//----------------------------------------------------------------------

int QueryWaitOnReadyForPulse(UQueryProgInfo &crntStatus, int timeoutMsec=0, int expectedState=TRUE)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    crntStatus.qpi.ReadyForPulse = FALSE;
    if (expectedState)
	printf("Waiting for ReadyForPulse == TRUE...\n");
    else
	printf("Waiting for ReadyForPulse == FALSE...\n");
    while (result == HCP_NO_ERR)
    {
	result = QueryProgress(crntStatus);
	if(crntStatus.qpi.ReadyForPulse == expectedState) break;
	if (timeoutMsec > 0)
	{
	    totalMsec += 100;
	    if (totalMsec >= timeoutMsec)
	    {
		printf("*** TIMEOUT ***\n");
		return HCP_SIGNAL_TIMEOUT;
	    }
	}
	Sleep(100);
    }
    return result;
}

//----------------------------------------------------------------------
//
//  ShowDllVersions
//
//----------------------------------------------------------------------

void ShowDllVersions()
{
    static char version[512];
    static char dllName[512];

    printf("calling vip_get_dlls_versions\n");
    int result = vip_get_dll_version(version, dllName, 512);
    if (result == HCP_NO_ERR)
    {
	char *v = version;
	char *n = dllName;
	int vLen = strlen(v);
	int nLen = strlen(n);
	printf("--------------------------------------------------------\n");
	while ((vLen > 0) && (nLen > 0))
	{
	    printf("%-24s %s\n", n, v);
	    v += (vLen + 1);
	    n += (nLen + 1);
	    vLen = strlen(v);
	    nLen = strlen(n);
	}
	printf("--------------------------------------------------------\n");
    }
}

//----------------------------------------------------------------------
//
//  ShowDiagData
//  ------------
//  L.04 only - special call to read out the complete diagnostic data,
//  from the most recent image acquistion in a "raw" form (temperatures
//  and voltages have not been scaled to physical units).
//  Results valid only AFTER an image has been acquired.
//
//----------------------------------------------------------------------

void ShowDiagData()
{
    UQueryProgInfo  uqpi;

    memset(&uqpi.qpidiag, 0, sizeof(SDiagData));
    uqpi.qpidiag.StructSize = sizeof(SDiagData);

    int result = vip_query_prog_info(HCP_U_QPIDIAGDATA, &uqpi);
    if (result == HCP_NO_ERR)
    {
	printf("  Receptor PanelType=%d, FwVersion=0x%.3X BoardId=%.4X %.4X %.4X\n",
	    uqpi.qpidiag.PanelType,
	    uqpi.qpidiag.FwVersion,
	    uqpi.qpidiag.BoardSNbr[2],
	    uqpi.qpidiag.BoardSNbr[1],
	    uqpi.qpidiag.BoardSNbr[0]);
	printf("  RcptFrameId=%d ExposureStatus=0x%.4X\n",
	    uqpi.qpidiag.RcptFrameId, uqpi.qpidiag.Exposed);
    }
    else
	printf("Diag data returns %d\n", result);
}

//----------------------------------------------------------------------
//
//  ShowFrameData
//  -------------
//  L.04 only - special call to read out the frame identification data
//  from the most recent image acquistion. Results valid only AFTER
//  an image has been acquired.
//
//----------------------------------------------------------------------

void ShowFrameData(int crntReq=0)
{
    UQueryProgInfo  uqpi;
    int  uType = HCP_U_QPIFRAME;
    if (crntReq)
	uType |= HCP_U_QPI_CRNT_DIAG_DATA;

    memset(&uqpi.qpitemps, 0, sizeof(SQueryProgInfoFrame));
    uqpi.qpitemps.StructSize = sizeof(SQueryProgInfoFrame);

    int result = vip_query_prog_info(uType, &uqpi);
    if (result == HCP_NO_ERR)
    {
	printf("RcptFrameId=%d ExposureStatus=0x%.4X\n",
	    uqpi.qpiframe.RcptFrameId, uqpi.qpiframe.Exposed);
    }
}

//----------------------------------------------------------------------
//
//  ShowReceptorData
//  ----------------
//  L.04 only - special call to read out receptor identification data
//  from the most recent image acquistion. Results valid only AFTER
//  an image has been acquired.
//
//----------------------------------------------------------------------

void ShowReceptorData()
{
    UQueryProgInfo  uqpi;
    int  uType = HCP_U_QPIRCPT;

    memset(&uqpi.qpircpt, 0, sizeof(SQueryProgInfoRcpt));
    uqpi.qpircpt.StructSize = 28; // sizeof(SQueryProgInfoRcpt);

    printf("Calling vip_query_prog_info(HCP_U_QPIRCPT, %d)\n", sizeof(SQueryProgInfoRcpt));
    int result = vip_query_prog_info(uType, &uqpi);
    if (result == HCP_NO_ERR)
    {
	printf("Receptor PanelType=%d, FwVersion=0x%.3X BoardId=%.2X%.2X%.2X\n",
	    uqpi.qpircpt.PanelType,
	    uqpi.qpircpt.FwVersion,
	    uqpi.qpircpt.BoardSNbr[1],
	    uqpi.qpircpt.BoardSNbr[1],
	    uqpi.qpircpt.BoardSNbr[0]);
    }
    else
	printf("returns %d\n", result);
}

//----------------------------------------------------------------------
//
//  ShowTemperatureData
//  -------------------
//  L.04 only - special call to read out receptor temperature sensor
//  data from the most recent image acquistion.
//  Results valid only AFTER an image has been acquired.
//
//----------------------------------------------------------------------

void ShowTemperatureData(int crntReq=0)
{
    UQueryProgInfo  uqpi;
    int  uType = HCP_U_QPITEMPS;
    if (crntReq)
	uType |= HCP_U_QPI_CRNT_DIAG_DATA;

    memset(&uqpi.qpitemps, 0, sizeof(SQueryProgInfoTemps));
    uqpi.qpitemps.StructSize = sizeof(SQueryProgInfoTemps);

    int result = vip_query_prog_info(uType, &uqpi);
    if (result == HCP_NO_ERR)
    {
	for (int i = 0; i < uqpi.qpitemps.NumSensors; i++)
	    printf("T[%d]=%5.2f\n", i, uqpi.qpitemps.Celsius[i]);
    }
}

//----------------------------------------------------------------------
//
//  ShowVoltageData
//  ---------------
//  L.04 only - special call to read out receptor voltage sensor
//  data from the most recent image acquistion.
//  Results valid only AFTER an image has been acquired.
//
//----------------------------------------------------------------------

void ShowVoltageData(int crntReq=0)
{
    UQueryProgInfo  uqpi;
    int  uType = HCP_U_QPIVOLTS;
    if (crntReq)
	uType |= HCP_U_QPI_CRNT_DIAG_DATA;

    memset(&uqpi.qpitemps, 0, sizeof(SQueryProgInfoVolts));
    uqpi.qpitemps.StructSize = sizeof(SQueryProgInfoVolts);

    int result = vip_query_prog_info(uType, &uqpi);
    if (result == HCP_NO_ERR)
    {
	for (int i = 0; i < uqpi.qpitemps.NumSensors; i++)
	    printf("V[%2d]=%f\n", i, uqpi.qpivolts.Volts[i]);
    }
}


//----------------------------------------------------------------------
//
//  ShowImageStatistics
//
//----------------------------------------------------------------------

void ShowImageStatistics(int npixels, USHORT *image_ptr)
{
    int nTotal;
    long minPixel, maxPixel;
    int i;
    double pixel, sumPixel;

    nTotal = 0;
    minPixel = 4095;
    maxPixel = 0;

    sumPixel = 0.0;

    for (i = 0; i < npixels; i++)
    {
	pixel = (double) image_ptr[i];
	sumPixel += pixel;
	if (image_ptr[i] > maxPixel)
	    maxPixel = image_ptr[i];
	if (image_ptr[i] < minPixel)
	    minPixel = image_ptr[i];
	nTotal++;
    }

    printf("Image: %d pixels, average=%9.2f min=%d max=%d\n",
	nTotal, sumPixel / nTotal, minPixel, maxPixel);
}


//----------------------------------------------------------------------
//
//  GetImageToFile
//
//----------------------------------------------------------------------

int GetImageToFile(int xSize, int ySize, char *filename, int imageType=VIP_CURRENT_IMAGE)
{
    int result;
    int mode_num = crntModeSelect;

    int npixels = xSize * ySize;

    USHORT *image_ptr = (USHORT *)malloc(npixels * sizeof(USHORT));

    result = vip_get_image(mode_num, imageType, xSize, ySize, image_ptr);

    if(result == HCP_NO_ERR)
    {
	ShowImageStatistics(npixels, image_ptr);

	// file on the host computer for storing the image
	FILE *finput = fopen(filename, "wb");
	if (finput == NULL)
	{
	    printf("Error opening image file to put file.");
	    exit(-1);
	}

	fwrite(image_ptr, sizeof(USHORT), npixels, finput);
	fclose(finput);

    }
    else
    {
	printf("*** vip_get_image returned error %d\n", result);
    }

    free(image_ptr);
    return HCP_NO_ERR;
}


//----------------------------------------------------------------------
//
//  PerformGainCalibration
//
//----------------------------------------------------------------------

int  PerformGainCalibration()
{
    UQueryProgInfo crntStatus;
    int  numGainCalImages = 4;

    printf("Perform gain calibration for radiographic mode\n");
    int  result = vip_reset_state();

// NOTE: for simplicity some error checking left out in the following
    int numOfstCal=2;
    vip_get_num_cal_frames(crntModeSelect, &numOfstCal);

    result = vip_gain_cal_prepare(crntModeSelect, false);
    if (result != HCP_NO_ERR)
    {
	printf("*** vip_gain_cal_prepare returns error %d\n", result);
	return result;
    }

    result = vip_sw_handshaking(VIP_SW_PREPARE, 1);
    if (result != HCP_NO_ERR)
    {
	printf("*** vip_sw_handshaking(VIP_SW_PREPARE, 1) returns error %d\n", result);
	return result;
    }

    printf("INITIAL DARK-FIELD IMAGES - DO NOT EXPOSE TO X-RAYS\n");
    QueryWaitOnNumFrames(crntStatus, numOfstCal, 15000);
    printf("Initial offset calibration complete\n\n");

    int numPulses=0;
    vip_enable_sw_handshaking(TRUE);

    //	printf("\nPress any key to begin flat field acquisition\n");
    //	while(!_kbhit()) Sleep (100);

    for (numPulses = 0; numPulses < numGainCalImages; )
    {
	printf("FLAT FIELD EXPOSURE %d\n", numPulses);
	QueryWaitOnReadyForPulse(crntStatus, 5000);
	vip_sw_handshaking(VIP_SW_VALID_XRAYS, TRUE);
	Sleep(300);
	printf("READY FOR X-RAYS - EXPOSE NOW =============>\n");
	result = QueryWaitOnNumPulsesChange(crntStatus, 0);
	if (result != HCP_NO_ERR)
	{
	    vip_reset_state();
	    return result;
	}
	vip_sw_handshaking(VIP_SW_VALID_XRAYS, FALSE);
	numPulses = crntStatus.qpi.NumPulses;
    }

    printf("Setting PREPARE=0\n");
    result = vip_sw_handshaking(VIP_SW_PREPARE, 0);
    if (QueryWaitOnComplete(crntStatus, 10000) != HCP_NO_ERR)
    {
	vip_reset_state();
	return result;
    }

    printf("Gain calibration complete\n");

    // no need to do this if not using Hw handshaking at all
    result = vip_enable_sw_handshaking(FALSE);

    return result;
}


//----------------------------------------------------------------------
//
//  PerformRadAcquisition
//  ---------------------
//  This is the recommended method for radiographic acquisition.
//  The "Hardware Handshaking" calls are used to initiate the
//  acquisition.
//
//----------------------------------------------------------------------

int  PerformRadAcquisition()
{
    int  result;
    UQueryProgInfo crntStatus;
    SModeInfo  modeInfo;

    GetModeInfo(modeInfo);

    printf ("Calling vip_enable_sw_handshaking(FALSE)\n");
    result = vip_enable_sw_handshaking(FALSE);
    if (result != HCP_NO_ERR)
    {
	printf("*** returns error %d\n", result);
	return result;
    }

    printf("Calling vip_io_enable(HS_ACTIVE)\n");
    result = vip_io_enable(HS_ACTIVE);
    if (result == HCP_NO_ERR)
    {
	result = QueryWaitOnReadyForPulse(crntStatus, 5000);
	if (result == HCP_NO_ERR)
	{
	    printf("READY FOR X-RAYS - EXPOSE AT ANY TIME\n");
	    result = QueryWaitOnNumPulsesChange(crntStatus, 0);
	    if (result == HCP_NO_ERR)
	    {
		QueryWaitOnNumFrames(crntStatus, 1, 0);
		result = GetImageToFile(modeInfo.ColsPerFrame,
		    modeInfo.LinesPerFrame,
		    "preview.raw", VIP_PREVIEW_IMAGE);
		result = QueryWaitOnComplete(crntStatus, 0);
	    }
	    if (result == HCP_NO_ERR)
	    {
		result = GetImageToFile(modeInfo.ColsPerFrame,
		    modeInfo.LinesPerFrame,
		    "newimage.raw");
		ShowReceptorData();
		ShowFrameData();
		ShowTemperatureData();
		ShowVoltageData();
	    }
	    else if (result == HCP_SIGNAL_KEY_PRESSED)
	    {
		printf("Calling vip_io_enable(HS_STANDBY)\n");
		vip_io_enable(HS_STANDBY);
		result = QueryWaitOnReadyForPulse(crntStatus, 5000, FALSE);
	    }
	    else
		printf("*** Acquisition terminated with error %d\n", result);
	    vip_io_enable(HS_STANDBY);
	}
    }
    else
    {
	printf("*** returns error %d - acquisition not enabled\n", result);
    }

    printf("Acquisition complete\n");
    return result;
}

//----------------------------------------------------------------------
//
//  PerformSwRadAcquisition
//  -----------------------
//  This is a variant of the radiographic acquisition in which the
//  "Software Handshaking" calls are used. For the new rad panels,
//  starting with L.04 build 26, this method does not enable the logic
//  for a synchronized X-ray exposure, so that a dark image is captured.
//
//----------------------------------------------------------------------

int  PerformSwRadAcquisition()
{
    int result = HCP_NO_ERR;
    UQueryProgInfo crntStatus;
    SModeInfo  modeInfo;

    result = GetModeInfo(modeInfo);
    if (result != HCP_NO_ERR)
    {
	printf("*** unable to get mode info\n");
	return result;
    }

    printf("Enabling software handshaking...\n");
    result = vip_enable_sw_handshaking(TRUE);
    if (result != HCP_NO_ERR)
    {
	printf("*** vip_enable_sw_handshaking returned error %d\n", result);
	return result;
    }
    result = vip_sw_handshaking(VIP_SW_PREPARE, 1);
    if (result == HCP_NO_ERR)
    {
	result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 1);
	if (result == HCP_NO_ERR)
	{
	    result = QueryWaitOnComplete(crntStatus, 0);
	    result = vip_sw_handshaking(VIP_SW_VALID_XRAYS, 0);
	}
	result = vip_sw_handshaking(VIP_SW_PREPARE, 0);
    }

    if (result == HCP_NO_ERR)
	result = GetImageToFile(modeInfo.ColsPerFrame, modeInfo.LinesPerFrame, "newimage.raw");

    // no need to do this if not using Hw handshaking at all
    vip_enable_sw_handshaking(FALSE);
    return result;
}

//----------------------------------------------------------------------
//
//  DisplayPrompt
//
//----------------------------------------------------------------------

void DisplayPrompt()
{
    printf("\n------------------------------\n");
    printf("Select operation:\n");
    printf("1 - Acquire radiographic image\n");
    printf("2 - Acquire dark image using SwHandshaking calls\n");
    printf("3 - Perform gain calibration\n");
    printf("0 - Exit\n");
}

//----------------------------------------------------------------------
//  main
//----------------------------------------------------------------------
int 
main(int argc, char* argv[])
{
    char *path = default_path;
    int choice = 0;
    int result;
    SOpenReceptorLink orl;
    SSysInfo sysInfo;

    memset (&orl, 0, sizeof(SOpenReceptorLink));
    printf ("Welcome to acquire_4030e\n");

    // Check for receptor path on the command line
    if (argc > 1) {
	path = argv[1];
    }

    orl.StructSize = sizeof(SOpenReceptorLink);
    strcpy(orl.RecDirPath, path);

    // if we want to turn debug on so that it flushes to a file ..
    // or other settings see Virtual CP Communications Manual uncomment
    // and modify the following line if required
    //	orl.DebugMode = HCP_DBG_ON_FLSH;
    printf("Opening link to %s\n", orl.RecDirPath);
    result = vip_open_receptor_link (&orl);
    printf ("Result = %04x\n", result);

    // The following call is for test purposes only
    result = DisableMissingCorrections(result);

    printf("Calling vip_check_link\n");
    result = CheckRecLink();
    printf("vip_check_link returns %d\n", result);

    if (result == HCP_NO_ERR)
    {
	GetSysInfo(sysInfo);

	result = vip_select_mode (crntModeSelect);

	if (result == HCP_NO_ERR)
	{
	    DisplayPrompt();
	    for (bool running = true; running;)
	    {
		if (_kbhit())
		{
		    int keyCode = _getch();
		    printf("%c\n", keyCode);
		    switch (keyCode)
		    {
		    case '1':
			PerformRadAcquisition();
			break;
		    case '2':
			PerformSwRadAcquisition();
			break;
		    case '3':
			PerformGainCalibration();
			break;
		    case '0':
			running = false;
			break;
		    }
		    if (running)
			DisplayPrompt();
		}
	    }
	}
	else {
	    printf ("vip_select_mode(%d) returns error %d\n", 
		crntModeSelect, result);
	}

	vip_close_link();
    }
    else
	printf("vip_open_receptor_link returns error %d\n", result);

    //printf("\n**Hit any key to exit");
    //_getch();
    //while(!_kbhit()) Sleep (100);

    return 0;
}
