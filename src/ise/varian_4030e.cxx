/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "ise_config.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <QMutex>
#include <QMutexLocker>
#include "HcpErrors.h"
#include "HcpFuncDefs.h"
#include "iostatus.h"

#include "aqprintf.h"
#include "dips_panel.h"
#include "varian_4030e.h"

/* This mutex is a class static variable */
QMutex Varian_4030e::vip_mutex;

#define HCP_SIGNAL_TIMEOUT        (-2)

Varian_4030e::Varian_4030e (int idx)
{
    this->idx = idx;
    current_mode = 0;
}

Varian_4030e::~Varian_4030e (void)
{
}

const char*
Varian_4030e::error_string (int error_code)
{
    if (error_code <= 128) {
        return HcpErrStrList[error_code];
    } else {
        return "";
    }
}

int
Varian_4030e::check_link()
{
    SCheckLink clk;
    memset (&clk, 0, sizeof(SCheckLink));
    clk.StructSize = sizeof(SCheckLink);
    vip_mutex.lock ();
    vip_select_receptor (this->receptor_no);
    int result = vip_check_link (&clk);
    while (result != HCP_NO_ERR) {
        vip_mutex.unlock ();
        Sleep (1000);
        aqprintf ("Retry vip_check_link.\n");
        vip_mutex.lock ();
        vip_select_receptor (this->receptor_no);
        result = vip_check_link (&clk);
    }
    vip_mutex.unlock ();
    return result;
}

int 
Varian_4030e::open_link (const char *path)
{
    int result;
    SOpenReceptorLink orl;
    memset (&orl, 0, sizeof(SOpenReceptorLink));
    orl.StructSize = sizeof(SOpenReceptorLink);
    strcpy (orl.RecDirPath, path);

    // if we want to turn debug on so that it flushes to a file ..
    // or other settings see Virtual CP Communications Manual uncomment
    // and modify the following line if required
    //	orl.DebugMode = HCP_DBG_ON_FLSH;
    aqprintf("Opening link to %s\n", orl.RecDirPath);
    result = vip_open_receptor_link (&orl);

    this->receptor_no = orl.RcptNum;
    aqprintf ("Receptor number %d\n", this->receptor_no);
    return result;
}

void
Varian_4030e::close_link ()
{
    vip_close_link (this->receptor_no);
}

//----------------------------------------------------------------------
//  DisableMissingCorrections
//  -------------------------
//  This call allows the code to be used in test and development
//  environments in which not all correction files are available
//  for the receptor. It disables all image corrections on any of
//  the non-fatal codes HCP_OFST_ERR, HCP_GAIN_ERR< HCP_DFCT_ERR.
//----------------------------------------------------------------------
int 
Varian_4030e::disable_missing_corrections (int result)
{
    SCorrections corr;
    memset(&corr, 0, sizeof(SCorrections));
    corr.StructSize = sizeof(SCorrections);

    QMutexLocker mutex_locker (&vip_mutex);
    vip_select_receptor (this->receptor_no);

    /* If caller has no error, try to fetch the correction error */
    if (result == HCP_NO_ERR) {
        result = vip_get_correction_settings(&corr);
    }

#if defined (commentout)
    switch (result)
    {
    case HCP_OFST_ERR:
        aqprintf ("Requested corrections not available: offset file missing\n");
        break;
    case HCP_GAIN_ERR:
        aqprintf ("Requested corrections not available: gain file missing\n");
        break;
    case HCP_DFCT_ERR:
        aqprintf ("Requested corrections not available: defect file missing\n");
        break;
    default:
        return result;
    }
#endif

    // this means not all correction files are available
    // here we will just turn corrections off but IN REAL APPLICATION
    // WE MUST BE SURE CORRECTIONS ARE ON AND THE RECEPTOR IS CALIBRATED
    memset(&corr, 0, sizeof(SCorrections));
    corr.StructSize = sizeof(SCorrections);
    result = vip_set_correction_settings(&corr);
    if (result == VIP_NO_ERR) {
        aqprintf("Corrections are off\n");
    }

    return result;
}

int 
Varian_4030e::get_mode_info (SModeInfo &modeInfo, int current_mode)
{
    int result = HCP_NO_ERR;

    memset(&modeInfo, 0, sizeof(modeInfo));
    modeInfo.StructSize = sizeof(SModeInfo);

    QMutexLocker mutex_locker (&vip_mutex);
    //vip_select_receptor (this->receptor_no);
    result = vip_get_mode_info (current_mode, &modeInfo);
    return result;
}

int 
Varian_4030e::print_mode_info ()
{
    SModeInfo modeInfo;
    int result = get_mode_info (modeInfo, this->current_mode);

    if (result == HCP_NO_ERR) {
        aqprintf (">> ModeDescription=\"%s\"\n", 
            modeInfo.ModeDescription);
        aqprintf (">> AcqType=             %5d\n", 
            modeInfo.AcqType);
        aqprintf (">> FrameRate=          %6.3f,"
            " AnalogGain=         %6.3f\n",
            modeInfo.FrameRate, modeInfo.AnalogGain);
        aqprintf (">> LinesPerFrame=       %5d,"
            " ColsPerFrame=        %5d\n",
            modeInfo.LinesPerFrame, modeInfo.ColsPerFrame);
        aqprintf (">> LinesPerPixel=       %5d,"
            " ColsPerPixel=        %5d\n",
            modeInfo.LinesPerPixel, modeInfo.ColsPerPixel);
    } else {
        aqprintf ("**** vip_get_mode_info returns error %d\n", 
            result);
    }
    return result;
}

void 
Varian_4030e::print_sys_info (void)
{
    SSysInfo sysInfo;
    int result = HCP_NO_ERR;

    memset(&sysInfo, 0, sizeof(sysInfo));
    sysInfo.StructSize = sizeof(SSysInfo);
    vip_mutex.lock ();
    vip_select_receptor (this->receptor_no);
    result = vip_get_sys_info (&sysInfo);
    vip_mutex.unlock ();
    if (result == HCP_NO_ERR) {
        aqprintf("> SysDescription=\"%s\"\n", 
            sysInfo.SysDescription);
        aqprintf("> NumModes=         %5d,   DfltModeNum=   %5d\n", 
            sysInfo.NumModes, sysInfo.DfltModeNum);
        aqprintf("> MxLinesPerFrame=  %5d,   MxColsPerFrame=%5d\n", 
            sysInfo.MxLinesPerFrame, sysInfo.MxColsPerFrame);
        aqprintf("> MxPixelValue=     %5d,   HasVideo=      %5d\n",
            sysInfo.MxPixelValue, sysInfo.HasVideo);
        aqprintf("> StartUpConfig=    %5d,   NumAsics=      %5d\n",
            sysInfo.StartUpConfig, sysInfo.NumAsics);
        aqprintf("> ReceptorType=     %5d\n", 
            sysInfo.ReceptorType);
    } else {
        aqprintf("**** vip_get_sys_info returns error %d\n", result);
    }
}

int 
Varian_4030e::query_prog_info (UQueryProgInfo &crntStatus, bool show_all)
{
    UQueryProgInfo prevStatus = crntStatus;
    memset(&crntStatus, 0, sizeof(SQueryProgInfo));
    crntStatus.qpi.StructSize = sizeof(SQueryProgInfo);

//    QMutexLocker mutex_locker (&vip_mutex);
//    vip_select_receptor (this->receptor_no);
    int result = vip_query_prog_info (HCP_U_QPI, &crntStatus);
    if (result != HCP_NO_ERR) {
        aqprintf ("**** vip_query_prog_info returns error %d (%s)\n", result,
	    Varian_4030e::error_string (result));
        return result;
    }

    if (show_all
        || (prevStatus.qpi.NumFrames != crntStatus.qpi.NumFrames)
        || (prevStatus.qpi.Complete != crntStatus.qpi.Complete)
        || (prevStatus.qpi.NumPulses != crntStatus.qpi.NumPulses)
        || (prevStatus.qpi.ReadyForPulse != crntStatus.qpi.ReadyForPulse))
    {
        aqprintf("frames=%d complete=%d pulses=%d ready=%d\n",
            crntStatus.qpi.NumFrames,
            crntStatus.qpi.Complete,
            crntStatus.qpi.NumPulses,
            crntStatus.qpi.ReadyForPulse);
    }
    return result;
}

int 
Varian_4030e::wait_on_complete (UQueryProgInfo &crntStatus, int timeoutMsec)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    crntStatus.qpi.Complete = FALSE;
    aqprintf("Waiting for Complete == TRUE...\n");
    while (result == HCP_NO_ERR)
    {
        result = query_prog_info (crntStatus);
        if(crntStatus.qpi.Complete == TRUE) break;
        if (timeoutMsec > 0)
        {
            totalMsec += 100;
            if (totalMsec >= timeoutMsec)
            {
                aqprintf("*** TIMEOUT ***\n");
                return HCP_SIGNAL_TIMEOUT;
            }
        }
        Sleep(100);
    }
    return result;
}

int 
Varian_4030e::wait_on_num_frames (
    UQueryProgInfo &crntStatus, int numRequested, int timeoutMsec)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    crntStatus.qpi.Complete = FALSE;
    aqprintf("Waiting for Complete == TRUE...\n");
    while (result == HCP_NO_ERR)
    {
        result = query_prog_info (crntStatus);
        if(crntStatus.qpi.NumFrames >= numRequested)
            break;
        if (timeoutMsec > 0)
        {
            totalMsec += 100;
            if (totalMsec >= timeoutMsec)
            {
                aqprintf("*** TIMEOUT ***\n");
                return HCP_SIGNAL_TIMEOUT;
            }
        }
        Sleep(100);
    }
    return result;
}

int 
Varian_4030e::wait_on_num_pulses (UQueryProgInfo &crntStatus, int timeoutMsec)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    int numPulses = crntStatus.qpi.NumPulses;
    aqprintf("Waiting for Complete == TRUE...\n");
    while (result == HCP_NO_ERR)
    {
        result = query_prog_info (crntStatus);
        if(crntStatus.qpi.NumPulses != numPulses)
            break;
        if (timeoutMsec > 0)
        {
            totalMsec += 100;
            if (totalMsec >= timeoutMsec)
            {
                aqprintf("*** TIMEOUT ***\n");
                return HCP_SIGNAL_TIMEOUT;
            }
        }
        Sleep(100);
    }
    return result;
}

/* busy / wait loop until panel is ready for pulse */
int 
Varian_4030e::wait_on_ready_for_pulse (
    UQueryProgInfo &crntStatus, 
    int timeoutMsec,
    int expectedState
)
{
    int result = HCP_NO_ERR;
    int totalMsec = 0;

    crntStatus.qpi.ReadyForPulse = FALSE;
    if (expectedState) {
        aqprintf ("Waiting for ReadyForPulse == TRUE...\n");
    } else {
        aqprintf ("Waiting for ReadyForPulse == FALSE...\n");
    }

    bool first = true;
    while (result == HCP_NO_ERR) {
        result = query_prog_info (crntStatus, first);
        first = false;
        if (crntStatus.qpi.ReadyForPulse == expectedState) {
            break;
        }
        if (timeoutMsec > 0) {
            totalMsec += 100;
            if (totalMsec >= timeoutMsec) {
                aqprintf("*** TIMEOUT ***\n");
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

    aqprintf("calling vip_get_dlls_versions\n");
    int result = vip_get_dll_version(version, dllName, 512);
    if (result == HCP_NO_ERR)
    {
        char *v = version;
        char *n = dllName;
        int vLen = strlen(v);
        int nLen = strlen(n);
        aqprintf("--------------------------------------------------------\n");
        while ((vLen > 0) && (nLen > 0))
        {
            aqprintf("%-24s %s\n", n, v);
            v += (vLen + 1);
            n += (nLen + 1);
            vLen = strlen(v);
            nLen = strlen(n);
        }
        aqprintf("--------------------------------------------------------\n");
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
        aqprintf("  Receptor PanelType=%d, FwVersion=0x%.3X BoardId=%.4X %.4X %.4X\n",
            uqpi.qpidiag.PanelType,
            uqpi.qpidiag.FwVersion,
            uqpi.qpidiag.BoardSNbr[2],
            uqpi.qpidiag.BoardSNbr[1],
            uqpi.qpidiag.BoardSNbr[0]);
        aqprintf("  RcptFrameId=%d ExposureStatus=0x%.4X\n",
            uqpi.qpidiag.RcptFrameId, uqpi.qpidiag.Exposed);
    }
    else
        aqprintf("Diag data returns %d\n", result);
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
        aqprintf("RcptFrameId=%d ExposureStatus=0x%.4X\n",
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

    aqprintf("Calling vip_query_prog_info(HCP_U_QPIRCPT, %d)\n", 
        sizeof(SQueryProgInfoRcpt));
    int result = vip_query_prog_info(uType, &uqpi);
    if (result == HCP_NO_ERR) {
        aqprintf(
            "Receptor PanelType=%d, FwVersion=0x%.3X "
            "BoardId=%.2X%.2X%.2X\n",
            uqpi.qpircpt.PanelType,
            uqpi.qpircpt.FwVersion,
            uqpi.qpircpt.BoardSNbr[1],
            uqpi.qpircpt.BoardSNbr[1],
            uqpi.qpircpt.BoardSNbr[0]);
    } else {
        aqprintf ("*** vip_query_prog_info returns %d (%s)\n", result,
            Varian_4030e::error_string (result));
    }
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
            aqprintf("T[%d]=%5.2f\n", i, uqpi.qpitemps.Celsius[i]);
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
            aqprintf("V[%2d]=%f\n", i, uqpi.qpivolts.Volts[i]);
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

    aqprintf("Image: %d pixels, average=%9.2f min=%d max=%d\n",
        nTotal, sumPixel / nTotal, minPixel, maxPixel);
}


//----------------------------------------------------------------------
//
//  GetImageToFile
//
//----------------------------------------------------------------------

int 
Varian_4030e::get_image_to_file (int xSize, int ySize, 
    char *filename, int imageType)
{
    int result;
    int mode_num = this->current_mode;

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
            aqprintf("Error opening image file to put file.");
            exit(-1);
        }

        fwrite(image_ptr, sizeof(USHORT), npixels, finput);
        fclose(finput);

    }
    else
    {
        aqprintf("*** vip_get_image returned error %d\n", result);
    }

    free(image_ptr);
    return HCP_NO_ERR;
}


int 
Varian_4030e::get_image_to_dips (Dips_panel *dp, int xSize, int ySize)
{
    int result;
    int mode_num = this->current_mode;

    int npixels = xSize * ySize;

    USHORT *image_ptr = (USHORT *)malloc(npixels * sizeof(USHORT));

    result = vip_get_image(mode_num, VIP_CURRENT_IMAGE, 
        xSize, ySize, image_ptr);

    if (result == HCP_NO_ERR) {
        ShowImageStatistics(npixels, image_ptr);
    } else {
        aqprintf("*** vip_get_image returned error %d\n", result);
        return HCP_NO_ERR;
    }

    dp->wait_for_dips ();

    for (int i = 0; i < xSize * ySize; i++) {
        dp->pixelp[i] = image_ptr[i];
    }

    dp->send_image ();

    free(image_ptr);
    return HCP_NO_ERR;
}

int 
Varian_4030e::rad_acquisition (Dips_panel *dp)
{
    int  result;
    UQueryProgInfo crntStatus;
    SModeInfo  modeInfo;

    this->get_mode_info (modeInfo, this->current_mode);

    // aqprintf ("Calling vip_enable_sw_handshaking(FALSE)\n");
    result = vip_enable_sw_handshaking (FALSE);
    if (result != HCP_NO_ERR) {
        aqprintf ("**** vip_enable_sw_handshaking returns error %d\n", 
            result);
        return result;
    }

    //aqprintf("Calling vip_io_enable(HS_ACTIVE)\n");
    vip_mutex.lock ();
    vip_select_receptor (this->receptor_no);
    result = vip_io_enable (HS_ACTIVE);
    vip_mutex.unlock ();
    if (result != HCP_NO_ERR) {
        aqprintf("**** returns error %d - acquisition not enabled\n", result);
        return result;
    }

    result = wait_on_ready_for_pulse (crntStatus, 5000, TRUE);
    if (result == HCP_NO_ERR) {

        aqprintf("READY FOR X-RAYS - EXPOSE AT ANY TIME\n");
        /* Close relay to generator */
        /* Poll generator pins 8/26, look for de-assert */
        /* Close, then open pins 3/4 to paxscan */

        result = this->wait_on_num_pulses (crntStatus, 0);
        if (result == HCP_NO_ERR) {
            result = this->wait_on_num_frames (crntStatus, 1, 0);
            if (result != HCP_NO_ERR) {
                aqprintf ("***** Didn't find expected NUM FRAMES?\n");
            }
            result = this->get_image_to_dips (
                dp, modeInfo.ColsPerFrame,
                modeInfo.LinesPerFrame);
            result = this->wait_on_complete (crntStatus, 0);
        }

        if (result == HCP_NO_ERR) {
#if defined (commentout)
            result = this->get_image_to_file (modeInfo.ColsPerFrame,
                modeInfo.LinesPerFrame,
                "newimage.raw");
#endif
            ShowReceptorData();
            ShowFrameData();
            ShowTemperatureData();
            ShowVoltageData();
        }
        else {
            aqprintf("*** Acquisition terminated with error %d\n", result);
        }
        vip_io_enable(HS_STANDBY);
    }
    /* If FgClockRate is low, such as 3.5, we sometimes get 
       spurious frame/complete prog_status.  This is cleared by 
       setting to HS_STANDBY, then setting back to HS_ACTIVE */
    vip_io_enable(HS_STANDBY);
    return result;
}
