/* -----------------------------------------------------------------------
See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
----------------------------------------------------------------------- */
#include "ise_config.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "HcpErrors.h"
#include "HcpFuncDefs.h"
#include "iostatus.h"

#include "aqprintf.h"
#include "dips_panel.h"
#include "varian_4030e.h"
#include "YK16GrayImage.h"
#include <fstream>
#include <QMessageBox>
#include "Acquire_4030e_DlgControl.h"
#include <QSystemSemaphore>

#define HCP_SIGNAL_TIMEOUT        (-2)

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
	minPixel = 65535;
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


void Varian_4030e::CalcImageInfo (double& meanVal, double& STDV, double& minVal, double& maxVal,
    int sizeX, int sizeY, USHORT* pImg) //all of the images taken before will be inspected
{
    int nTotal;
    long minPixel, maxPixel;
    int i;
    double pixel, sumPixel;

    int npixels = sizeX*sizeY;
    nTotal = 0;
    minPixel = 65535;
    maxPixel = 0;
    sumPixel = 0.0;

    for (i = 0; i < npixels; i++)
    {
        pixel = (double) pImg[i];
        sumPixel += pixel;
        if (pImg[i] > maxPixel)
            maxPixel = pImg[i];
        if (pImg[i] < minPixel)
            minPixel = pImg[i];
        nTotal++;
    }

    double meanPixelval = sumPixel / (double)nTotal;

    double sqrSum = 0.0;
    for (i = 0; i < npixels; i++)
    {
        sqrSum = sqrSum + pow(((double)pImg[i] - meanPixelval),2.0);
    }
    double SD = sqrt(sqrSum/(double)nTotal);

    meanVal = meanPixelval;
    STDV = SD;
    minVal = minPixel;
    maxVal = maxPixel;
    return;
}

bool Varian_4030e::SameImageExist(IMGINFO& curInfo, int& sameImgIndex)
{
    bool result = false;

    std::vector<IMGINFO>::iterator it;

    IMGINFO compInfo;
    //SD and AVG
    int order = 0;
    for (it = m_vImageInfo.begin() ; it != m_vImageInfo.end() ; it++)
    {
        compInfo = (*it);
        if (fabs(curInfo.meanVal - compInfo.meanVal) < 0.0001 &&
            fabs(curInfo.SD - compInfo.SD) < 0.0001
        )
        {
            result = true;
            sameImgIndex = order;
            return result;
        }
        order++;
    }
    return result;
}

int
Varian_4030e::get_image_to_buf (int xSize, int ySize)
{
    bool bDefectMapApply = m_pParent->m_dlgControl->ChkBadPixelCorrApply->isChecked();
    bool bDarkCorrApply = m_pParent->m_dlgControl->ChkDarkFieldApply->isChecked();
    bool bGainCorrApply = m_pParent->m_dlgControl->ChkGainCorrectionApply->isChecked();
    bool bRawSaveForDebug = m_pParent->m_dlgControl->ChkRawSave->isChecked();
    bool bDarkSaveForDebug = m_pParent->m_dlgControl->ChkDarkCorrectedSave->isChecked();

    int result;
    int mode_num = this->current_mode;
    int npixels = xSize * ySize;

    USHORT *image_ptr = (USHORT *) malloc (npixels * sizeof(USHORT));

    result = vip_get_image (mode_num, VIP_CURRENT_IMAGE,
        xSize, ySize, image_ptr);

    bool bImageDuplicationOccurred = false;

    double tmpMean = 0.0;
    double tmpSD = 0.0;
    double tmpMin = 0.0;
    double tmpMax = 0.0;

    CalcImageInfo (tmpMean, tmpSD, tmpMin, tmpMax, xSize, ySize, image_ptr);

    IMGINFO tmpInfo;
    tmpInfo.meanVal = tmpMean;
    tmpInfo.SD = tmpSD;
    tmpInfo.minVal = tmpMin;
    tmpInfo.maxVal = tmpMax;

    aqprintf ("IMG_INSPECTION(Mean|SD|MIN|MAX): %3.2f | %3.2f | %3.1f | %3.1f\n", tmpInfo.meanVal, tmpInfo.SD, tmpInfo.minVal, tmpInfo.maxVal);

    int sameImageIndex = -1;
    if (SameImageExist(tmpInfo, sameImageIndex))
    {
        aqprintf("******SAME_IMAGE_ERROR!! prevImgNum [%d]\n", sameImageIndex);
        bImageDuplicationOccurred = true;
    }
    m_vImageInfo.push_back(tmpInfo);

    if (result != HCP_NO_ERR)
    {
        aqprintf("*** vip_get_image returned error %d\n", result);
    }

    if (bDarkCorrApply)
        aqprintf("Dark Correction On\n");
    if (bGainCorrApply)
        aqprintf("Gain Correction On\n");
    if (bDefectMapApply)
        aqprintf("Bad Pixel Correction On\n");

    // Begin correcting images
    USHORT *pImageCorr = (USHORT *)malloc (npixels * sizeof(USHORT));

    /* For debugging purposes, we sometimes want to save the original 
       (uncorrected) images and also the images with only dark field
       corrections. */
    if (bRawSaveForDebug)
    {
        for (int i = 0; i < xSize * ySize; i++) {
            m_pParent->m_pCurrImageRaw->m_pData[i] = image_ptr[i]; //raw image
        }
    }

    if (bDarkSaveForDebug)
    {
        if (m_pParent->m_pDarkImage->IsEmpty())
        {
            aqprintf("DEBUG for dark-corrected image. No dark ref image. raw image will be sent.\n");
            for (int i = 0; i < xSize * ySize; i++){
                m_pParent->m_pCurrImageDarkCorrected->m_pData[i] = image_ptr[i];
            }
        }
        else
        {
            for (int i = 0; i < xSize * ySize; i++) {
                if (image_ptr[i] > m_pParent->m_pDarkImage->m_pData[i])
                    m_pParent->m_pCurrImageDarkCorrected->m_pData[i] = image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i];
                else
                    m_pParent->m_pCurrImageDarkCorrected->m_pData[i] = 0;
            }
        }

        if (bDefectMapApply)
        {
            if (!m_pParent->m_vBadPixelMap.empty())
                m_pParent->m_pCurrImageDarkCorrected->DoPixelReplacement(m_pParent->m_vBadPixelMap);
        }
    }

    if (!bDarkCorrApply && !bGainCorrApply)
    {
        for (int i = 0; i < xSize * ySize; i++) {
            pImageCorr[i] = image_ptr[i];
        }
    }
    else if (bDarkCorrApply && !bGainCorrApply)
    {
        if (m_pParent->m_pDarkImage->IsEmpty())
        {
            aqprintf("No dark ref image. raw image will be sent.\n");
            for (int i = 0; i < xSize * ySize; i++) {
                pImageCorr[i] = image_ptr[i];
            }
        }
        else
        {
            for (int i = 0; i < xSize * ySize; i++) {
                if (image_ptr[i] > m_pParent->m_pDarkImage->m_pData[i])
                    pImageCorr[i] = image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i];
                else
                    pImageCorr[i] = 0;
            }
        }
    }
    else if (!bDarkCorrApply && bGainCorrApply)
    {
        if (m_pParent->m_pGainImage->IsEmpty())
        {
            aqprintf("No gain ref image. raw image will be sent.\n");
            for (int i = 0; i < xSize * ySize; i++) {
                pImageCorr[i] = image_ptr[i]; //raw image
            }
        }
        else
        {
            //get a mean value for m_pGainImage
            double sum = 0.0;
            double MeanVal = 0.0;
            for (int i = 0; i < xSize * ySize; i++) {
                sum = sum + m_pParent->m_pGainImage->m_pData[i];
            }
            MeanVal = sum / (double) (xSize*ySize);

            for (int i = 0; i < xSize * ySize; i++) {
                if (m_pParent->m_pGainImage->m_pData[i] == 0)
                    pImageCorr[i] = image_ptr[i];
                else
                    pImageCorr[i] = (USHORT)((double)image_ptr[i]/(double)(m_pParent->m_pGainImage->m_pData[i])*MeanVal);
            }
        }

        double CalibF = 0.0;
        for (int i = 0; i < xSize * ySize; i++) {
            CalibF = m_pParent->m_dlgControl->lineEditSingleCalibFactor->text().toDouble();
            pImageCorr[i] = (unsigned short)(pImageCorr[i] * CalibF);
        }
    }

    else if (bDarkCorrApply && bGainCorrApply)
    {
        bool bRawImage = false;
        if (m_pParent->m_pDarkImage->IsEmpty())
        {
            aqprintf("No dark ref image. raw image will be sent.\n");
            bRawImage = true;
        }
        if (m_pParent->m_pGainImage->IsEmpty())
        {
            aqprintf("No gain ref image. raw image will be sent.\n");
            bRawImage = true;
        }

        if (bRawImage)
        {
            for (int i = 0; i < xSize * ySize; i++) {
                pImageCorr[i] = image_ptr[i]; //raw image
            }
        }
        else //if not raw image
        {
            //get a mean value for m_pGainImage
            double sum = 0.0;
            double MeanVal = 0.0;
            for (int i = 0; i < xSize * ySize; i++) {
                sum = sum + (m_pParent->m_pGainImage->m_pData[i] - m_pParent->m_pDarkImage->m_pData[i]);
            }
            MeanVal = sum/(double)(xSize*ySize);

            double denom = 0.0;
            int iDenomLessZero = 0;
            int iDenomLessZero_RawIsGreaterThanDark = 0;
            int iDenomLessZero_RawIsSmallerThanDark = 0;
            int iDenomOK_RawValueMinus = 0;
            int iValOutOfRange = 0;

            for (int i = 0; i < xSize * ySize; i++)
            {
                denom = (double) (m_pParent->m_pGainImage->m_pData[i] - m_pParent->m_pDarkImage->m_pData[i]);

                if (denom <= 0)
                {
                    iDenomLessZero++;

                    if (image_ptr[i] > m_pParent->m_pDarkImage->m_pData[i])
                    {
                        pImageCorr[i] = image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i];
                        iDenomLessZero_RawIsGreaterThanDark++;
                    }
                    else
                    {
                        pImageCorr[i] = 0;
                        iDenomLessZero_RawIsSmallerThanDark++;
                    }
                }
                else
                {
                    double tmpVal = 0.0;
                    tmpVal = (image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i]) / denom * MeanVal;

                    if (tmpVal < 0)
                    {
                        pImageCorr[i] = 0;
                        iDenomOK_RawValueMinus++;
                    }
                    else
                    {
                        if (tmpVal > 65535) //16bit max value
                            iValOutOfRange++;

                        pImageCorr[i] = (USHORT)tmpVal;
                    }
                }
            }//end of for
        }//end if not bRawImage

        //Apply Single Calib. Factor
        double CalibF = 0.0;

        for (int i = 0; i < xSize * ySize; i++) {
            CalibF = m_pParent->m_dlgControl->lineEditSingleCalibFactor->text().toDouble();
            pImageCorr[i] = (unsigned short)(pImageCorr[i] * CalibF);
        }

        //aqprintf("CalibF = %3.2f\n", CalibF);
    } // else if (m_bDarkCorrApply && m_bGainCorrApply)

    //Customized Thresholding after Gain Corr

    unsigned short customThreVal = m_pParent->m_dlgControl->lineEditForcedThresholdVal->text().toDouble();  //13000
    bool enableCustomThre = m_pParent->m_dlgControl->ChkForcedThresholding->isChecked();

    //aqprintf("ThreVal = %d\n", customThreVal);

    if (enableCustomThre)
    {
        for (int i = 0; i < xSize * ySize; i++)
        {
            if (pImageCorr[i] >= customThreVal)
                pImageCorr[i] = customThreVal;
        }
    }

    if (bDefectMapApply && !m_pParent->m_vBadPixelMap.empty()) //pixel replacement
    {
        aqprintf("Bad pixel correction is under progress. Bad pixel numbers = %d.\n",m_pParent->m_vBadPixelMap.size());
        std::vector<BADPIXELMAP>::iterator it;
        int oriIdx, replIdx;

        for (it = m_pParent->m_vBadPixelMap.begin() ; it != m_pParent->m_vBadPixelMap.end() ; it++)
        {
            BADPIXELMAP tmpData= (*it);
            oriIdx = tmpData.BadPixY * xSize + tmpData.BadPixX;
            replIdx = tmpData.ReplPixY * xSize + tmpData.ReplPixX;
            pImageCorr[oriIdx] = pImageCorr[replIdx];
        }
    }


    /////////////////////***************Manual Correction END***********///////////////////

    if (result == HCP_NO_ERR)
    {
        int size = xSize*ySize;

        if (m_pParent->m_pCurrImage->IsEmpty() || m_pParent->m_pCurrImage->m_iWidth != xSize || m_pParent->m_pCurrImage->m_iHeight != ySize)
            m_pParent->m_pCurrImage->CreateImage(xSize, ySize, 0);

        for (int i = 0  ; i<size ; i++)
            m_pParent->m_pCurrImage->m_pData[i] = pImageCorr[i];
    }
    else
    {
        aqprintf("*** vip_get_image returned error %d\n", result);
    }

    free(image_ptr);
    free(pImageCorr);

    if (bImageDuplicationOccurred)
        return HCP_SAME_IMAGE_ERROR;

    return HCP_NO_ERR;
}

bool Varian_4030e::CopyFromBufAndSendToDips (Dips_panel *dp) //get cur image to curImage
{
    if (m_pParent->m_pCurrImage == NULL)
    {
        aqprintf("Cur image not created\n");
        return false;
    }

    if (m_pParent->m_pCurrImage->IsEmpty())
    {
        aqprintf("No buffer image to send\n");
        return false;
    }
    if (dp->width != m_pParent->m_pCurrImage->m_iWidth || dp->height != m_pParent->m_pCurrImage->m_iHeight)
    {
        aqprintf("Image size is not matched!\n");
        return false;
    }

    dp->wait_for_dips ();

    int npixels = m_pParent->m_pCurrImage->m_iWidth * m_pParent->m_pCurrImage->m_iHeight;
    for (int i = 0  ; i<npixels ; i++)
        dp->pixelp[i] = m_pParent->m_pCurrImage->m_pData[i];

    dp->send_image (); //Sending IMage to dips message sent

    return true;
}



Varian_4030e::Varian_4030e (int idx)
{
	this->idx = idx;
	current_mode = 0;
}


Varian_4030e::Varian_4030e (int idx, Acquire_4030e_child* pParent)
{
	this->idx = idx;
	current_mode = 0;

	m_pParent = pParent;

	int m_iSizeX = 0;
	int m_iSizeY = 0;
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
Varian_4030e::check_link(int MaxRetryCnt)
{
	SCheckLink clk;
	memset (&clk, 0, sizeof(SCheckLink));
	clk.StructSize = sizeof(SCheckLink);

	vip_select_receptor (this->receptor_no);
	int result = vip_check_link (&clk);

	int tmpCnt = 0;

	while (result != HCP_NO_ERR) {
		aqprintf ("Retry vip_check_link.\n"); //frequently shown at second panel initialization
		vip_select_receptor (this->receptor_no);
		result = vip_check_link (&clk);

		tmpCnt++;
		if (tmpCnt > MaxRetryCnt)
		{
			return RESTART_NEEDED;
			break;
		}
	}

	return result;
}

int
Varian_4030e::open_link (int panelIdx, const char *path)
{
	int result;
	SOpenReceptorLink orl;
	memset (&orl, 0, sizeof(SOpenReceptorLink));
	orl.StructSize = sizeof(SOpenReceptorLink);
	strcpy (orl.RecDirPath, path);

	orl.RcptNum = panelIdx;
	this->receptor_no = orl.RcptNum;

	// if we want to turn debug on so that it flushes to a file ..
	// or other settings see Virtual CP Communications Manual uncomment
	// and modify the following line if required

	//orl.DebugMode = HCP_DBG_ON_DLG;
	orl.DebugMode = HCP_DBG_ON_FLSH;

	//HCP_DBG_OFF 0 // no debug
	//HCP_DBG_ON 1 // debug on – output //written to file when debug is turned off
	//HCP_DBG_ON_FLSH 2// debug on – output //written to file //continuously
	//HCP_DBG_ON_DLG 3// debug on – output //written to file when //debug is turned off and //output to a dialog //window

	aqprintf("Opening link to %s\n", orl.RecDirPath);
	result = vip_open_receptor_link (&orl); //What if panel is unplugged? //what are this func returning?
	this->receptor_no = orl.RcptNum;

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
		//PrintCurrentTime();
	}

	return result;
}

int
Varian_4030e::get_mode_info (SModeInfo &modeInfo, int current_mode)
{
	int result = HCP_NO_ERR;

	memset(&modeInfo, 0, sizeof(modeInfo));
	modeInfo.StructSize = sizeof(SModeInfo);

	result = vip_get_mode_info (current_mode, &modeInfo);

	this->m_iSizeX = modeInfo.ColsPerFrame;
	this->m_iSizeY = modeInfo.LinesPerFrame;

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

	vip_select_receptor (this->receptor_no);
	result = vip_get_sys_info (&sysInfo);

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

int Varian_4030e::query_prog_info (UQueryProgInfo &crntStatus, bool show_all)
{
	UQueryProgInfo prevStatus = crntStatus;

	memset(&crntStatus, 0, sizeof(SQueryProgInfo));
	crntStatus.qpi.StructSize = sizeof(SQueryProgInfo);

	int result = vip_query_prog_info (HCP_U_QPI, &crntStatus);

	if (result != HCP_NO_ERR) { // After acquisition,no data error is normal.
		aqprintf ("**** vip_query_prog_info returns error %d (%s)\n", result,
			Varian_4030e::error_string (result));
		return result;
	}

	m_pParent->m_dlgControl->EditFrames->setText(QString("%1").arg(crntStatus.qpi.NumFrames));
	m_pParent->m_dlgControl->EditComplete->setText(QString("%1").arg(crntStatus.qpi.Complete));
	m_pParent->m_dlgControl->EditPulses->setText(QString("%1").arg(crntStatus.qpi.NumPulses));
	m_pParent->m_dlgControl->EditReady->setText(QString("%1").arg(crntStatus.qpi.ReadyForPulse));


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

int Varian_4030e::DBG_get_image_to_buf (unsigned short* pImgData, int xSize, int ySize) //get cur image to curImage
{
    bool bDefectMapApply = m_pParent->m_dlgControl->ChkBadPixelCorrApply->isChecked();
    bool bDarkCorrApply = m_pParent->m_dlgControl->ChkDarkFieldApply->isChecked();
    bool bGainCorrApply = m_pParent->m_dlgControl->ChkGainCorrectionApply->isChecked();
    bool bRawSaveForDebug = m_pParent->m_dlgControl->ChkRawSave->isChecked();
    bool bDarkSaveForDebug = m_pParent->m_dlgControl->ChkDarkCorrectedSave->isChecked();

    int result;
    int mode_num = this->current_mode;
    int npixels = xSize * ySize;

    USHORT *image_ptr = (USHORT *)malloc(npixels * sizeof(USHORT));

    int size = xSize*ySize;
    for (int i = 0 ; i<size ; i++)
    {
        image_ptr[i] = pImgData[i];
    }

    bool bImageDuplicationOccurred = false;

    double tmpMean = 0.0;
    double tmpSD = 0.0;
    double tmpMin = 0.0;
    double tmpMax = 0.0;

    CalcImageInfo (tmpMean, tmpSD, tmpMin, tmpMax, xSize, ySize, image_ptr);

    IMGINFO tmpInfo;
    tmpInfo.meanVal = tmpMean;
    tmpInfo.SD = tmpSD;
    tmpInfo.minVal = tmpMin;
    tmpInfo.maxVal = tmpMax;

    aqprintf("IMG_INSPECTION(Mean|SD|MIN|MAX): %3.2f | %3.2f | %3.1f | %3.1f\n", tmpInfo.meanVal, tmpInfo.SD, tmpInfo.minVal, tmpInfo.maxVal);

    int sameImageIndex = -1;
    /*if (SameImageExist(tmpInfo, sameImageIndex))
      {
      aqprintf("******SAME_IMAGE_ERROR!! prevImgNum [%d]\n", sameImageIndex);
      bImageDuplicationOccurred = true;
      }*/
    //m_vImageInfo.push_back(tmpInfo);

    /*if (result != HCP_NO_ERR)
      {
      aqprintf("*** vip_get_image returned error %d\n", result);
      }*/


    if (bDarkCorrApply)
        aqprintf("Dark Correction On\n");
    if (bGainCorrApply)
        aqprintf("Gain Correction On\n");
    if (bDefectMapApply)
        aqprintf("Bad Pixel Correction On\n");

    /////////////////////***************Manul Correction START***********///////////////////
    USHORT *pImageCorr = (USHORT *)malloc(npixels * sizeof(USHORT));

    /* DEBUGGING CODE. When save the real image, Raw images and Dark-only corrected images should be saved for debugging purpose */

    if (bRawSaveForDebug)
    {
        for (int i = 0; i < xSize * ySize; i++) {
            m_pParent->m_pCurrImageRaw->m_pData[i] = image_ptr[i]; //raw image
        }
    }

    if (bDarkSaveForDebug)
    {
        if (m_pParent->m_pDarkImage->IsEmpty())
        {
            aqprintf("DEBUG for dark-corrected image. No dark ref image. raw image will be sent.\n");
            for (int i = 0; i < xSize * ySize; i++){
                m_pParent->m_pCurrImageDarkCorrected->m_pData[i] = image_ptr[i]; //raw image
            }
        }
        else
        {
            for (int i = 0; i < xSize * ySize; i++) {
                if (image_ptr[i] > m_pParent->m_pDarkImage->m_pData[i])
                    m_pParent->m_pCurrImageDarkCorrected->m_pData[i] = image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i];
                else
                    m_pParent->m_pCurrImageDarkCorrected->m_pData[i] = 0;
            }
        }

        if (bDefectMapApply)
        {
            if (!m_pParent->m_vBadPixelMap.empty())
                m_pParent->m_pCurrImageDarkCorrected->DoPixelReplacement(m_pParent->m_vBadPixelMap);
        }
    }
    /* DEBUGGING CODE. When save the real image, Raw images and Dark-only corrected images should be saved for debugging purpose */ //END

    if (!bDarkCorrApply && !bGainCorrApply)
    {
        for (int i = 0; i < xSize * ySize; i++) {
            pImageCorr[i] = image_ptr[i]; //raw image
        }
    }
    else if (bDarkCorrApply && !bGainCorrApply)
    {
        if (m_pParent->m_pDarkImage->IsEmpty())
        {
            aqprintf("No dark ref image. raw image will be sent.\n");
            for (int i = 0; i < xSize * ySize; i++) {
                pImageCorr[i] = image_ptr[i]; //raw image
            }
        }
        else
        {
            for (int i = 0; i < xSize * ySize; i++) {
                if (image_ptr[i] > m_pParent->m_pDarkImage->m_pData[i])
                    pImageCorr[i] = image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i];
                else
                    pImageCorr[i] = 0;
            }
        }
    }
    else if (!bDarkCorrApply && bGainCorrApply)
    {
        if (m_pParent->m_pGainImage->IsEmpty())
        {
            aqprintf("No gain ref image. raw image will be sent.\n");
            for (int i = 0; i < xSize * ySize; i++) {
                pImageCorr[i] = image_ptr[i]; //raw image
            }
        }
        else
        {
            //get a mean value for m_pGainImage
            double sum = 0.0;
            double MeanVal = 0.0;
            for (int i = 0; i < xSize * ySize; i++) {
                sum = sum + m_pParent->m_pGainImage->m_pData[i];
            }
            MeanVal = sum/(double)(xSize*ySize);

            for (int i = 0; i < xSize * ySize; i++) {
                if (m_pParent->m_pGainImage->m_pData[i] == 0)
                    pImageCorr[i] = image_ptr[i];
                else
                    pImageCorr[i] = (USHORT)((double)image_ptr[i]/(double)(m_pParent->m_pGainImage->m_pData[i])*MeanVal);
            }
        }


        double CalibF = 0.0;

        for (int i = 0; i < xSize * ySize; i++) {
            CalibF = m_pParent->m_dlgControl->lineEditSingleCalibFactor->text().toDouble();
            pImageCorr[i] = (unsigned short)(pImageCorr[i] * CalibF);
        }

    }

    else if (bDarkCorrApply && bGainCorrApply)
    {
        //aqprintf("Dark and gain correction\n");

        bool bRawImage = false;
        if (m_pParent->m_pDarkImage->IsEmpty())
        {
            aqprintf("No dark ref image. raw image will be sent.\n");
            bRawImage = true;
        }
        if (m_pParent->m_pGainImage->IsEmpty())
        {
            aqprintf("No gain ref image. raw image will be sent.\n");
            bRawImage = true;
        }

        if (bRawImage)
        {
            for (int i = 0; i < xSize * ySize; i++) {
                pImageCorr[i] = image_ptr[i]; //raw image
            }
        }
        else //if not raw image
        {
            //get a mean value for m_pGainImage
            double sum = 0.0;
            double MeanVal = 0.0;
            for (int i = 0; i < xSize * ySize; i++) {
                sum = sum + (m_pParent->m_pGainImage->m_pData[i] - m_pParent->m_pDarkImage->m_pData[i]);
            }
            MeanVal = sum/(double)(xSize*ySize);

            double denom = 0.0;
            int iDenomLessZero = 0;
            int iDenomLessZero_RawIsGreaterThanDark = 0;
            int iDenomLessZero_RawIsSmallerThanDark = 0;
            int iDenomOK_RawValueMinus = 0;
            int iValOutOfRange = 0;


            for (int i = 0; i < xSize * ySize; i++)
            {
                denom = (double)(m_pParent->m_pGainImage->m_pData[i] - m_pParent->m_pDarkImage->m_pData[i]);

                if (denom <= 0)
                {
                    iDenomLessZero++;

                    if (image_ptr[i] > m_pParent->m_pDarkImage->m_pData[i])
                    {
                        pImageCorr[i] = image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i];
                        iDenomLessZero_RawIsGreaterThanDark++;
                    }
                    else
                    {
                        pImageCorr[i] = 0;
                        iDenomLessZero_RawIsSmallerThanDark++;
                    }
                }
                else
                {
                    double tmpVal = 0.0;
                    tmpVal = (image_ptr[i] - m_pParent->m_pDarkImage->m_pData[i]) / denom * MeanVal;

                    if (tmpVal < 0)
                    {
                        pImageCorr[i] = 0;
                        iDenomOK_RawValueMinus++;
                    }
                    else
                    {
                        if (tmpVal > 65535) //16bit max value
                            iValOutOfRange++;

                        pImageCorr[i] = (USHORT)tmpVal;
                    }
                }
            }//end of for
        }//end if not bRawImage

        //Apply Single Calib. Factor
        double CalibF = 0.0;

        for (int i = 0; i < xSize * ySize; i++) {
            CalibF = m_pParent->m_dlgControl->lineEditSingleCalibFactor->text().toDouble();
            pImageCorr[i] = (unsigned short)(pImageCorr[i] * CalibF);
        }

        //aqprintf("CalibF = %3.2f\n", CalibF);
    } // else if (m_bDarkCorrApply && m_bGainCorrApply)

    //Customized Thresholding after Gain Corr

    unsigned short customThreVal = m_pParent->m_dlgControl->lineEditForcedThresholdVal->text().toDouble();  //13000
    bool enableCustomThre = m_pParent->m_dlgControl->ChkForcedThresholding->isChecked();

    //aqprintf("ThreVal = %d\n", customThreVal);

    if (enableCustomThre)
    {
        for (int i = 0; i < xSize * ySize; i++)
        {
            if (pImageCorr[i] >= customThreVal)
                pImageCorr[i] = customThreVal;
        }
    }

    if (bDefectMapApply && !m_pParent->m_vBadPixelMap.empty()) //pixel replacement
    {
        aqprintf("Bad pixel correction is under progress. Bad pixel numbers = %d.\n",m_pParent->m_vBadPixelMap.size());
        std::vector<BADPIXELMAP>::iterator it;
        int oriIdx, replIdx;

        for (it = m_pParent->m_vBadPixelMap.begin() ; it != m_pParent->m_vBadPixelMap.end() ; it++)
        {
            BADPIXELMAP tmpData= (*it);
            oriIdx = tmpData.BadPixY * xSize + tmpData.BadPixX;
            replIdx = tmpData.ReplPixY * xSize + tmpData.ReplPixX;
            pImageCorr[oriIdx] = pImageCorr[replIdx];
        }
    }


    /////////////////////***************Manual Correction END***********///////////////////

    /*if(result == HCP_NO_ERR)
      {*/
    int imgSize = xSize*ySize;

    if (m_pParent->m_pCurrImage->IsEmpty() || m_pParent->m_pCurrImage->m_iWidth != xSize || m_pParent->m_pCurrImage->m_iHeight != ySize)
        m_pParent->m_pCurrImage->CreateImage(xSize, ySize, 0);

    for (int i = 0  ; i<size ; i++)
        m_pParent->m_pCurrImage->m_pData[i] = pImageCorr[i];
    //}
    /*else
      {
      aqprintf("*** vip_get_image returned error %d\n", result);
      }*/

    free(image_ptr);
    free(pImageCorr);


    //if (bImageDuplicationOccurred)
    //	return HCP_SAME_IMAGE_ERROR;

    return HCP_NO_ERR;
}
