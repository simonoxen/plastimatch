#include "dlgprogbaryk.h"
#include "aqprintf.h"

DlgProgBarYK::DlgProgBarYK(QWidget *parent)
	: QDialog(parent)
{
	ui.setupUi(this);
	m_curVal = 0;
}

DlgProgBarYK::~DlgProgBarYK()
{
}

void DlgProgBarYK::SetProgVal( int perVal )
{
	m_curVal = perVal;
	ui.progBar->setValue(m_curVal);
	//ui.lbStatusText->setText("Progressing...");
}


void DlgProgBarYK::SetProgVal(int perVal, QString statusMsg )
{
	m_curVal = perVal;
	ui.progBar->setValue(m_curVal);
	ui.lbStatusText->setText(statusMsg);
}

void DlgProgBarYK::FuncForProgressVal()
{
	m_curVal = ui.progBar->value();
	show();
	aqprintf("Progress value changed\n");

	if (m_curVal >= ui.progBar->maximum()-1)//max =100 then 99
	{
		hide(); //redundent hide() call is OK?
		aqprintf("Hided\n");
	}
}