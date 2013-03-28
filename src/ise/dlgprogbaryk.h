#ifndef DLGPROGBARYK_H
#define DLGPROGBARYK_H

#include <QDialog>
#include "ui_DlgProgress.h"

class DlgProgBarYK : public QDialog
{
	Q_OBJECT

public:
	DlgProgBarYK(QWidget *parent);
	~DlgProgBarYK();

	void SetProgVal(int perVal);
	void SetProgVal(int perVal, QString statusMsg);
	int m_curVal;

public slots:
	void FuncForProgressVal();

	//QString m_curStatusMsg;

public:
	Ui::DlgProgressYK ui;
	
};

#endif // DLGPROGBARYK_H
